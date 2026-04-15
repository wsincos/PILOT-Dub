import token
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Dict, Optional
from .utils import make_pad_mask
from .codebooks_patterns import CodebooksPatternProvider

import torch
import random
from .sampler import topk_sampling

class VoiceCraftDataProcessor(nn.Module):
    def __init__(
        self,
        pattern_provider: CodebooksPatternProvider,

        n_codebooks: int,
        n_special: int,
        empty_id: int,
        eog_id: int,
        pad_id: int,
        eos_id: int,
        audio_vocab_size: int,
        max_n_spans: int,
        special_first: bool,
        encodec_sr: int,
    ):
        super().__init__()
        self.pattern_provider = pattern_provider
        self.n_codebooks = n_codebooks

        # Token IDs
        self.audio_vocab_size = audio_vocab_size
        self.n_special = n_special
        self.empty_id = empty_id 
        self.eog_id = eog_id
        self.pad_id = pad_id
        self.eos_id = eos_id
        if self.eos_id > 0:
            assert self.eos_id != self.pad_id and self.eos_id != self.empty_id, self.eos_id
        self.register_buffer("eog", torch.full((n_codebooks, 1), self.eog_id, dtype=torch.long))
        self.register_buffer("eos", torch.full((n_codebooks, 1), self.eos_id, dtype=torch.long))
        
        self.max_n_spans = max_n_spans
        self.special_first = special_first
        self.encodec_sr = encodec_sr
        self.audio_mask_span_len = 3
        self.audio_mask_span_max_len = 10
        
    def _rearrange_y(self, 
                     y: torch.Tensor, 
                     mask_intervals: list,
                     )-> list:
        """        
        Args:
            y: [B, K, T] original audio tokens
            mask_intervals: list of mask intervals for each sample in the batch
        """
        rearranged_y = []

        # for mask segments, EOG will be added at the end of every segment
        # for non-mask segments
        
        for cur_y, mask_interval in zip(y, mask_intervals): # Traverse Batch
            non_mask_prompts = cur_y[:, :mask_interval[0]] # [K, T1]
            mask_prompts = cur_y[:, mask_interval[0]:mask_interval[1]]
            mask_prompts = torch.cat([mask_prompts, self.eog], dim=-1) # add EOG at the end of mask segment as placeholder
            # List[Tensor]: [non_mask_prompts, mask_prompts]
            full_prompts = [non_mask_prompts, mask_prompts]
            rearranged_y.append(full_prompts)
        return rearranged_y # [List[Tensor], List[Tensor](len: 2) , ...] (len:B)
    
    def _apply_delay_pattern(self, rearranged_y: list):
        """
        Shift each segment in rearranged_y according to pattern_provider.

        Args:
            rearranged_y: List[List[Tensor]], come from _rearrange_y
        Returns:
            shifted_y: List[List[Tensor]], shifted segments
        """
        # NOTE: T of shifted segments is T+K, not T+K-1(as following), because we insert blank column at the beginning
        shifted_y = []

        for i, segments in enumerate(rearranged_y): # Traverse Batch
            cur_batch_shifted = [] # save the shifted segments for current batch sample
            for seg in segments: # for every Prompt/Target segment
                # seg shape: [K, T_i]

                # pattern_provider need Batch demension
                # unsqueeze: [K, T] -> [1, K, T]
                seg = seg.unsqueeze(0) 

                # res: (inputs, targets, mask)
                # inputs: [B, K, T]
                # targets: [B, K, T + K]
                # mask: [B, 1, T + K]
                T = seg.shape[-1]
                pattern = self.pattern_provider.get_pattern(T)
                res = pattern.build_pattern_sequence(
                    z=seg.contiguous(), 
                    special_token=self.empty_id,
                    keep_only_valid_steps=False # Full length must be retained, not truncated
                )
                # res[0]: [1, K, T + K]
                shifted_seg = res[0].squeeze(0) # [K, T + K]
                cur_batch_shifted.append(shifted_seg)
            shifted_y.append(cur_batch_shifted)
        return shifted_y
            
    
    def _insert_placeholder(self, shifted_y: list):
        """
        Insert Mask token (EOG) at shifted y, and record Mask position and corresponding Embedding index。
        
        inserted_y: [[P1], [M], [T1]]
        mask_positions: [time_index1]
        mask_indices: [M0]

        Args:
            shifted_y: List[List[Tensor]], come from _apply_delay_pattern
        Returns:
            inserted_y: List[List[Tensor]], add some [K, 1] EOG Token
            mask_positions: List[List[int]], Time Step: record position of Mask in the sequence after flatten
            mask_indices: List[List[int]], Records the placeholder index for each span
        """
        inserted_y = []
        mask_positions = [] # record Mask position (Time Index)
        mask_indices = []   # record used Mask Embedding (Vector Index)

        placeholder_token = self.eog
        # placeholder will take the place of Masked segments
        # it will be replaced by corresponding Mask Embedding later at `_embed_y`

        # NOTE: if debug rearranged_y, below code should be changed correspondingly
        for i, segments in enumerate(shifted_y): # segments lens: 2
            cur_inserted_segments = []
            cur_mask_pos = []

            num_masks = max(len(segments) - 1, 0)

            # mask index: [0, 1, ..., max_n_spans-1]
            emb_inds = list(range(self.max_n_spans))
            emb_inds_use = emb_inds[:num_masks] # [0]
            cur_mask_inds = emb_inds_use
            mask_indices.append(cur_mask_inds)

            # 2. insert placeholder tokens
            current_t = 0
            for j in range(len(segments) - 1):
                seg = segments[j]
                cur_inserted_segments.append(seg)
                current_t += seg.shape[1] 
                
                # record new Mask position
                cur_mask_pos.append(current_t)
                # insert placeholder token
                cur_inserted_segments.append(placeholder_token)
                current_t += 1 

            cur_inserted_segments.append(segments[-1])
            inserted_y.append(cur_inserted_segments)
            mask_positions.append(cur_mask_pos)

        # inserted_y: [[P1], [M], [T1]]
        # mask_positions: [time_index1, ...]
        # mask_indices: [M0, ...]
        return inserted_y, mask_positions, mask_indices

    def apply_random_audio_mask(self, y: torch.Tensor, y_lens: torch.Tensor) -> torch.Tensor:
        masked = y.clone()
        if self.audio_mask_span_len <= 0:
            return masked
        for i, y_len in enumerate(y_lens.tolist()):
            span_len = random.randint(self.audio_mask_span_len, self.audio_mask_span_max_len)
            if y_len <= span_len + 2:
                continue
            start = random.randint(1, y_len - span_len - 1)
            masked[i, :, start:start + span_len] = self.empty_id
        return masked

    def prepare_sequence_with_video_tokens(
        self,
        x_emb: torch.Tensor,
        x_lens: torch.Tensor,
        embedded_y: torch.Tensor,
        audio_total_lens: torch.Tensor,
        ref_length_delayed: torch.Tensor,
        video_tokens_emb: Optional[torch.Tensor],
        video_tokens_lens: Optional[torch.Tensor],
        audio_ref_token: torch.Tensor,
        video_len_token: Optional[torch.Tensor],
        target_start_token: torch.Tensor,
        target_end_token: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Sequence layout with optional video-prefix components:
        [TEXT] + [<AUDIO_REF>] + [REF AUDIO]
        + [<VIDEO_LEN_BUCKET>] + [VIDEO TOKENS]
        + [<TARGET_END>] + [<TARGET_START>] + [TARGET AUDIO + EOG]
        """
        B = embedded_y.shape[0]
        seqs = []
        seq_lens = []
        audio_offsets = []
        ref_offsets = []
        ref_lengths = []
        target_offsets = []
        target_lengths = []

        for i in range(B):
            x_len = int(x_lens[i].item())
            ref_len = ref_length_delayed[i].item()
            audio_total_len = int(audio_total_lens[i].item())
            prefix_segments = [
                audio_ref_token[i : i + 1],
                embedded_y[i : i + 1, :ref_len, :],
            ]
            if video_len_token is not None:
                prefix_segments.append(video_len_token[i : i + 1])
            if video_tokens_emb is not None and video_tokens_lens is not None:
                v_len = int(video_tokens_lens[i].item())
                prefix_segments.append(video_tokens_emb[i : i + 1, :v_len, :])
            if target_end_token is not None:
                prefix_segments.append(target_end_token[i : i + 1])
            prefix_segments.append(target_start_token[i : i + 1])
            audio_prefix = torch.cat(
                prefix_segments,
                dim=1,
            )

            target_part = embedded_y[i : i + 1, ref_len:audio_total_len, :]
            seq = torch.cat([x_emb[i:i + 1, :x_len, :], audio_prefix, target_part], dim=1)
            seqs.append(seq)
            seq_lens.append(seq.shape[1])

            audio_offsets.append(x_len + audio_prefix.shape[1])
            ref_offsets.append(x_len + 1)
            ref_lengths.append(ref_len)
            target_offsets.append(x_len + audio_prefix.shape[1])
            target_lengths.append(target_part.shape[1])

        max_len = max(seq_lens)
        padded = embedded_y.new_zeros((B, max_len, embedded_y.shape[-1]))
        for i, seq in enumerate(seqs):
            padded[i, :seq.shape[1], :] = seq

        return {
            "sequence": padded,
            "sequence_lens": torch.LongTensor(seq_lens).to(padded.device),
            "audio_offsets": torch.LongTensor(audio_offsets).to(padded.device),
            "ref_offsets": torch.LongTensor(ref_offsets).to(padded.device),
            "ref_lengths": torch.LongTensor(ref_lengths).to(padded.device),
            "target_offsets": torch.LongTensor(target_offsets).to(padded.device),
            "target_lengths": torch.LongTensor(target_lengths).to(padded.device),
        }

    def build_causal_masks(self, seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = int(seq_lens.max().item())
        attn_mask = torch.triu(
            torch.ones(max_len, max_len, device=seq_lens.device, dtype=torch.bool),
            diagonal=1,
        )
        pad_mask = torch.arange(max_len, device=seq_lens.device)[None, :] >= seq_lens[:, None]
        return pad_mask, attn_mask

    def gather_sequence_outputs(
        self,
        y_out: torch.Tensor,
        offsets: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, dim_model = y_out.shape
        max_len = int(lengths.max().item()) if lengths.numel() > 0 else 0
        gathered = y_out.new_zeros((batch_size, max_len, dim_model))

        for i in range(batch_size):
            length = int(lengths[i].item())
            if length <= 0:
                continue
            start = int(offsets[i].item())
            end = start + length
            gathered[i, :length, :] = y_out[i, start:end, :]
        return gathered

    def select_audio_outputs(
        self,
        y_out: torch.Tensor,
        seq_lens: torch.Tensor,
        x_lens: torch.Tensor,
        ref_length_delayed: torch.Tensor,
        v_lens: torch.Tensor,
        new_y_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove non-audio tokens (<AUDIO_REF>, <VIDEO_LEN>, video tokens) from decoder outputs.
        Returns padded audio-only outputs aligned with embedded_y.
        """
        B, _, D = y_out.shape
        max_audio_len = int(new_y_lens.max().item())
        y_out_audio = y_out.new_zeros((B, max_audio_len, D))

        for i in range(B):
            text_len = int(x_lens[i].item())
            y_len_total = int(seq_lens[i].item()) - text_len
            y_out_i = y_out[i, text_len:text_len + y_len_total, :]
            ref_len = int(ref_length_delayed[i].item())
            v_len = int(v_lens[i].item())

            idx = 0
            idx += 1  # <AUDIO_REF>
            ref_part = y_out_i[idx : idx + ref_len]
            idx += ref_len
            idx += 1  # <VIDEO_LEN>
            idx += v_len  # VIDEO TOKENS
            idx += 1  # <TARGET_START>
            target_part = y_out_i[idx:]

            y_audio = torch.cat([ref_part, target_part], dim=0)
            y_out_audio[i, : y_audio.shape[0], :] = y_audio

        return y_out_audio

    def select_target_outputs(
        self,
        y_out: torch.Tensor,
        x_lens: torch.Tensor,
        ref_length_delayed: torch.Tensor,
        v_lens: torch.Tensor,
        new_y_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select only target-audio outputs (after removing non-audio tokens).
        """
        B, _, D = y_out.shape
        pred_start = ref_length_delayed
        target_lens = new_y_lens - pred_start
        max_target_len = int(target_lens.max().item())
        y_out_target = y_out.new_zeros((B, max_target_len, D))

        for i in range(B):
            text_len = int(x_lens[i].item())
            ref_len = int(ref_length_delayed[i].item())
            v_len = int(v_lens[i].item())
            target_len = int(target_lens[i].item())

            start = text_len + 1 + ref_len + 1 + v_len + 1
            end = start + target_len
            if target_len <= 0:
                continue
            y_out_target[i, :target_len, :] = y_out[i, start:end, :]

        return y_out_target

    def _batchify_samples(self, inserted_y: list):
        """
        concatenate segments, and do Batch Padding.
        
        Args:
            inserted_y: List[List[Tensor]], come from _insert_placeholder
        Returns:
            cated_y: [B, K, Max_T]
            output_y_lens: [B]
        """
        cated_y = []
        new_y_lens = []

        # 1. concat (Prompt + Placeholders + Targets)
        for i, segments in enumerate(inserted_y):
            # segments: [K, T_seg]
            sample_y = torch.cat(segments, dim=1) # [K, Total_T]
            sample_y = sample_y.transpose(0, 1)  # [Total_T, K]

            cated_y.append(sample_y)
            new_y_lens.append(sample_y.shape[0])

        # 2. Batch Padding
        # pad_sequence input: list of [T, K], output: [Max_T, B, K]
        padded_y = nn.utils.rnn.pad_sequence(
            cated_y, 
            batch_first=False, 
            padding_value=self.pad_id
        )
        
        # 3. adjust [Max_T, B, K] -> [B, K, T]
        cated_y = padded_y.permute(1, 2, 0).contiguous()
        #convert lengths
        output_y_lens = torch.LongTensor(new_y_lens).to(cated_y.device)
        return cated_y, output_y_lens
    
    def _embed_y(self, 
                 cated_y: torch.Tensor, 
                 audio_embedding: nn.Module,
                 ):
        """
        1. get Audio Embedding.
        2. add Embeddings of all codebooks.
        
        Args:
            cated_y: [B, K, T] (from _batchify_samples)
        Returns:
            embedded_y: [B, T, D] convert codebook into Embeddings
        """
        # 1. Lookup Audio Embeddings
        embedded_y = audio_embedding(cated_y) # [B, K, T] -> [B, K, T, D]
        
        # 2. Summation over codebooks
        # [B, K, T, D] -> [B, T, D] 
        embedded_y = embedded_y.sum(dim=1)

        return embedded_y

    def prepare_masks(self, x, x_lens, y_emb, y_lens):
        """
        concatenate text token x and audio token y, prepare padding mask and attention mask for Transformer.
        xy_attn_mask:
        ----------------------+---------------------+
        |  Quadrant 1(X->X)   |  Quadrant 2(X->Y)   |
        |    [x_attn_mask]    |  [all True:  Block] |
        | Text Autoregression | text can't see audio|
        |     [T_x, T_x]      |     [T_x, T_y]      |
        ----------------------+---------------------+
        |  Quadrant 3(Y->X)   |  Quadrant 4(Y->Y)   |
        |[all False: visiable]|   [y_attn_mask]     |
        | audio can see text  | Audio Autoregression|
        |     [T_y, T_x]      |     [T_y, T_y]      |
        ----------------------+---------------------+


        Args:
            x: [B, T_x, D] text embeddings
            x_lens: [B] lengths of text tokens
            y_emb: [B, T_y, D] audio embeddings
            y_lens: [B] lengths of audio tokens
        Returns:
            xy_pad_mask: [B, T_x + T_y] (True for PAD positions)
            xy_attn_mask: [T_x + T_y, T_x + T_y]
        """
        
        device = x.device
        T_x = x.shape[1]
        T_y = y_emb.shape[1]

        # 1. Padding mask
        x_pad_mask = make_pad_mask(x_lens, max_len=T_x).to(device)  # [B, T_x]
        y_pad_mask = make_pad_mask(y_lens, max_len=T_y).to(device)  # [B, T_y]
        xy_pad_mask = torch.cat([x_pad_mask, y_pad_mask], dim=1)  # [B, T_x + T_y]

        # 2. Attention Mask
        # diagonal=1 means True for all above the main diagonal, False for the main diagonal and below.
        
        # x_attn_mask = torch.triu(torch.ones(T_x, T_x, device=device), diagonal=1).bool()
        # y_attn_mask = torch.triu(torch.ones(T_y, T_y, device=device), diagonal=1).bool()

        # x_part = torch.nn.functional.pad(x_attn_mask, (0, T_y), value=True) # [T_x, T_x + T_y]
        # y_part = torch.nn.functional.pad(y_attn_mask, (T_x, 0), value=False) # [T_y, T_x + T_y]
        # xy_attn_mask = torch.cat([x_part, y_part], dim=0)
        xy_attn_mask = torch.triu(torch.ones(T_x+T_y, T_x+T_y, device=device), diagonal=1).bool()

        return xy_pad_mask, xy_attn_mask

    def _remove_mask(self, 
                     logits: torch.Tensor, 
                     mask_positions: list, 
                     new_y_lens: list):
        """
        remove Mask token from prediction results.

        Args:
            logits: [B, K, T, card]
            mask_positions: List[List[int]], Time Step of MASK
            new_y_lens: List[int], lengths of y after inserting Mask tokens
        Returns:
            logits_use: List[List[Tensor]], segments without Mask tokens
        """
        # logits: [B, K, S, card]
        logits_use = []
        for i in range(len(logits)):
            # 1. Recover all valid intervals for this sample
            #   [0, m1), [m1+1, m2), [m2+1, end) ...
            positions = [-1] + mask_positions[i] + [new_y_lens[i]]
            intervals = []
            for j in range(len(positions) - 1):
                start = positions[j] + 1
                end = positions[j+1]
                intervals.append((start, end))

            # 2. slice logits according to intervals
            cur_logits_segments = []
            for start, end in intervals:
                # logits[i]: [K, T, card] -> slice -> [K, T_seg, card]
                seg = logits[i, :, start:end, :]
                cur_logits_segments.append(seg)
            logits_use.append(cur_logits_segments) # List of segments for each sample
        return logits_use

    def _revert_delay_sequence(self, sequence, special_token: int, keep_only_valid_steps: bool = False):
        """
        Revert the delay pattern applied to the sequence.
        NOTE: if original_sequence is [K, T], can use [[original_sequence]] or original_sequence.view(1, 1, K, T) as input.
              By analogy, original_sequence with [B, K, T] can also use original_sequence.unsqueeze(1) or original_sequence.unsqueeze(0) as input.

              return shape is List[List[Tensor]] with shape [K, T_original] or [K, T_reverted]

        Args:
            sequence: List
            special_token: int, token used for padding
            keep_only_valid_steps: bool, whether to keep only valid steps after reverting
        Returns:
            reverted_sequence: [B, K, T_original] or [B, K, T_reverted]
        """
        sequence_last = []
        for i, segments in enumerate(sequence): # for every sample in the batch
            cur_sample_reverted = []
            
            for j, seg in enumerate(segments): # for every segment
                # seg: [K, T_shifted]
                T_original = seg.shape[-1] - self.n_codebooks + 1
                special_col = torch.full((self.n_codebooks, 1), special_token, device=seg.device, dtype=seg.dtype)
                seg = torch.cat([special_col, seg], dim=-1)

                pattern = self.pattern_provider.get_pattern(T_original)

                seg = seg.unsqueeze(0) # [1, K, T_shifted]
                # revert_pattern_sequence: (values, indexes, mask)
                res = pattern.revert_pattern_sequence(
                    seg, 
                    special_token,
                    keep_only_valid_steps=keep_only_valid_steps
                )
                reverted = res[0].squeeze(0) # [K, T_original] or [K, T_reverted]
                cur_sample_reverted.append(reverted)
            sequence_last.append(cur_sample_reverted)
        return sequence_last

    def _revert_delay_logits(self, logits_use: list, rearranged_y: list):
        """
        Args:
            logits_use: outputs of _remove_mask
            rearranged_y: original Target segments before shifting
        """
        logits_final = []

        for i, segments in enumerate(logits_use):
            cur_sample_logits = []
            target_segments = rearranged_y[i]

            for j, seg_logit in enumerate(segments):
                # seg_logit: [K, T_shifted, card]
                # target_seg: [K, T_original]
                target_len = target_segments[j].shape[1]
                pattern = self.pattern_provider.get_pattern(target_len)
                
                # 2. revert
                inp = seg_logit.unsqueeze(0).permute(0, 3, 1, 2).contiguous() # [1, card, K, T_shifted]
                # revert_pattern_logits: (values, indexes, mask)
                res = pattern.revert_pattern_logits(
                    inp, 
                    0, # no special_token
                    keep_only_valid_steps=False
                )
                reverted = res[0].permute(0, 2, 3, 1).squeeze(0) # [1, K, T, carb]
                cur_sample_logits.append(reverted)
            logits_final.append(cur_sample_logits)
        return logits_final
        

    def prepare_audio_input(self, 
                            y: torch.Tensor, 
                            y_lens: torch.Tensor,
                            split_lens: List[int],
                            audio_embedding: nn.Module,
                            ):
        """
        Prepare training data with masked audio tokens.

        Args:
            y: [B, K, T] original audio tokens
            y_lens: [B] lengths of original audio tokens
        Returns:
            embedded_y: [B, T_new, D] embedded audio tokens
            new_y_lens: [B] lengths of new audio tokens
            rearranged_y: List[List[Tensor]], rearranged original audio tokens for loss calculation
        """
        # 1. prepare mask intervals
        mask_intervals = []
        for i, y_len in enumerate(y_lens):
            split_len = split_lens[i]
            mask_intervals.append([split_len, y_len])
        # 2. rearrange y
        rearranged_y = self._rearrange_y(y, mask_intervals) # for each sample: [[P1], [T1]]
        # 3. shift y
        shifted_y = self._apply_delay_pattern(rearranged_y)
        # 4. concatenate and pad
        cated_y, new_y_lens = self._batchify_samples(shifted_y) 
        # 5. embed y
        embedded_y = self._embed_y(cated_y, audio_embedding)

        return {
            "embedded_y": embedded_y,
            "new_y_lens": new_y_lens,
            "rearranged_y": rearranged_y
        }    

    def build_shifted_target_tokens(
        self,
        rearranged_y: list,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build delayed-pattern token ids for target segments only.

        Args:
            rearranged_y: List[List[Tensor]] from prepare_audio_input, where the last segment is target.

        Returns:
            shifted_targets: [B, K, T_shifted_target]
            shifted_target_lens: [B]
        """
        shifted_targets = []
        shifted_lens = []
        max_len = 0

        for segments in rearranged_y:
            target_seg = segments[-1].unsqueeze(0)  # [1, K, T]
            target_len = target_seg.shape[-1]
            pattern = self.pattern_provider.get_pattern(target_len)
            res = pattern.build_pattern_sequence(
                z=target_seg.contiguous(),
                special_token=self.empty_id,
                keep_only_valid_steps=False,
            )
            shifted_target = res[0].squeeze(0)  # [K, T + K]
            shifted_targets.append(shifted_target)
            shifted_lens.append(shifted_target.shape[-1])
            max_len = max(max_len, shifted_target.shape[-1])

        if max_len == 0:
            empty = torch.zeros((len(rearranged_y), self.n_codebooks, 0), dtype=torch.long)
            return empty, torch.zeros((len(rearranged_y),), dtype=torch.long)

        padded = torch.full(
            (len(rearranged_y), self.n_codebooks, max_len),
            fill_value=self.empty_id,
            dtype=shifted_targets[0].dtype,
            device=shifted_targets[0].device,
        )
        for i, shifted_target in enumerate(shifted_targets):
            padded[i, :, : shifted_target.shape[-1]] = shifted_target

        return padded, torch.tensor(shifted_lens, dtype=torch.long, device=padded.device)


    def prepare_video_input(self, v, v_lens=None):
        device = v.device

        if len(v.shape) == 3 and v.shape[-1] == 1024:
            if v.shape[1] > 1:
                v_upsampled = v.transpose(1, 2)
                v_upsampled = F.interpolate(v_upsampled, scale_factor=2, mode="linear", align_corners=False)
                v_upsampled = v_upsampled.transpose(1, 2)
            else:
                v_upsampled = v.repeat_interleave(2, dim=1)

            if v_lens is None:
                new_lens = torch.full(
                    (v.shape[0],), v_upsampled.shape[1], device=device, dtype=torch.long
                )
            else:
                new_lens = v_lens * 2
            return new_lens, v_upsampled

        raise ValueError(
            f"Expected pre-extracted lip_feature [B, T, 1024], got {tuple(v.shape)}. "
            "Please convert mp4 to npy features first and load them in the dataset."
        )

    def decoder_forward(self, 
                        decoder: nn.Module, 
                        xy_input: torch.Tensor, 
                        x_lens: torch.Tensor,
                        xy_attn_mask: torch.Tensor,
                        xy_pad_mask: torch.Tensor,
                        past = None,
                        last_n_tokens: int = 3,
                        ):
        """
        Forward pass through the decoder with prepared masks.
        """
        if past is None:
            output = decoder(
                xy_input,
                mask=xy_attn_mask,
                src_key_padding_mask=xy_pad_mask
            )
            return output, None
        if past.ndim > 3: # uses kvcache, only need to pass the last tokens, this doesn't work with multi-span speech editing yet
            q_len = xy_input.shape[1] 
            
            xy_input = xy_input[:, -last_n_tokens: , :] # [B, last_n_tokens, D]
            xy_attn_mask = xy_attn_mask[-last_n_tokens:, :] # [last_n_tokens_quire, total_keys]
            
        # # decoder returns：
        # if is_src_tuple: x = (x, stage_embedding) -> return: (tensor, None)
        # if present != None: x = [x, present] -> return: [(tensor, None), present] (real outputs)
        out, present = decoder(
            (xy_input, None), 
            mask=xy_attn_mask, 
            src_key_padding_mask=xy_pad_mask,
            past=past
        )
        if isinstance(out, tuple): # get rid of stage_embedding
            out = out[0]

        if out.shape[1] > x_lens.max(): # the first pass, not kvcache yet
            return out, present
        else: # used kvcache
            return out, present

    def post_process_logits(self, 
                            logits: torch.Tensor, 
                            mask_positions: list, 
                            new_y_lens: list,
                            rearranged_y: list):
        """
        Post-process logits to remove mask tokens and revert pattern shifting.

        Args:
            logits: [B, K, T, card]
            mask_positions: List[List[int]], Time Step of MASK
            new_y_lens: List[int], lengths of y after inserting Mask tokens
            rearranged_y: List[List[Tensor]], rearranged original audio tokens for loss calculation
        Returns:
            aligned_logits: [K, T_of_all_Batch, card]
            aligned_targets: [K, T_of_all_Batch]
        """
        # 1. remove mask tokens
        logits_use = self._remove_mask(logits, mask_positions, new_y_lens)

        # 2. revert pattern shifting
        logits_final = self._revert_delay_logits(logits_use, rearranged_y) # List[List[Tensor]]  List[Tensor]: [K, T_seg, card]

        flat_logits_list = []
        flat_targets_list = []

        assert len(logits_final) == len(rearranged_y), "Batch size mismatch"

        for i in range(len(logits_final)): # for every batch
            # for j in range(len(logits_final[i])): # for every segment
            #     flat_logits_list.append(logits_final[i][j]) 
            #     flat_targets_list.append(rearranged_y[i][j])

            # change loss calculation, we don't need to calculate loss for these existing tokens
            # only calculate loss for the target segments
            flat_logits_list.append(logits_final[i][-1]) # only target segments
            flat_targets_list.append(rearranged_y[i][-1]) # only target segments
        
        
        # flat_logits_list: List of [K, T_i, card], 
        # len(flat_logits_list) = num(segments in batch1) + num(segments in batch2) + ...

        aligned_logits = torch.cat(flat_logits_list, dim=1) # [K, T_of_all_Batch, card]
        aligned_targets = torch.cat(flat_targets_list, dim=1) # [K, T_of_all_Batch]

        return aligned_logits, aligned_targets

    def post_process_target_logits(
        self,
        logits: torch.Tensor,
        rearranged_y: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Post-process logits for target segment only (no mask removal).
        """
        target_segments = [segments[-1] for segments in rearranged_y]
        logits_use = []
        for i, target_seg in enumerate(target_segments):
            expected_len = target_seg.shape[1] + self.n_codebooks
            cur_logits = logits[i]
            if cur_logits.shape[1] < expected_len:
                raise ValueError(
                    f"Target logits shorter than expected: logits_len={cur_logits.shape[1]}, "
                    f"expected_len={expected_len}"
                )
            cur_logits = cur_logits[:, :expected_len, :]
            logits_use.append([cur_logits])
        target_wrapped = [[seg] for seg in target_segments]

        logits_final = self._revert_delay_logits(logits_use, target_wrapped)
        flat_logits_list = []
        flat_targets_list = []
        for i in range(len(logits_final)):
            flat_logits_list.append(logits_final[i][0])
            flat_targets_list.append(target_segments[i])

        aligned_logits = torch.cat(flat_logits_list, dim=1)
        aligned_targets = torch.cat(flat_targets_list, dim=1)
        return aligned_logits, aligned_targets

    def decode_target_predictions(
        self,
        logits: torch.Tensor,
        rearranged_y: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode target-segment predictions back to the original real-time axis.

        Returns:
            pred_tokens: [B, K, T_max]
            pred_lengths: [B]
        """
        target_segments = [segments[-1] for segments in rearranged_y]
        logits_use = []
        target_wrapped = []
        pred_lengths = []
        for i, target_seg in enumerate(target_segments):
            expected_len = target_seg.shape[1] + self.n_codebooks
            cur_logits = logits[i]
            if cur_logits.shape[1] < expected_len:
                raise ValueError(
                    f"Target logits shorter than expected: logits_len={cur_logits.shape[1]}, "
                    f"expected_len={expected_len}"
                )
            cur_logits = cur_logits[:, :expected_len, :]
            logits_use.append([cur_logits])
            target_wrapped.append([target_seg])
            pred_lengths.append(target_seg.shape[1])

        logits_final = self._revert_delay_logits(logits_use, target_wrapped)
        max_len = max(pred_lengths) if pred_lengths else 0
        pred_tokens = logits.new_zeros((len(target_segments), self.n_codebooks, max_len), dtype=torch.long)
        for i in range(len(logits_final)):
            pred = logits_final[i][0].argmax(dim=-1)
            pred_tokens[i, :, : pred.shape[1]] = pred
        pred_lengths = torch.tensor(pred_lengths, device=logits.device, dtype=torch.long)
        return pred_tokens, pred_lengths

    def decode_target_q0_reprs(
        self,
        codebook_reprs: torch.Tensor,
        rearranged_y: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode continuous codebook representations back to the original real-time
        target axis and extract q0 only.

        Args:
            codebook_reprs: [B, T_shifted, K, D_repr]
            rearranged_y: target segment is stored in segments[-1]

        Returns:
            q0_reprs: [B, T_max, D_repr]
            q0_lengths: [B]
            q0_tokens: [B, T_max]
        """
        if codebook_reprs.ndim != 4:
            raise ValueError(
                f"Expected codebook_reprs shape [B,T,K,D], got {tuple(codebook_reprs.shape)}"
            )

        target_segments = [segments[-1] for segments in rearranged_y]
        batch_size = len(target_segments)
        repr_dim = codebook_reprs.shape[-1]
        max_len = max((seg.shape[1] for seg in target_segments), default=0)
        device = codebook_reprs.device

        q0_reprs = codebook_reprs.new_zeros((batch_size, max_len, repr_dim))
        q0_tokens = torch.full(
            (batch_size, max_len),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        q0_lengths = []

        codebook_reprs = codebook_reprs.permute(0, 2, 1, 3).contiguous()  # [B, K, T, D]

        for i, target_seg in enumerate(target_segments):
            target_len = target_seg.shape[1]
            expected_len = target_len + self.n_codebooks
            cur_repr = codebook_reprs[i, :, :expected_len, :]  # [K, T_shifted, D]

            inp = cur_repr.unsqueeze(0).permute(0, 3, 1, 2).contiguous()  # [1, D, K, T_shifted]
            pattern = self.pattern_provider.get_pattern(target_len)
            res = pattern.revert_pattern_logits(
                inp,
                0,
                keep_only_valid_steps=False,
            )
            reverted = res[0].permute(0, 2, 3, 1).squeeze(0)  # [K, T, D]
            q0 = reverted[0]
            q0_reprs[i, :target_len, :] = q0
            q0_tokens[i, :target_len] = target_seg[0].to(device=device, dtype=torch.long)
            q0_lengths.append(target_len)

        q0_lengths = torch.tensor(q0_lengths, device=device, dtype=torch.long)
        return q0_reprs, q0_lengths, q0_tokens

    def post_process_ref_logits(
        self,
        logits: torch.Tensor,
        rearranged_y: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Post-process logits for reference segment only (no mask removal).
        """
        ref_segments = [segments[0] for segments in rearranged_y]
        logits_use = []
        valid_indices = []
        for i, ref_seg in enumerate(ref_segments):
            if ref_seg.shape[1] == 0:
                continue
            expected_len = ref_seg.shape[1] + self.n_codebooks
            cur_logits = logits[i]
            if cur_logits.shape[1] < expected_len:
                raise ValueError(
                    f"Reference logits shorter than expected: logits_len={cur_logits.shape[1]}, "
                    f"expected_len={expected_len}"
                )
            cur_logits = cur_logits[:, :expected_len, :]
            logits_use.append([cur_logits])
            valid_indices.append(i)

        if not logits_use:
            empty_logits = logits.new_zeros((self.n_codebooks, 0, logits.shape[-1]))
            empty_targets = torch.zeros((self.n_codebooks, 0), device=logits.device, dtype=torch.long)
            return empty_logits, empty_targets

        ref_wrapped = [[ref_segments[i]] for i in valid_indices]
        logits_final = self._revert_delay_logits(logits_use, ref_wrapped)
        flat_logits_list = []
        flat_targets_list = []
        for j, i in enumerate(valid_indices):
            flat_logits_list.append(logits_final[j][0])
            flat_targets_list.append(ref_segments[i])

        aligned_logits = torch.cat(flat_logits_list, dim=1)
        aligned_targets = torch.cat(flat_targets_list, dim=1)
        return aligned_logits, aligned_targets
    
    def get_empty_embedding(self, audio_embedding: nn.Module):
        """
        Get the embedding vector for the empty token.

        Returns:
            empty_emb: [1, 1, D]
        """
        # 1. Construct input tensor: [1, K, 1] filled with empty_id
        # Batch=1, K=n_codebooks, T=1
        device = next(audio_embedding.parameters()).device
        empty_ids = torch.full((1, self.n_codebooks, 1), self.empty_id, dtype=torch.long, device=device)
        
        # 2. Forward through AudioEmbedding: 
        emb = audio_embedding(empty_ids) # [1, K, 1, D]
        
        # 3. Sum over codebooks (dim=1): [1, 1, D]
        empty_emb = emb.sum(dim=1)
        return empty_emb
    
    def sample_codebooks_token(self, n_eog, logits, codebook_done, top_k, top_p, temperature, prev_token, 
                               consec_silence_count, stop_repetition, silence_tokens, curr_y_len, x_len,
                               step, total_generate_len):
        
        # for TTS task, EOS(if exist) is used for ending generation
        # end_token = self.eos_id if self.eos_id>0 else self.eog_id
        end_token = self.eog_id
        
        # if self.eos_id > 0: # for TTS inference, if we select end_token=eos_id, EOG should not be generated
        #     logits[:, self.eog_id] = -1e9
        logits[:, self.eos_id] = -1e9
        
        if step < total_generate_len: # not the last step, cannot generate end_token
            logits[:, self.empty_id] = -1e9
            logits[:, self.pad_id] = -1e9
            logits[:, end_token] = -1e9
        elif step == total_generate_len: # the last step, must generate end_token
            logits[0, end_token] = 1e9

        if n_eog == 0:
            # # the following code is contained in the above code
            # logits: [K, card]
            # for k in range(1, len(codebook_done)):
            #     logits[k, end_token] = -1e9 # other codebooks cannot generate end_token
            #     logits[k, self.empty_id] = -1e9
            # if step <= self.encodec_sr // 5:
            #     logits[:, end_token] = -1e9

            # if satisfy three conditions, punish the logits to avoid repetition generate silence token by reduce its probability
            # stop_repetition: user defined threshold, if > 0, means enable this feature
            # prev_token in silence_tokens: previous generated token is a silence token
            # consec_silence_count > stop_repetition: the number of consecutive previous silence tokens exceed the threshold
            if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                if logits[0, prev_token] < 0:
                    logits[0, prev_token] *= (consec_silence_count - stop_repetition +1)
                else:
                    logits[0, prev_token] /= (consec_silence_count - stop_repetition +1)

            samples = topk_sampling(logits, top_k=top_k, top_p=top_p, temperature=temperature)  # [K, 1]
            # for the first several generation tokens, only partial codebooks are useful(delay pattern)
            # we should pad the unused codebooks with empty token
            if step < self.n_codebooks - 1:
                for idx in range(1, self.n_codebooks - step):
                    samples[-idx, 0] = self.empty_id
            
            token_0 = samples[0, 0].item()

            if (token_0 == end_token or 
                torch.argmax(logits[0], dim=-1) == end_token or
                curr_y_len > x_len * (self.encodec_sr // 5)
            ):
                codebook_done[0] = True
            
            # silence token handling
            if token_0 in silence_tokens and token_0 == prev_token:
                consec_silence_count += 1
            else:
                consec_silence_count = 0
            prev_token = token_0

        else:
            # n_eog > 0
            for i in range(n_eog+1, self.n_codebooks):
                logits[i, end_token] = -1e9
                logits[i, self.empty_id] = -1e9
            
            samples = topk_sampling(logits, top_k=top_k, top_p=top_p, temperature=temperature)  # [K, 1]

            for i in range(n_eog):
                samples[i, 0] = self.empty_id # codebooks before n_eog are padded with empty token
            
            # if codebook1 generate EOG, then other codebooks should also generate EOG
            samples[n_eog, 0] = end_token
            codebook_done[n_eog] = True
        
        return samples, codebook_done, prev_token, consec_silence_count



    def sample_codebooks_token_batch(self, n_eog, logits, codebook_done, top_k, top_p, temperature, prev_tokens, 
                               consec_silence_counts, stop_repetition, silence_tokens, curr_y_lens, x_lens,
                               step, total_generate_len, keep):
        
        # for TTS task, EOS(if exist) is used for ending generation
        # end_token = self.eos_id if self.eos_id > 0 else self.eog_id
        end_token = self.eog_id
        batch_size = logits.shape[0]
        
        # if self.eos_id > 0: # for TTS inference, if we select end_token=eos_id, EOG should not be generated
        #     logits[:, :, self.eog_id] = -1e9
        logits[:, :, self.eos_id] = -1e9
        
        # TODO: 这里应该是存在一个bug，原文是在尝试多个batch去做并行生成，但是由于video的长度限制，所以需要固定eos的时间
        # 他这里强制设定了在total_generate_len的时候生成end_token，那么所有batch都会在这个时候生成end，而keep只会保留最后一个
        # 所以这里并行就没有任何意义了
        if step < total_generate_len:
            logits[:, :, self.empty_id] = -1e9
            logits[:, :, self.pad_id] = -1e9
            logits[:, :, end_token] = -1e9
        elif step == total_generate_len:
            logits[:, 0, end_token] = 1e9
                

        if n_eog == 0:
            ## the following code is contained in the above code, it is redundant
            # # logits: [K, card]
            # for k in range(1, len(codebook_done)): # for the after codebooks, they cannot generate end_token and empty_token
            #     logits[:, k, end_token] = -1e9
            #     logits[:, k, self.empty_id] = -1e9
            # if step <= self.encodec_sr // 5:
            #     logits[:, :, end_token] = -1e9

            # if satisfy three conditions, punish the logits to avoid repetition generate silence token by reduce its probability
            # stop_repetition: user defined threshold, if > 0, means enable this feature
            # prev_token in silence_tokens: previous generated token is a silence token
            # consec_silence_count > stop_repetition: the number of consecutive previous silence tokens exceed the threshold
            for b in range(batch_size):
                prev_token = prev_tokens[b]
                consec_silence_count = consec_silence_counts[b]
                if stop_repetition > 0 and prev_token in silence_tokens and consec_silence_count > stop_repetition:
                    if logits[b, 0, prev_token] < 0:
                        logits[b, 0, prev_token] *= (consec_silence_count - stop_repetition +1)
                    else:
                        logits[b, 0, prev_token] /= (consec_silence_count - stop_repetition +1)

            samples = topk_sampling(
                logits.reshape(batch_size * self.n_codebooks, logits.shape[-1]), 
                top_k=top_k, top_p=top_p, temperature=temperature
            )  # [B*K, 1]
            samples = samples.reshape(batch_size, self.n_codebooks, 1)  # [B, K, 1]
            
            for b in range(batch_size):
                # for the first several generation tokens, only partial codebooks are useful(delay pattern)
                # we should pad the unused codebooks with empty token
                if step < self.n_codebooks - 1:
                    for idx in range(1, self.n_codebooks - step):
                        samples[b, -idx, 0] = self.empty_id
                token_0 = samples[b, 0, 0].item()
                if (token_0 == end_token or 
                    torch.argmax(logits[b, 0], dim=-1) == end_token or
                    curr_y_lens[b] > x_lens[b] * (self.encodec_sr // 5)
                    ): 
                    samples[b, 0, 0] = end_token
                    codebook_done[0] = True
                    keep = b
                
            
                # silence token handling
                if token_0 in silence_tokens and token_0 == prev_tokens[b]:
                    consec_silence_count += 1
                else:
                    consec_silence_count = 0
                prev_tokens[b] = token_0
            

        else: # n_eog > 0
            for i in range(n_eog+1, self.n_codebooks):
                logits[:, i, end_token] = -1e9
                logits[:, i, self.empty_id] = -1e9
            
            samples = topk_sampling(
                logits.reshape(batch_size * self.n_codebooks, logits.shape[-1]),
                top_k=top_k, top_p=top_p, temperature=temperature)  # [B*K, 1]
            samples = samples.reshape(batch_size, self.n_codebooks, 1)

            for i in range(n_eog):
                samples[keep, i, 0] = self.empty_id
            
            # if codebook1 generate end_token, then other codebooks should also generate EOG
            samples[keep, n_eog, 0] = end_token
            codebook_done[n_eog] = True
        
        return samples, codebook_done, prev_tokens, consec_silence_counts, keep


    def prepare_generate_inputs(
            self,
            y: torch.Tensor,
            v: torch.Tensor,
            audio_embedding: nn.Module,
        ) -> Dict:
        assert y.dim() == 3, "y must be [B, K, T]. Got {y.shape}"
        if self.special_first:
            y = y + int(self.n_special)
        
        y_len = y.shape[2]
        y_lens = torch.LongTensor([y_len]).to(y.device)

        if v.dim() == 2:
            v = v.unsqueeze(0)
        if v.dim() != 3:
            raise ValueError(f"Expected v as [B, T, C] or [T, C], got {tuple(v.shape)}")
        v_lens = torch.LongTensor([v.shape[1]]).to(y.device)
        new_lens, v_upsampled = self.prepare_video_input(v, v_lens=v_lens)
        
        mask_intervals = []
        # For inference, treat the target span as "masked" to append only EOG placeholder.
        # This keeps the prompt segment length consistent with training delay pattern.
        mask_intervals.append([y_lens[0], y_lens[0] + new_lens[0]]) # only one sample in batch
        # NOTE: mask_interval length is larger than original y_len, so the mask segment will only contain self.eog
        rearranged_y = self._rearrange_y(y, mask_intervals) # [[P: 136], [eog: 1]]
        rearranged_y = [[segments[0]] for segments in rearranged_y]
        shifted_y = self._apply_delay_pattern(rearranged_y) # [[P: 140]]
        cated_y, new_y_lens = self._batchify_samples(shifted_y) # [1, 4, 140]


        # Keep full delayed prompt length (T + n_codebooks) so ref_length_delayed
        # matches training and aligns with <VIDEO_LEN>/<VIDEO TOKENS>/<TARGET_START>.

        embedded_y = self._embed_y(cated_y, audio_embedding)
        # [Prompts]
        # v_input = v_upsampled

        return {
            "embedded_y": embedded_y,
            "new_y_lens": new_y_lens,
            "rearranged_y": rearranged_y,
            "v_lens": new_lens,
            "v_upsampled": v_upsampled,
        }
    

    def reconstruct_sequence(self,
                             y_sample,
                             generated_span):
        """
        Reconstruct the full sequence from generated span for TTS task.
        Args:
            y_sample: [B, K, T_final] original audio tokens
            generated_span: [K, T_span] or [B, K, T_span] generated span
        Returns:
            res: [K, T_final] reconstructed sequence
        """
        if generated_span.dim() == 2:
            unshifted_spans = self._revert_delay_sequence([[generated_span]], self.empty_id)[0][0]
            unshifted_spans = unshifted_spans.unsqueeze(0) # [1, K, T_span]
        elif generated_span.dim() == 3:
            unshifted_spans = self._revert_delay_sequence(generated_span.unsqueeze(0), self.empty_id)[0]
            unshifted_spans = torch.stack([u for u in unshifted_spans], dim=0) # [B, K, T_span]
        else:
            raise ValueError("generated_span must be [K, T_span] or [B, K, T_span]")
            # check if there any tokens id >= audio_vocab_sizes
        assert torch.all(unshifted_spans < self.audio_vocab_size), "Generated token id exceed audio vocab size"

        res = torch.cat([y_sample, unshifted_spans], dim=-1) # [B, K, T_final]
        if self.special_first:
            res = res - int(self.n_special)
            unshifted_spans = unshifted_spans - int(self.n_special)
        return res, unshifted_spans
