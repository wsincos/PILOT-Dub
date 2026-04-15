import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .base import GenerativeAudioModel
from src.modeling.modules.data_processors import VoiceCraftDataProcessor


class VoiceCraftDubModel(GenerativeAudioModel):
    def __init__(
        self,
        text_embedding: nn.Module,
        text_positional_embedding: nn.Module,
        audio_embedding: nn.Module,
        audio_positional_embedding: nn.Module,
        video_tokenizer: nn.Module,
        audio_ref_token: nn.Module,
        video_len_token: Optional[nn.Module],
        target_start_token: nn.Module,
        decoder: nn.Module,
        predict_layer: nn.Module,
        processor: VoiceCraftDataProcessor,
        enable_audio_mask: bool,
        ref_audio_positional_embedding: Optional[nn.Module] = None,
        shared_audio_positional_embedding: Optional[nn.Module] = None,
        ref_audio_segment_embedding: Optional[nn.Module] = None,
        video_segment_embedding: Optional[nn.Module] = None,
        target_audio_segment_embedding: Optional[nn.Module] = None,
        target_end_segment_embedding: Optional[nn.Module] = None,
        text_ref_segment_embedding: Optional[nn.Module] = None,
        text_target_segment_embedding: Optional[nn.Module] = None,
        cross_attention: Optional[nn.Module] = None,
        in_decoder_cross_attention: Optional[nn.Module] = None,
        audio_sync_projector: Optional[nn.Module] = None,
        video_sync_projector: Optional[nn.Module] = None,
        ctc_head: Optional[nn.Module] = None,
        alignment_head: Optional[nn.Module] = None,
        progress_head: Optional[nn.Module] = None,
        boundary_head: Optional[nn.Module] = None,
        plan_encoder: Optional[nn.Module] = None,
        plan_progress_head: Optional[nn.Module] = None,
        plan_boundary_head: Optional[nn.Module] = None,
        plan_acoustic_head: Optional[nn.Module] = None,
        plan_router: Optional[nn.Module] = None,
        hierarchical_codebook_conditioner: Optional[nn.Module] = None,
        q0_controller: Optional[nn.Module] = None,
        q0_planner_state_adapter: Optional[nn.Module] = None,
        sync_codebook_weights: Optional[list[float]] = None,
        use_sync_loss: bool = False,
        use_ctc_loss: bool = False,
        use_alignment_loss: bool = False,
        use_progress_loss: bool = False,
        use_boundary_loss: bool = False,
        use_planner: bool = False,
        use_plan_conditioning: bool = False,
        use_plan_progress_loss: bool = False,
        use_plan_boundary_loss: bool = False,
        use_plan_acoustic_loss: bool = False,
        use_codebook_routing: bool = False,
        use_per_codebook_real_time_routing: bool = False,
        use_hierarchical_codebook_conditioning: bool = False,
        use_q0_controller: bool = False,
        use_q0_planner_state_adapter: bool = False,
        use_history_corruption: bool = False,
        history_corruption_max_rate: float = 0.0,
        history_corruption_warmup_steps: int = 0,
        use_scheduled_sampling: bool = False,
        scheduled_sampling_start_rate: float = 0.0,
        scheduled_sampling_max_rate: float = 0.0,
        scheduled_sampling_warmup_steps: int = 0,
        sync_loss_type: str = "cosine",
        sync_nce_temperature: float = 0.07,
        sync_negative_delta: int = 2,
        sync_video_source: str = "raw_video",
        sync_detach_audio: bool = False,
        sync_detach_video: bool = True,
        use_target_end_token: bool = False,
        use_video_prefix: bool = True,
        use_video_len_token: bool = True,
        use_in_decoder_cross_attention: bool = False,
        in_decoder_cross_attention_num_layers: int = 0,
        text_ref_start_token_id: Optional[int] = None,
        text_target_start_token_id: Optional[int] = None,
        q0_loop_threshold: float = 0.6,
        q0_eos_threshold: float = 0.5,
        q0_loop_logit_penalty: float = 2.0,
        q0_eog_logit_penalty: float = 4.0,
        q0_planner_adapter_ablate_plan_hidden: bool = False,
        q0_planner_adapter_ablate_occurrence: bool = False,
        q0_planner_adapter_ablate_remaining: bool = False,
        q0_planner_adapter_ablate_stop: bool = False,
        q0_planner_adapter_ablate_viseme: bool = False,
        q0_planner_adapter_plan_shift: int = 0,
        q0_planner_adapter_batch_shuffle: bool = False,
        q0_planner_adapter_disable: bool = False,
        use_visual_decode_bias: bool = False,
        visual_decode_prior_path: Optional[str] = None,
        visual_decode_bias_weight: float = 0.0,
        visual_decode_bias_clamp: float = 3.0,
        visual_decode_activity_silence_weight: float = 0.0,
        use_g_embedding_loss: bool = False,
        g_embedding_temperature: float = 0.1,
        g_embedding_codebook_weights: Optional[list[float]] = None,
        g_embedding_detach_targets: bool = True,
        plan_acoustic_temperature: float = 0.07,
        plan_acoustic_negative_delta: int = 1,
        plan_acoustic_target_weights: Optional[list[float]] = None,
        plan_acoustic_detach_targets: bool = True,
    ):
        super().__init__()

        self.text_embedding = text_embedding
        self.text_positional_embedding = text_positional_embedding
        self.audio_embedding = audio_embedding
        self.audio_positional_embedding = audio_positional_embedding
        self.video_tokenizer = video_tokenizer
        self.audio_ref_token = audio_ref_token
        self.video_len_token = video_len_token
        self.target_start_token = target_start_token
        self.decoder = decoder
        self.predict_layer = predict_layer
        self.processor = processor
        self.enable_audio_mask = enable_audio_mask
        self.n_codebooks = len(audio_embedding)

        self.ref_audio_positional_embedding = ref_audio_positional_embedding
        self.shared_audio_positional_embedding = shared_audio_positional_embedding
        self.ref_audio_segment_embedding = ref_audio_segment_embedding
        self.video_segment_embedding = video_segment_embedding
        self.target_audio_segment_embedding = target_audio_segment_embedding
        self.target_end_segment_embedding = target_end_segment_embedding
        self.text_ref_segment_embedding = text_ref_segment_embedding
        self.text_target_segment_embedding = text_target_segment_embedding
        self.cross_attention = cross_attention
        self.use_sync_loss = use_sync_loss
        self.use_ctc_loss = use_ctc_loss
        self.use_alignment_loss = use_alignment_loss
        self.use_progress_loss = use_progress_loss
        self.use_boundary_loss = use_boundary_loss
        self.use_planner = use_planner
        self.use_plan_conditioning = use_plan_conditioning
        self.use_plan_progress_loss = use_plan_progress_loss
        self.use_plan_boundary_loss = use_plan_boundary_loss
        self.use_plan_acoustic_loss = use_plan_acoustic_loss
        self.use_codebook_routing = use_codebook_routing
        self.use_per_codebook_real_time_routing = use_per_codebook_real_time_routing
        self.use_hierarchical_codebook_conditioning = use_hierarchical_codebook_conditioning
        self.use_q0_controller = use_q0_controller
        self.use_q0_planner_state_adapter = use_q0_planner_state_adapter
        # Do not keep auxiliary branches registered in DDP when their losses are disabled.
        # Otherwise DDP sees trainable parameters that never contribute to the loss.
        self.audio_sync_projector = audio_sync_projector if self.use_sync_loss else None
        self.video_sync_projector = video_sync_projector if self.use_sync_loss else None
        self.ctc_head = ctc_head if self.use_ctc_loss else None
        self.alignment_head = alignment_head if self.use_alignment_loss else None
        self.progress_head = progress_head if self.use_progress_loss else None
        self.boundary_head = boundary_head if self.use_boundary_loss else None
        self.plan_encoder = plan_encoder if self.use_planner else None
        self.plan_progress_head = plan_progress_head if self.use_plan_progress_loss else None
        self.plan_boundary_head = plan_boundary_head if self.use_plan_boundary_loss else None
        self.plan_acoustic_head = plan_acoustic_head if self.use_plan_acoustic_loss else None
        self.plan_router = plan_router if self.use_codebook_routing else None
        self.hierarchical_codebook_conditioner = (
            hierarchical_codebook_conditioner if self.use_hierarchical_codebook_conditioning else None
        )
        self.q0_controller = q0_controller if self.use_q0_controller else None
        self.q0_planner_state_adapter = (
            q0_planner_state_adapter if self.use_q0_planner_state_adapter else None
        )
        self.use_history_corruption = use_history_corruption
        self.history_corruption_max_rate = history_corruption_max_rate
        self.history_corruption_warmup_steps = history_corruption_warmup_steps
        self.use_scheduled_sampling = use_scheduled_sampling
        self.scheduled_sampling_start_rate = scheduled_sampling_start_rate
        self.scheduled_sampling_max_rate = scheduled_sampling_max_rate
        self.scheduled_sampling_warmup_steps = scheduled_sampling_warmup_steps
        self.current_train_step = 0
        self.sync_loss_type = sync_loss_type
        self.sync_nce_temperature = sync_nce_temperature
        self.sync_negative_delta = sync_negative_delta
        self.sync_video_source = str(sync_video_source)
        self.sync_detach_audio = bool(sync_detach_audio)
        self.sync_detach_video = bool(sync_detach_video)
        self.use_target_end_token = use_target_end_token
        self.use_video_prefix = use_video_prefix
        self.use_video_len_token = use_video_len_token
        self.use_in_decoder_cross_attention = use_in_decoder_cross_attention
        self.in_decoder_cross_attention_num_layers = in_decoder_cross_attention_num_layers
        self.text_ref_start_token_id = text_ref_start_token_id
        self.text_target_start_token_id = text_target_start_token_id
        self.q0_loop_threshold = q0_loop_threshold
        self.q0_eos_threshold = q0_eos_threshold
        self.q0_loop_logit_penalty = q0_loop_logit_penalty
        self.q0_eog_logit_penalty = q0_eog_logit_penalty
        self.q0_planner_adapter_ablate_plan_hidden = q0_planner_adapter_ablate_plan_hidden
        self.q0_planner_adapter_ablate_occurrence = q0_planner_adapter_ablate_occurrence
        self.q0_planner_adapter_ablate_remaining = q0_planner_adapter_ablate_remaining
        self.q0_planner_adapter_ablate_stop = q0_planner_adapter_ablate_stop
        self.q0_planner_adapter_ablate_viseme = q0_planner_adapter_ablate_viseme
        self.q0_planner_adapter_plan_shift = int(q0_planner_adapter_plan_shift)
        self.q0_planner_adapter_batch_shuffle = q0_planner_adapter_batch_shuffle
        self.q0_planner_adapter_disable = q0_planner_adapter_disable
        self.use_visual_decode_bias = use_visual_decode_bias
        self.visual_decode_bias_weight = float(visual_decode_bias_weight)
        self.visual_decode_bias_clamp = float(visual_decode_bias_clamp)
        self.visual_decode_activity_silence_weight = float(visual_decode_activity_silence_weight)
        self.use_g_embedding_loss = bool(use_g_embedding_loss)
        self.g_embedding_temperature = float(g_embedding_temperature)
        self.g_embedding_detach_targets = bool(g_embedding_detach_targets)
        self.plan_acoustic_temperature = float(plan_acoustic_temperature)
        self.plan_acoustic_negative_delta = int(plan_acoustic_negative_delta)
        self.plan_acoustic_detach_targets = bool(plan_acoustic_detach_targets)
        self.visual_decode_prior_path = visual_decode_prior_path
        if self.use_visual_decode_bias:
            if visual_decode_prior_path is None:
                raise ValueError("visual_decode_prior_path must be set when use_visual_decode_bias=True")
            prior = torch.load(visual_decode_prior_path, map_location="cpu")
            log_bias = prior["log_bias"] if isinstance(prior, dict) else prior
            if log_bias.ndim != 2:
                raise ValueError(f"Expected visual decode prior [C,V], got {tuple(log_bias.shape)}")
            self.register_buffer("visual_decode_log_bias", log_bias.float(), persistent=False)
        else:
            self.visual_decode_log_bias = None
        if sync_codebook_weights is not None:
            self.register_buffer(
                "sync_codebook_weights",
                torch.tensor(sync_codebook_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.sync_codebook_weights = None
        if g_embedding_codebook_weights is not None:
            if len(g_embedding_codebook_weights) != self.n_codebooks:
                raise ValueError(
                    f"Expected {self.n_codebooks} g embedding weights, got {len(g_embedding_codebook_weights)}"
                )
            self.register_buffer(
                "g_embedding_codebook_weights",
                torch.tensor(g_embedding_codebook_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.g_embedding_codebook_weights = None
        if plan_acoustic_target_weights is not None:
            if len(plan_acoustic_target_weights) != self.n_codebooks:
                raise ValueError(
                    f"Expected {self.n_codebooks} plan acoustic target weights, got {len(plan_acoustic_target_weights)}"
                )
            self.register_buffer(
                "plan_acoustic_target_weights",
                torch.tensor(plan_acoustic_target_weights, dtype=torch.float32),
                persistent=False,
            )
        else:
            self.plan_acoustic_target_weights = None
        self.use_v4 = (
            self.ref_audio_positional_embedding is not None
            and self.shared_audio_positional_embedding is not None
        )
        base_in_decoder_cross_attention = in_decoder_cross_attention
        if (
            self.use_in_decoder_cross_attention
            and base_in_decoder_cross_attention is not None
            and self.in_decoder_cross_attention_num_layers > 0
        ):
            self.in_decoder_cross_attention_layers = nn.ModuleList(
                [copy.deepcopy(base_in_decoder_cross_attention) for _ in range(self.in_decoder_cross_attention_num_layers)]
            )
        else:
            self.in_decoder_cross_attention_layers = nn.ModuleList()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _build_shifted_positions(
        self,
        shifted_lengths: torch.Tensor,
        anchor_lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
        skip_leading_blank: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_len <= 0:
            empty = torch.zeros((shifted_lengths.shape[0], 0), device=device, dtype=torch.long)
            mask = torch.zeros((shifted_lengths.shape[0], 0), device=device, dtype=torch.bool)
            return empty, mask
        base = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
        base = base.expand(shifted_lengths.shape[0], -1)
        if skip_leading_blank:
            base_for_position = torch.clamp(base - 1, min=0)
        else:
            base_for_position = base
        clip_max = torch.clamp(anchor_lengths.to(device=device, dtype=torch.long) - 1, min=0).unsqueeze(1)
        positions = torch.minimum(base_for_position, clip_max)
        valid = base < shifted_lengths.to(device=device).unsqueeze(1)
        if skip_leading_blank:
            valid = valid & (base > 0)
        return positions, valid

    def _add_type_embedding(
        self,
        embeddings: torch.Tensor,
        type_embedding: Optional[nn.Module],
    ) -> torch.Tensor:
        if type_embedding is None:
            return embeddings
        return embeddings + type_embedding(
            batch_size=embeddings.shape[0],
            seq_len=embeddings.shape[1],
            device=embeddings.device,
        )

    def _encode_segment(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        positional_embedding: nn.Module,
        type_embedding: Optional[nn.Module],
        position_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if position_mask is None or bool(position_mask.all().item()):
            encoded = positional_embedding(embeddings, positions=positions)
        else:
            if hasattr(positional_embedding, "position_embeddings"):
                pos_embeddings = positional_embedding.position_embeddings(
                    positions.to(device=embeddings.device, dtype=torch.long)
                )
                pos_embeddings = pos_embeddings * position_mask.unsqueeze(-1).to(embeddings.dtype)
                encoded = embeddings + pos_embeddings
                encoded = positional_embedding.dropout(encoded)
            else:
                encoded_all = positional_embedding(embeddings, positions=positions)
                encoded = embeddings + (encoded_all - embeddings) * position_mask.unsqueeze(-1).to(embeddings.dtype)
        return self._add_type_embedding(encoded, type_embedding)

    def _masked_mean(
        self,
        sequence: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        max_len = sequence.shape[1]
        mask = torch.arange(max_len, device=sequence.device).unsqueeze(0) < lengths.unsqueeze(1)
        masked = sequence * mask.unsqueeze(-1).to(sequence.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
        return masked.sum(dim=1) / denom

    def _extract_target_text_memory(
        self,
        x_emb: torch.Tensor,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_target_start_token_id is None:
            return x_emb, x_lens.to(device=x_emb.device, dtype=torch.long)

        batch_size = x_emb.shape[0]
        target_segments = []
        target_lens = []
        max_len = 0
        seq_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = seq_positions < x_lens.unsqueeze(1)
        target_token_mask = x == int(self.text_target_start_token_id)
        has_target = target_token_mask.any(dim=1)
        target_start_idx = target_token_mask.float().argmax(dim=1)

        for b in range(batch_size):
            if not bool(has_target[b].item()):
                curr_len = int(x_lens[b].item())
                seg = x_emb[b, :curr_len, :]
            else:
                start = int(target_start_idx[b].item())
                curr_valid = valid_mask[b]
                seg_mask = (seq_positions[b] >= start) & curr_valid
                seg = x_emb[b, seg_mask, :]
            target_segments.append(seg)
            target_lens.append(seg.shape[0])
            max_len = max(max_len, seg.shape[0])

        padded = x_emb.new_zeros((batch_size, max_len, x_emb.shape[-1]))
        for b, seg in enumerate(target_segments):
            if seg.shape[0] > 0:
                padded[b, : seg.shape[0], :] = seg
        return padded, torch.tensor(target_lens, device=x_emb.device, dtype=torch.long)

    def _extract_target_text_token_ids(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.text_target_start_token_id is None:
            return x, x_lens.to(device=x.device, dtype=torch.long)

        batch_size = x.shape[0]
        target_segments = []
        target_lens = []
        max_len = 0
        seq_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(batch_size, -1)
        valid_mask = seq_positions < x_lens.unsqueeze(1)
        target_token_mask = x == int(self.text_target_start_token_id)
        has_target = target_token_mask.any(dim=1)
        target_start_idx = target_token_mask.float().argmax(dim=1)

        for b in range(batch_size):
            if not bool(has_target[b].item()):
                curr_len = int(x_lens[b].item())
                seg = x[b, :curr_len]
            else:
                start = int(target_start_idx[b].item())
                curr_valid = valid_mask[b]
                seg_mask = (seq_positions[b] >= start) & curr_valid
                seg = x[b, seg_mask]
            target_segments.append(seg)
            target_lens.append(seg.shape[0])
            max_len = max(max_len, seg.shape[0])

        padded = torch.full((batch_size, max_len), fill_value=0, device=x.device, dtype=torch.long)
        for b, seg in enumerate(target_segments):
            if seg.shape[0] > 0:
                padded[b, : seg.shape[0]] = seg
        return padded, torch.tensor(target_lens, device=x.device, dtype=torch.long)

    def _extract_ref_audio_memory(
        self,
        y_emb: torch.Tensor,
        ref_lengths: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if ref_lengths is None or ref_lengths.numel() == 0:
            return None, None
        max_ref_len = int(ref_lengths.max().item()) if ref_lengths.numel() > 0 else 0
        if max_ref_len <= 0:
            return None, None
        return y_emb[:, :max_ref_len, :], ref_lengths.to(device=y_emb.device, dtype=torch.long)

    def _build_plan_outputs(
        self,
        video_embeddings: Optional[torch.Tensor],
        video_lens: Optional[torch.Tensor],
        text_memory: Optional[torch.Tensor],
        text_lens: Optional[torch.Tensor],
        text_token_ids: Optional[torch.Tensor],
        ref_memory: Optional[torch.Tensor],
        ref_lens: Optional[torch.Tensor],
        text_summary: Optional[torch.Tensor],
        ref_summary: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if (
            not self.use_planner
            or self.plan_encoder is None
            or video_embeddings is None
            or video_lens is None
        ):
            return None
        outputs = self.plan_encoder(
            video_embeddings=video_embeddings,
            video_lens=video_lens.to(device=video_embeddings.device, dtype=torch.long),
            text_memory=text_memory,
            text_lens=text_lens.to(device=video_embeddings.device, dtype=torch.long) if text_lens is not None else None,
            text_token_ids=text_token_ids.to(device=video_embeddings.device, dtype=torch.long) if text_token_ids is not None else None,
            ref_memory=ref_memory,
            ref_lens=ref_lens.to(device=video_embeddings.device, dtype=torch.long) if ref_lens is not None else None,
            text_summary=text_summary,
            ref_summary=ref_summary,
        )
        if isinstance(outputs, dict):
            return outputs
        return {"plan_hidden": outputs}

    def _shift_plan_hidden(
        self,
        plan_hidden: Optional[torch.Tensor],
        shift_steps: int,
    ) -> Optional[torch.Tensor]:
        if plan_hidden is None:
            return None
        shift_steps = int(shift_steps)
        if shift_steps == 0 or plan_hidden.shape[1] == 0:
            return plan_hidden

        max_shift = max(plan_hidden.shape[1] - 1, 0)
        if max_shift <= 0:
            return plan_hidden
        shift_steps = max(min(shift_steps, max_shift), -max_shift)

        shifted = torch.empty_like(plan_hidden)
        if shift_steps > 0:
            shifted[:, :shift_steps, :] = plan_hidden[:, :1, :].expand(-1, shift_steps, -1)
            shifted[:, shift_steps:, :] = plan_hidden[:, :-shift_steps, :]
        else:
            pos_shift = -shift_steps
            shifted[:, -pos_shift:, :] = plan_hidden[:, -1:, :].expand(-1, pos_shift, -1)
            shifted[:, :-pos_shift, :] = plan_hidden[:, pos_shift:, :]
        return shifted

    def _gather_q0_planner_states(
        self,
        q0_repr: torch.Tensor,
        q0_lengths: torch.Tensor,
        plan_hidden: torch.Tensor,
        plan_outputs: Optional[Dict[str, torch.Tensor]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        batch_size, max_q0_len, _ = q0_repr.shape
        plan_len = plan_hidden.shape[1]
        if max_q0_len <= 0 or plan_len <= 0:
            empty_plan = plan_hidden.new_zeros((batch_size, max_q0_len, plan_hidden.shape[-1]))
            return {
                "plan_state": empty_plan,
                "occurrence": None,
                "remaining": None,
                "stop": None,
                "viseme_logits": None,
            }

        token_positions = torch.arange(max_q0_len, device=q0_repr.device, dtype=torch.long).unsqueeze(0)
        valid = token_positions < q0_lengths.to(device=q0_repr.device, dtype=torch.long).unsqueeze(1)
        plan_positions = torch.div(token_positions, 2, rounding_mode="floor")
        if self.q0_planner_adapter_plan_shift != 0:
            plan_positions = plan_positions + int(self.q0_planner_adapter_plan_shift)
        plan_positions = plan_positions.clamp(min=0, max=plan_len - 1)
        plan_positions = plan_positions.expand(batch_size, -1)

        plan_state = torch.gather(
            plan_hidden,
            1,
            plan_positions.unsqueeze(-1).expand(-1, -1, plan_hidden.shape[-1]),
        )
        plan_state = torch.where(valid.unsqueeze(-1), plan_state, torch.zeros_like(plan_state))
        if self.q0_planner_adapter_ablate_plan_hidden:
            plan_state = torch.zeros_like(plan_state)

        def gather_optional(name: str) -> Optional[torch.Tensor]:
            if plan_outputs is None or plan_outputs.get(name) is None:
                return None
            value = plan_outputs[name]
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            value = torch.gather(
                value,
                1,
                plan_positions.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
            )
            return torch.where(valid.unsqueeze(-1), value, torch.zeros_like(value))

        occurrence = gather_optional("plan_occurrence_pred")
        remaining = gather_optional("plan_remaining_pred")
        stop = gather_optional("plan_stop_logits")
        viseme_logits = gather_optional("plan_viseme_logits")

        if self.q0_planner_adapter_ablate_occurrence and occurrence is not None:
            occurrence = torch.zeros_like(occurrence)
        if self.q0_planner_adapter_ablate_remaining and remaining is not None:
            remaining = torch.zeros_like(remaining)
        if self.q0_planner_adapter_ablate_stop and stop is not None:
            stop = torch.zeros_like(stop)
        if self.q0_planner_adapter_ablate_viseme and viseme_logits is not None:
            viseme_logits = torch.zeros_like(viseme_logits)

        if self.q0_planner_adapter_batch_shuffle and batch_size > 1:
            roll = 1
            plan_state = plan_state.roll(shifts=roll, dims=0)
            if occurrence is not None:
                occurrence = occurrence.roll(shifts=roll, dims=0)
            if remaining is not None:
                remaining = remaining.roll(shifts=roll, dims=0)
            if stop is not None:
                stop = stop.roll(shifts=roll, dims=0)
            if viseme_logits is not None:
                viseme_logits = viseme_logits.roll(shifts=roll, dims=0)

        return {
            "plan_state": plan_state,
            "occurrence": occurrence,
            "remaining": remaining,
            "stop": stop,
            "viseme_logits": viseme_logits,
        }

    def _apply_q0_planner_state_adapter_to_aligned_logits(
        self,
        aligned_logits: torch.Tensor,
        q0_repr: Optional[torch.Tensor],
        q0_lengths: Optional[torch.Tensor],
        plan_hidden: Optional[torch.Tensor],
        plan_outputs: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        if (
            not self.use_q0_planner_state_adapter
            or self.q0_planner_state_adapter is None
            or self.q0_planner_adapter_disable
            or q0_repr is None
            or q0_lengths is None
            or plan_hidden is None
            or not hasattr(self.predict_layer, "heads")
            or len(self.predict_layer.heads) == 0
        ):
            return aligned_logits, metrics

        states = self._gather_q0_planner_states(q0_repr, q0_lengths, plan_hidden, plan_outputs)
        adapted_q0, delta = self.q0_planner_state_adapter(
            q0_repr=q0_repr,
            plan_state=states["plan_state"],
            occurrence=states["occurrence"],
            remaining=states["remaining"],
            stop=states["stop"],
            viseme_logits=states["viseme_logits"],
        )
        q0_logits = self.predict_layer.heads[0].classifier(adapted_q0)
        flat_logits = []
        for b in range(q0_lengths.shape[0]):
            length = int(q0_lengths[b].item())
            if length > 0:
                flat_logits.append(q0_logits[b, :length, :])
        if not flat_logits:
            return aligned_logits, metrics
        flat_q0_logits = torch.cat(flat_logits, dim=0)
        if flat_q0_logits.shape[0] != aligned_logits.shape[1]:
            return aligned_logits, metrics
        aligned_logits = aligned_logits.clone()
        aligned_logits[0] = flat_q0_logits.to(dtype=aligned_logits.dtype)
        valid_mask = torch.arange(q0_repr.shape[1], device=q0_repr.device).unsqueeze(0) < q0_lengths.unsqueeze(1)
        if bool(valid_mask.any().item()):
            metrics["q0_planner_adapter_delta_norm"] = (
                delta.norm(dim=-1)[valid_mask] / q0_repr.norm(dim=-1).clamp_min(1e-6)[valid_mask]
            ).mean().detach()
        metrics["q0_planner_adapter_gate"] = self.q0_planner_state_adapter.gate.detach()
        return aligned_logits, metrics

    def _apply_q0_planner_state_adapter_step(
        self,
        q0_step_repr: torch.Tensor,
        step: int,
        plan_hidden: Optional[torch.Tensor],
        plan_outputs: Optional[Dict[str, torch.Tensor]],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        metrics: Dict[str, torch.Tensor] = {}
        if (
            not self.use_q0_planner_state_adapter
            or self.q0_planner_state_adapter is None
            or self.q0_planner_adapter_disable
            or plan_hidden is None
        ):
            return q0_step_repr, metrics
        plan_len = plan_hidden.shape[1]
        if plan_len <= 0:
            return q0_step_repr, metrics
        plan_idx = max(int(step) - 1, 0) // 2
        if self.q0_planner_adapter_plan_shift != 0:
            plan_idx += int(self.q0_planner_adapter_plan_shift)
        plan_idx = min(plan_idx, plan_len - 1)
        plan_idx = max(plan_idx, 0)
        plan_state = plan_hidden[:, plan_idx : plan_idx + 1, :]
        if self.q0_planner_adapter_ablate_plan_hidden:
            plan_state = torch.zeros_like(plan_state)

        def pick_optional(name: str) -> Optional[torch.Tensor]:
            if plan_outputs is None or plan_outputs.get(name) is None:
                return None
            value = plan_outputs[name][:, plan_idx : plan_idx + 1]
            if value.ndim == 2:
                value = value.unsqueeze(-1)
            return value

        occurrence = pick_optional("plan_occurrence_pred")
        remaining = pick_optional("plan_remaining_pred")
        stop = pick_optional("plan_stop_logits")
        viseme_logits = pick_optional("plan_viseme_logits")
        if self.q0_planner_adapter_ablate_occurrence and occurrence is not None:
            occurrence = torch.zeros_like(occurrence)
        if self.q0_planner_adapter_ablate_remaining and remaining is not None:
            remaining = torch.zeros_like(remaining)
        if self.q0_planner_adapter_ablate_stop and stop is not None:
            stop = torch.zeros_like(stop)
        if self.q0_planner_adapter_ablate_viseme and viseme_logits is not None:
            viseme_logits = torch.zeros_like(viseme_logits)

        adapted, delta = self.q0_planner_state_adapter(
            q0_repr=q0_step_repr,
            plan_state=plan_state,
            occurrence=occurrence,
            remaining=remaining,
            stop=stop,
            viseme_logits=viseme_logits,
        )
        metrics["q0_planner_adapter_delta_norm"] = (
            delta.norm(dim=-1) / q0_step_repr.norm(dim=-1).clamp_min(1e-6)
        ).mean().detach()
        metrics["q0_planner_adapter_gate"] = self.q0_planner_state_adapter.gate.detach()
        return adapted, metrics

    def _add_text_segment_embeddings(
        self,
        x_emb: torch.Tensor,
        x: torch.Tensor,
        x_lens: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.text_ref_segment_embedding is None
            or self.text_target_segment_embedding is None
            or self.text_target_start_token_id is None
        ):
            return x_emb

        seq_positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        valid_mask = seq_positions < x_lens.unsqueeze(1)
        target_mask = torch.zeros_like(valid_mask)
        target_token_mask = x == int(self.text_target_start_token_id)
        has_target = target_token_mask.any(dim=1)

        if bool(has_target.any().item()):
            target_start_idx = target_token_mask.float().argmax(dim=1)
            target_mask = (seq_positions >= target_start_idx.unsqueeze(1)) & valid_mask

        ref_mask = valid_mask & ~target_mask
        ref_type = self.text_ref_segment_embedding(
            batch_size=x_emb.shape[0],
            seq_len=x_emb.shape[1],
            device=x_emb.device,
        )
        target_type = self.text_target_segment_embedding(
            batch_size=x_emb.shape[0],
            seq_len=x_emb.shape[1],
            device=x_emb.device,
        )
        x_emb = x_emb + ref_type * ref_mask.unsqueeze(-1).to(dtype=x_emb.dtype)
        x_emb = x_emb + target_type * target_mask.unsqueeze(-1).to(dtype=x_emb.dtype)
        return x_emb

    def _build_shifted_fusion_positions(
        self,
        shifted_lengths: torch.Tensor,
        anchor_lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if max_len <= 0:
            return torch.zeros((shifted_lengths.shape[0], 0), device=device, dtype=torch.long)
        base = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
        base = base.expand(shifted_lengths.shape[0], -1)
        clip_max = torch.clamp(anchor_lengths.to(device=device, dtype=torch.long) - 1, min=0).unsqueeze(1)
        fusion_positions = torch.minimum(base - 1, clip_max)
        valid = base < shifted_lengths.to(device=device).unsqueeze(1)
        invalid_fill = torch.full_like(fusion_positions, -1)
        return torch.where(valid, fusion_positions, invalid_fill)

    def _get_in_decoder_layer_map(self) -> Dict[int, int]:
        if not self.use_in_decoder_cross_attention or len(self.in_decoder_cross_attention_layers) == 0:
            return {}
        total_layers = len(self.decoder.layers)
        num_layers = min(self.in_decoder_cross_attention_num_layers, total_layers)
        start = total_layers - num_layers
        return {start + idx: idx for idx in range(num_layers)}

    def _apply_in_decoder_video_attention(
        self,
        hidden_states: torch.Tensor,
        video_memory: Optional[torch.Tensor],
        target_query_positions: Optional[torch.Tensor],
        target_offsets: torch.Tensor,
        target_lengths: torch.Tensor,
        video_lens: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        layer_map = self._get_in_decoder_layer_map()
        if layer_idx not in layer_map:
            return hidden_states, None
        if video_memory is None or target_query_positions is None:
            return hidden_states, None

        block = self.in_decoder_cross_attention_layers[layer_map[layer_idx]]
        fused_hidden = hidden_states.clone()
        total_delta_norm = hidden_states.new_tensor(0.0)
        valid_samples = 0
        for i in range(hidden_states.shape[0]):
            target_len = int(target_lengths[i].item())
            if target_len <= 0:
                continue
            start = int(target_offsets[i].item())
            end = start + target_len
            query_slice = hidden_states[i : i + 1, start:end, :]
            position_slice = target_query_positions[i : i + 1, :target_len]
            fused_slice = block(
                query=query_slice,
                context=video_memory[i : i + 1],
                query_positions=position_slice,
                context_lens=video_lens[i : i + 1].to(hidden_states.device),
            )
            delta = fused_slice - query_slice
            delta_norm = delta.norm(dim=-1)
            base_norm = query_slice.norm(dim=-1).clamp_min(1e-6)
            total_delta_norm = total_delta_norm + (delta_norm / base_norm).mean()
            valid_samples += 1
            fused_hidden[i, start:end, :] = fused_slice[0]
        layer_metrics = {
            "gate_value": block.gate.detach(),
            "fusion_delta_norm": (total_delta_norm / max(valid_samples, 1)).detach(),
        }
        return fused_hidden, layer_metrics

    def _forward_encoder_layer_with_video(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        src_key_padding_mask: Optional[torch.Tensor],
        video_memory: Optional[torch.Tensor],
        target_query_positions: Optional[torch.Tensor],
        target_offsets: torch.Tensor,
        target_lengths: torch.Tensor,
        video_lens: torch.Tensor,
        layer_idx: int,
        past: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        if not getattr(layer, "norm_first", False):
            output = layer(
                hidden_states,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                past=past,
            )
            present = None
            if isinstance(output, list):
                output, present = output
            return output, present, None

        sa_out, present = layer._sa_block(
            layer.norm1(hidden_states),
            src_mask,
            src_key_padding_mask,
            past,
        )
        hidden_states = hidden_states + sa_out
        hidden_states, layer_metrics = self._apply_in_decoder_video_attention(
            hidden_states=hidden_states,
            video_memory=video_memory,
            target_query_positions=target_query_positions,
            target_offsets=target_offsets,
            target_lengths=target_lengths,
            video_lens=video_lens,
            layer_idx=layer_idx,
        )
        hidden_states = hidden_states + layer._ff_block(layer.norm2(hidden_states))
        return hidden_states, present, layer_metrics

    def _decoder_forward_with_in_decoder_video_attention(
        self,
        xy_input: torch.Tensor,
        xy_attn_mask: torch.Tensor,
        xy_pad_mask: torch.Tensor,
        video_memory: Optional[torch.Tensor],
        target_query_positions: Optional[torch.Tensor],
        target_offsets: torch.Tensor,
        target_lengths: torch.Tensor,
        video_lens: torch.Tensor,
        past: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
        output = xy_input
        all_present = []
        layer_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
        for layer_idx, layer in enumerate(self.decoder.layers):
            output, present, curr_layer_metrics = self._forward_encoder_layer_with_video(
                layer=layer,
                hidden_states=output,
                src_mask=xy_attn_mask,
                src_key_padding_mask=xy_pad_mask,
                video_memory=video_memory,
                target_query_positions=target_query_positions,
                target_offsets=target_offsets,
                target_lengths=target_lengths,
                video_lens=video_lens,
                layer_idx=layer_idx,
                past=None if past is None else past[layer_idx],
            )
            if curr_layer_metrics is not None:
                layer_metrics[layer_idx] = curr_layer_metrics
            if present is not None:
                all_present.append(present)

        if self.decoder.norm is not None:
            output = self.decoder.norm(output)
        if all_present:
            return output, torch.stack(all_present, dim=0), layer_metrics
        return output, None, layer_metrics

    def _build_generate_target_query_positions(
        self,
        target_lengths: torch.Tensor,
        anchor_lengths: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        max_len = int(target_lengths.max().item()) if target_lengths.numel() > 0 else 0
        if max_len <= 0:
            return torch.zeros((target_lengths.shape[0], 0), device=device, dtype=torch.long)
        base = torch.arange(max_len, device=device, dtype=torch.long).unsqueeze(0)
        base = base.expand(target_lengths.shape[0], -1)
        valid = base < target_lengths.unsqueeze(1)
        clipped = torch.minimum(
            torch.clamp(base - 1, min=0),
            torch.clamp(anchor_lengths.unsqueeze(1) - 1, min=0),
        )
        positions = torch.where(base == 0, torch.full_like(clipped, -1), clipped)
        return torch.where(valid, positions, torch.full_like(positions, -1))

    def _prepare_v4_audio_embeddings(
        self,
        embedded_y: torch.Tensor,
        delayed_ref_lengths: torch.Tensor,
        split_lengths: torch.Tensor,
        target_shifted_lengths: torch.Tensor,
        target_anchor_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = embedded_y.device
        batch_size, _, dim_model = embedded_y.shape
        y_emb = embedded_y.new_zeros(embedded_y.shape)
        max_ref_len = int(delayed_ref_lengths.max().item()) if delayed_ref_lengths.numel() > 0 else 0
        max_target_len = int(target_shifted_lengths.max().item()) if target_shifted_lengths.numel() > 0 else 0
        ref_positions, ref_position_mask = self._build_shifted_positions(
            delayed_ref_lengths,
            split_lengths,
            max_ref_len,
            device,
            skip_leading_blank=True,
        )
        target_positions, target_position_mask = self._build_shifted_positions(
            target_shifted_lengths,
            target_anchor_lens,
            max_target_len,
            device,
            skip_leading_blank=True,
        )
        target_fusion_positions = self._build_shifted_fusion_positions(
            target_shifted_lengths,
            target_anchor_lens,
            max_target_len,
            device,
        )

        for i in range(batch_size):
            ref_len = int(delayed_ref_lengths[i].item())
            if ref_len > 0:
                ref_slice = embedded_y[i:i + 1, :ref_len, :]
                ref_slice = self._encode_segment(
                    ref_slice,
                    ref_positions[i:i + 1, :ref_len],
                    self.ref_audio_positional_embedding,
                    self.ref_audio_segment_embedding,
                    position_mask=ref_position_mask[i:i + 1, :ref_len],
                )
                y_emb[i, :ref_len, :] = ref_slice[0]

            target_len = int(target_shifted_lengths[i].item())
            if target_len > 0:
                start = ref_len
                target_slice = embedded_y[i:i + 1, start:start + target_len, :]
                target_slice = self._encode_segment(
                    target_slice,
                    target_positions[i:i + 1, :target_len],
                    self.shared_audio_positional_embedding,
                    self.target_audio_segment_embedding,
                    position_mask=target_position_mask[i:i + 1, :target_len],
                )
                y_emb[i, start:start + target_len, :] = target_slice[0]

        return y_emb, target_fusion_positions

    def _prepare_v4_video_embeddings(
        self,
        video_embeddings: torch.Tensor,
        v_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions, _ = self._build_shifted_positions(
            v_lens,
            v_lens,
            int(video_embeddings.shape[1]),
            video_embeddings.device,
        )
        video_embeddings = self._encode_segment(
            video_embeddings,
            positions,
            self.shared_audio_positional_embedding,
            self.video_segment_embedding,
        )
        return video_embeddings, positions

    def _build_target_end_token(
        self,
        batch_size: int,
        v_lens: torch.Tensor,
        dim_model: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if not (self.use_v4 and self.use_target_end_token):
            return None
        end_positions = torch.clamp(v_lens.to(device=device, dtype=torch.long) - 1, min=0).unsqueeze(1)
        end_token = torch.zeros((batch_size, 1, dim_model), device=device)
        end_token = self.shared_audio_positional_embedding(end_token, positions=end_positions)
        return self._add_type_embedding(end_token, self.target_end_segment_embedding)

    def _fuse_target_inputs(
        self,
        y_emb: torch.Tensor,
        delayed_ref_lengths: torch.Tensor,
        target_shifted_lengths: torch.Tensor,
        cross_video_embeddings: Optional[torch.Tensor],
        target_fusion_positions: Optional[torch.Tensor],
        v_lens: torch.Tensor,
    ) -> torch.Tensor:
        if self.cross_attention is None or cross_video_embeddings is None or target_fusion_positions is None:
            return y_emb
        if int(target_shifted_lengths.max().item()) <= 0:
            return y_emb

        fused_y = y_emb.clone()
        for i in range(y_emb.shape[0]):
            ref_len = int(delayed_ref_lengths[i].item())
            target_len = int(target_shifted_lengths[i].item())
            if target_len <= 0:
                continue
            target_slice = y_emb[i : i + 1, ref_len : ref_len + target_len, :]
            fused_slice = self.cross_attention(
                target_slice,
                cross_video_embeddings[i : i + 1],
                target_fusion_positions[i : i + 1, :target_len],
                v_lens[i : i + 1].to(y_emb.device),
            )
            fused_y[i, ref_len : ref_len + target_len, :] = fused_slice[0]
        return fused_y

    def _apply_codebook_routing(
        self,
        target_hidden: torch.Tensor,
        plan_hidden: Optional[torch.Tensor],
        query_positions: Optional[torch.Tensor],
        video_lens: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if (
            not self.use_codebook_routing
            or self.plan_router is None
            or plan_hidden is None
            or query_positions is None
        ):
            return None, {}

        time_len = min(target_hidden.shape[1], query_positions.shape[1])
        if time_len <= 0:
            return None, {}

        routed_hidden, routing_metrics = self.plan_router(
            query_hidden=target_hidden[:, :time_len, :],
            plan_hidden=plan_hidden,
            query_positions=query_positions[:, :time_len],
            context_lens=video_lens.to(device=target_hidden.device, dtype=torch.long),
        )

        if time_len == target_hidden.shape[1]:
            return routed_hidden, routing_metrics

        padded = target_hidden.unsqueeze(1).expand(-1, self.n_codebooks, -1, -1).contiguous()
        padded[:, :, :time_len, :] = routed_hidden
        routing_metrics["routing_valid_mask"] = F.pad(
            routing_metrics["routing_valid_mask"],
            (0, target_hidden.shape[1] - time_len),
            value=False,
        )
        routing_metrics["routing_lambdas"] = F.pad(
            routing_metrics["routing_lambdas"],
            (0, 0, 0, target_hidden.shape[1] - time_len),
            value=0.5,
        )
        return padded, routing_metrics

    def _build_v3_video_positions(
        self,
        delayed_tar_start_ids: torch.Tensor,
        v_lens: torch.Tensor,
        max_v_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        video_positions = torch.zeros((v_lens.shape[0], max_v_len), device=device, dtype=torch.long)
        for i in range(v_lens.shape[0]):
            v_len = int(v_lens[i].item())
            if v_len <= 0:
                continue
            target_start = int(delayed_tar_start_ids[i].item())
            video_positions[i, :v_len] = torch.arange(
                target_start,
                target_start + v_len,
                device=device,
                dtype=torch.long,
            )
        return video_positions

    def _compute_vq_usage(
        self,
        token_ids: Optional[torch.Tensor],
        v_lens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if token_ids is None:
            return None
        with torch.no_grad():
            max_v = token_ids.shape[1]
            v_mask = torch.arange(max_v, device=token_ids.device)[None, :] < v_lens[:, None]
            valid_ids = token_ids[v_mask]
            if valid_ids.numel() == 0:
                return torch.tensor(0.0, device=token_ids.device)
            unique_codes = torch.unique(valid_ids)
            return unique_codes.numel() / float(self.video_tokenizer.vq.n_embed)

    def _build_per_codebook_real_time_index(
        self,
        shifted_lengths: torch.Tensor,
        anchor_lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_len <= 0:
            empty = torch.zeros(
                (shifted_lengths.shape[0], 0, self.n_codebooks),
                device=device,
                dtype=torch.long,
            )
            valid = torch.zeros_like(empty, dtype=torch.bool)
            return empty, valid

        base = torch.arange(max_len, device=device, dtype=torch.long).view(1, max_len, 1)
        base = base.expand(shifted_lengths.shape[0], -1, self.n_codebooks)
        codebook_ids = torch.arange(self.n_codebooks, device=device, dtype=torch.long).view(1, 1, self.n_codebooks)
        real_times = base - 1 - codebook_ids

        shifted_valid = base < shifted_lengths.to(device=device, dtype=torch.long).view(-1, 1, 1)
        time_valid = (real_times >= 0) & (
            real_times < anchor_lengths.to(device=device, dtype=torch.long).view(-1, 1, 1)
        )
        valid = shifted_valid & time_valid
        real_times = torch.where(valid, real_times, torch.full_like(real_times, -1))
        return real_times, valid

    def _build_per_codebook_routing_positions(
        self,
        shifted_lengths: torch.Tensor,
        anchor_lengths: torch.Tensor,
        max_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        real_times, _ = self._build_per_codebook_real_time_index(
            shifted_lengths=shifted_lengths,
            anchor_lengths=anchor_lengths,
            max_len=max_len,
            device=device,
        )
        return real_times

    def _apply_hierarchical_codebook_conditioning(
        self,
        codebook_hidden: Optional[torch.Tensor],
        rearranged_y: list,
        target_time_len: int,
    ) -> tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        if (
            not self.use_hierarchical_codebook_conditioning
            or self.hierarchical_codebook_conditioner is None
            or codebook_hidden is None
            or target_time_len <= 0
        ):
            return codebook_hidden, {}

        shifted_target_tokens, shifted_target_lens = self.processor.build_shifted_target_tokens(rearranged_y)
        shifted_target_tokens = shifted_target_tokens.to(codebook_hidden.device)
        shifted_target_lens = shifted_target_lens.to(codebook_hidden.device)
        if shifted_target_tokens.shape[-1] < target_time_len:
            raise ValueError(
                f"Shifted target tokens shorter than target hidden: "
                f"{shifted_target_tokens.shape[-1]} < {target_time_len}"
            )
        shifted_target_tokens = shifted_target_tokens[:, :, :target_time_len]
        conditioned_hidden, hier_metrics = self.hierarchical_codebook_conditioner(
            base_hidden=codebook_hidden,
            shifted_token_ids=shifted_target_tokens,
            audio_embedding=self.audio_embedding,
        )
        hier_metrics["hier_shifted_target_lens"] = shifted_target_lens.detach()
        return conditioned_hidden, hier_metrics

    def _pool_sync_codebook_reprs(
        self,
        codebook_reprs: torch.Tensor,
        shifted_lengths: torch.Tensor,
        anchor_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _, repr_dim = codebook_reprs.shape
        max_anchor = int(anchor_lengths.max().item()) if anchor_lengths.numel() > 0 else 0
        if max_anchor <= 0:
            empty = codebook_reprs.new_zeros((batch_size, 0, repr_dim))
            mask = torch.zeros((batch_size, 0), device=codebook_reprs.device, dtype=torch.bool)
            return empty, mask

        time_index, valid = self._build_per_codebook_real_time_index(
            shifted_lengths=shifted_lengths,
            anchor_lengths=anchor_lengths,
            max_len=codebook_reprs.shape[1],
            device=codebook_reprs.device,
        )
        pooled = codebook_reprs.new_zeros((batch_size, max_anchor, repr_dim))
        norm = codebook_reprs.new_zeros((batch_size, max_anchor, 1))

        weights = self.sync_codebook_weights.to(device=codebook_reprs.device, dtype=codebook_reprs.dtype)
        for q in range(self.n_codebooks):
            q_valid = valid[:, :, q]
            if not bool(q_valid.any().item()):
                continue
            q_times = time_index[:, :, q].clamp_min(0)
            q_repr = codebook_reprs[:, :, q, :] * q_valid.unsqueeze(-1).to(codebook_reprs.dtype) * weights[q]
            pooled.scatter_add_(1, q_times.unsqueeze(-1).expand(-1, -1, repr_dim), q_repr)
            q_weight = q_valid.to(codebook_reprs.dtype).unsqueeze(-1) * weights[q]
            norm.scatter_add_(1, q_times.unsqueeze(-1), q_weight)

        pooled = pooled / norm.clamp_min(1e-6)
        mask = torch.arange(max_anchor, device=codebook_reprs.device).unsqueeze(0) < anchor_lengths.unsqueeze(1)
        return pooled, mask

    def _compute_sync_loss_nce(
        self,
        audio_sync: torch.Tensor,
        video_sync: torch.Tensor,
        sync_mask: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        valid_positions = sync_mask.nonzero(as_tuple=False)
        if valid_positions.numel() == 0:
            return None, None, None

        audio_valid = audio_sync[sync_mask]
        video_valid = video_sync[sync_mask]
        cosine_matrix = audio_valid @ video_valid.transpose(0, 1)
        pos_alignment = cosine_matrix.diagonal().mean()

        batch_ids = valid_positions[:, 0]
        time_ids = valid_positions[:, 1]
        same_sample = batch_ids[:, None] == batch_ids[None, :]
        near_neighbor = (time_ids[:, None] - time_ids[None, :]).abs() <= self.sync_negative_delta
        negative_mask = ~(same_sample & near_neighbor)
        negative_mask.fill_diagonal_(False)
        neg_alignment = cosine_matrix.masked_select(negative_mask).mean() if bool(negative_mask.any().item()) else None

        if audio_valid.shape[0] <= 1:
            return audio_valid.new_tensor(0.0), pos_alignment, neg_alignment

        logits = cosine_matrix
        logits = logits / max(self.sync_nce_temperature, 1e-6)

        candidate_mask = ~(same_sample & near_neighbor)
        candidate_mask.fill_diagonal_(True)

        neg_fill = torch.finfo(logits.dtype).min
        masked_logits = logits.masked_fill(~candidate_mask, neg_fill)
        log_probs = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
        loss = -log_probs.diagonal().mean()
        return loss, pos_alignment, neg_alignment

    def _compute_sync_loss(
        self,
        pooled_audio: Optional[torch.Tensor],
        sync_mask: Optional[torch.Tensor],
        video_sync_inputs: Optional[torch.Tensor],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_sync_loss:
            return None, None, None
        if (
            pooled_audio is None
            or sync_mask is None
            or self.audio_sync_projector is None
            or self.video_sync_projector is None
            or video_sync_inputs is None
        ):
            return None, None, None
        if pooled_audio.numel() == 0 or not bool(sync_mask.any().item()):
            return None, None, None

        audio_inputs = pooled_audio.detach() if self.sync_detach_audio else pooled_audio
        video_inputs = video_sync_inputs.detach() if self.sync_detach_video else video_sync_inputs
        audio_sync = self.audio_sync_projector(audio_inputs)
        video_sync = self.video_sync_projector(video_inputs)
        max_common = min(audio_sync.shape[1], video_sync.shape[1])
        audio_sync = audio_sync[:, :max_common]
        video_sync = video_sync[:, :max_common]
        sync_mask = sync_mask[:, :max_common]

        if self.sync_loss_type == "nce":
            return self._compute_sync_loss_nce(audio_sync, video_sync, sync_mask)

        cosine = F.cosine_similarity(audio_sync, video_sync, dim=-1)
        loss = ((1.0 - cosine) * sync_mask.to(cosine.dtype)).sum() / sync_mask.sum().clamp_min(1)
        pos_alignment = cosine.masked_select(sync_mask).mean() if bool(sync_mask.any().item()) else None

        flat_audio = audio_sync[sync_mask]
        flat_video = video_sync[sync_mask]
        if flat_audio.numel() > 0 and flat_audio.shape[0] > 1:
            valid_positions = sync_mask.nonzero(as_tuple=False)
            cosine_matrix = flat_audio @ flat_video.transpose(0, 1)
            batch_ids = valid_positions[:, 0]
            time_ids = valid_positions[:, 1]
            same_sample = batch_ids[:, None] == batch_ids[None, :]
            near_neighbor = (time_ids[:, None] - time_ids[None, :]).abs() <= self.sync_negative_delta
            negative_mask = ~(same_sample & near_neighbor)
            negative_mask.fill_diagonal_(False)
            neg_alignment = cosine_matrix.masked_select(negative_mask).mean() if bool(negative_mask.any().item()) else None
        else:
            neg_alignment = None
        return loss, pos_alignment, neg_alignment

    def _compute_g_embedding_losses(
        self,
        codebook_reprs: Optional[torch.Tensor],
        rearranged_y: list,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not self.use_g_embedding_loss or codebook_reprs is None:
            return {}
        if codebook_reprs.ndim != 4:
            raise ValueError(f"Expected codebook_reprs [B,K,T,D], got {tuple(codebook_reprs.shape)}")

        shifted_targets, _ = self.processor.build_shifted_target_tokens(rearranged_y)
        shifted_targets = shifted_targets.to(device=codebook_reprs.device, dtype=torch.long)
        time_len = min(codebook_reprs.shape[2], shifted_targets.shape[2])
        if time_len <= 0:
            return {}
        shifted_targets = shifted_targets[:, :, :time_len]
        codebook_reprs = codebook_reprs[:, :, :time_len, :]

        if self.g_embedding_codebook_weights is None:
            weights = codebook_reprs.new_ones((self.n_codebooks,))
        else:
            weights = self.g_embedding_codebook_weights.to(
                device=codebook_reprs.device,
                dtype=codebook_reprs.dtype,
            )

        cos_losses = []
        ce_losses = []
        cos_means = []
        top1_values = []
        top10_values = []
        used_weights = []
        per_q: Dict[str, torch.Tensor] = {}

        for q in range(self.n_codebooks):
            targets_q = shifted_targets[:, q, :]
            valid = (targets_q >= 0) & (targets_q < int(self.processor.empty_id))
            if not bool(valid.any().item()):
                continue

            g_q = codebook_reprs[:, q, :, :][valid]
            target_ids = targets_q[valid]
            emb_module = self.audio_embedding.audio_embeddings[q]
            emb_weight = emb_module.word_embeddings.weight.to(
                device=g_q.device,
                dtype=g_q.dtype,
            )
            target_emb = emb_weight[target_ids]
            if self.g_embedding_detach_targets:
                target_emb = target_emb.detach()
                emb_weight_for_logits = emb_weight.detach()
            else:
                emb_weight_for_logits = emb_weight

            g_norm = F.normalize(g_q.float(), dim=-1).to(dtype=g_q.dtype)
            target_norm = F.normalize(target_emb.float(), dim=-1).to(dtype=g_q.dtype)
            cosine = (g_norm * target_norm).sum(dim=-1)
            cos_loss_q = 1.0 - cosine.mean()

            emb_norm = F.normalize(emb_weight_for_logits.float(), dim=-1).to(dtype=g_q.dtype)
            sim_logits = torch.matmul(g_norm, emb_norm.transpose(0, 1)) / max(self.g_embedding_temperature, 1e-6)
            ce_loss_q = F.cross_entropy(sim_logits, target_ids, reduction="mean")

            with torch.no_grad():
                pred = sim_logits.argmax(dim=-1)
                top1 = (pred == target_ids).float().mean()
                k = min(10, sim_logits.shape[-1])
                topk = sim_logits.topk(k, dim=-1).indices
                top10 = (topk == target_ids.unsqueeze(-1)).any(dim=-1).float().mean()

            weight_q = weights[q]
            cos_losses.append(cos_loss_q * weight_q)
            ce_losses.append(ce_loss_q * weight_q)
            cos_means.append(cosine.mean() * weight_q)
            top1_values.append(top1 * weight_q)
            top10_values.append(top10 * weight_q)
            used_weights.append(weight_q)
            per_q[f"g_embedding_cos_q{q}"] = cosine.mean().detach()
            per_q[f"g_embedding_ce_q{q}"] = ce_loss_q.detach()
            per_q[f"g_embedding_top1_q{q}"] = top1.detach()
            per_q[f"g_embedding_top10_q{q}"] = top10.detach()

        if not used_weights:
            return {}
        denom = torch.stack(used_weights).sum().clamp_min(1e-6)
        out: Dict[str, Optional[torch.Tensor]] = {
            "g_embedding_cos_loss": torch.stack(cos_losses).sum() / denom,
            "g_embedding_ce_loss": torch.stack(ce_losses).sum() / denom,
            "g_embedding_cos_mean": torch.stack(cos_means).sum().detach() / denom.detach(),
            "g_embedding_top1": torch.stack(top1_values).sum().detach() / denom.detach(),
            "g_embedding_top10": torch.stack(top10_values).sum().detach() / denom.detach(),
        }
        out.update(per_q)
        return out

    def _build_plan_acoustic_targets(
        self,
        rearranged_y: list,
        plan_lengths: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if plan_lengths is None or plan_lengths.numel() == 0:
            return None, None
        target_segments = [segments[-1] for segments in rearranged_y]
        max_plan = int(plan_lengths.max().item()) if plan_lengths.numel() > 0 else 0
        if max_plan <= 0:
            return None, None
        batch_size = len(target_segments)
        targets = torch.zeros(
            (batch_size, max_plan, self.audio_embedding[0].dim_model),
            device=device,
            dtype=dtype,
        )
        mask = torch.zeros((batch_size, max_plan), device=device, dtype=torch.bool)

        if self.plan_acoustic_target_weights is None:
            weights = torch.tensor([1.0, 0.5, 0.25, 0.125], device=device, dtype=dtype)
        else:
            weights = self.plan_acoustic_target_weights.to(device=device, dtype=dtype)
        weights = weights / weights.sum().clamp_min(1e-6)

        for b, target_seg in enumerate(target_segments):
            audio_len = int(target_seg.shape[1])
            plan_len = int(plan_lengths[b].item())
            if audio_len <= 0 or plan_len <= 0:
                continue
            proto = torch.zeros((audio_len, targets.shape[-1]), device=device, dtype=dtype)
            for q in range(self.n_codebooks):
                token_ids = target_seg[q].to(device=device, dtype=torch.long)
                emb_weight = self.audio_embedding.audio_embeddings[q].word_embeddings.weight.to(device=device, dtype=dtype)
                proto = proto + emb_weight[token_ids] * weights[q]
            proto = F.normalize(proto.float(), dim=-1).to(dtype=dtype)
            if self.plan_acoustic_detach_targets:
                proto = proto.detach()

            if audio_len == plan_len:
                targets[b, :plan_len] = proto
            else:
                resized = F.interpolate(
                    proto.transpose(0, 1).unsqueeze(0),
                    size=plan_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(0).transpose(0, 1).contiguous()
                targets[b, :plan_len] = F.normalize(resized.float(), dim=-1).to(dtype=dtype)
            mask[b, :plan_len] = True
        return targets, mask

    def _compute_plan_acoustic_loss(
        self,
        plan_hidden: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        rearranged_y: list,
    ) -> Dict[str, Optional[torch.Tensor]]:
        if (
            not self.use_plan_acoustic_loss
            or self.plan_acoustic_head is None
            or plan_hidden is None
            or plan_lengths is None
        ):
            return {}
        target_proto, valid_mask = self._build_plan_acoustic_targets(
            rearranged_y=rearranged_y,
            plan_lengths=plan_lengths,
            device=plan_hidden.device,
            dtype=plan_hidden.dtype,
        )
        if target_proto is None or valid_mask is None or not bool(valid_mask.any().item()):
            return {}

        pred_proto = self.plan_acoustic_head(plan_hidden)
        max_common = min(pred_proto.shape[1], target_proto.shape[1])
        pred_proto = pred_proto[:, :max_common]
        target_proto = target_proto[:, :max_common]
        valid_mask = valid_mask[:, :max_common]
        if not bool(valid_mask.any().item()):
            return {}

        pred_valid = pred_proto[valid_mask]
        tgt_valid = target_proto[valid_mask]
        cosine = (pred_valid * tgt_valid).sum(dim=-1)
        pos_alignment = cosine.mean()

        logits = pred_valid @ tgt_valid.transpose(0, 1)
        logits = logits / max(self.plan_acoustic_temperature, 1e-6)
        valid_positions = valid_mask.nonzero(as_tuple=False)
        batch_ids = valid_positions[:, 0]
        time_ids = valid_positions[:, 1]
        same_sample = batch_ids[:, None] == batch_ids[None, :]
        near_neighbor = (time_ids[:, None] - time_ids[None, :]).abs() <= self.plan_acoustic_negative_delta
        negative_mask = ~(same_sample & near_neighbor)
        negative_mask.fill_diagonal_(False)
        neg_alignment = logits.masked_select(negative_mask).mean() if bool(negative_mask.any().item()) else None

        if pred_valid.shape[0] <= 1:
            nce_loss = pred_valid.new_tensor(0.0)
        else:
            candidate_mask = ~(same_sample & near_neighbor)
            candidate_mask.fill_diagonal_(True)
            neg_fill = torch.finfo(logits.dtype).min
            masked_logits = logits.masked_fill(~candidate_mask, neg_fill)
            log_probs = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
            nce_loss = -log_probs.diagonal().mean()

        return {
            "plan_acoustic_loss": nce_loss,
            "plan_acoustic_pos_alignment": pos_alignment.detach(),
            "plan_acoustic_neg_alignment": None if neg_alignment is None else neg_alignment.detach(),
        }

    def _get_curriculum_rate(self, max_rate: float, warmup_steps: int) -> float:
        if max_rate <= 0:
            return 0.0
        if warmup_steps <= 0:
            return float(max_rate)
        step = max(int(getattr(self, "current_train_step", 0)), 0)
        return float(max_rate) * min(step / float(warmup_steps), 1.0)

    def _get_linear_schedule_rate(self, start_rate: float, end_rate: float, warmup_steps: int) -> float:
        start_rate = float(start_rate)
        end_rate = float(end_rate)
        if warmup_steps <= 0:
            return end_rate
        step = max(int(getattr(self, "current_train_step", 0)), 0)
        alpha = min(step / float(warmup_steps), 1.0)
        return start_rate + (end_rate - start_rate) * alpha

    def _apply_target_history_perturbation(
        self,
        y: torch.Tensor,
        y_lens: torch.Tensor,
        split_lengths: torch.Tensor,
        hc_rate: float = 0.0,
        ss_rate: float = 0.0,
        pred_target_tokens: Optional[torch.Tensor] = None,
        pred_target_lengths: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, int, int, int]:
        device = y.device
        y_mod = y.clone()
        total_target_frames = 0
        ss_applied = 0
        hc_applied = 0

        for i in range(y.shape[0]):
            target_start = int(split_lengths[i].item())
            target_end = int(y_lens[i].item())
            target_len = max(target_end - target_start, 0)
            if target_len <= 0:
                continue
            total_target_frames += target_len

            ss_mask = torch.zeros(target_len, device=device, dtype=torch.bool)
            if pred_target_tokens is not None and pred_target_lengths is not None and ss_rate > 0:
                use_len = min(target_len, int(pred_target_lengths[i].item()))
                if use_len > 0:
                    ss_mask[:use_len] = torch.rand(use_len, device=device) < ss_rate
                    ss_indices = ss_mask[:use_len].nonzero(as_tuple=False).flatten().tolist()
                    for idx in ss_indices:
                        y_mod[i, :, target_start + idx] = pred_target_tokens[i, :, idx]
                    ss_applied += len(ss_indices)

            if hc_rate > 0:
                hc_mask = (torch.rand(target_len, device=device) < hc_rate) & (~ss_mask)
                hc_indices = hc_mask.nonzero(as_tuple=False).flatten().tolist()
                for idx in hc_indices:
                    random_frame = torch.randint(0, int(self.processor.empty_id), (self.n_codebooks,), device=device, dtype=y.dtype)
                    y_mod[i, :, target_start + idx] = random_frame
                    hc_applied += 1

        return y_mod, total_target_frames, ss_applied, hc_applied

    def _prepare_training_batch_with_exposure_mitigation(
        self,
        batch: Dict,
        teacher_outputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> tuple[Dict, Dict[str, torch.Tensor]]:
        device = batch["y"].device
        ss_rate = self._get_linear_schedule_rate(self.scheduled_sampling_start_rate, self.scheduled_sampling_max_rate, self.scheduled_sampling_warmup_steps) if self.use_scheduled_sampling else 0.0
        hc_rate = self._get_curriculum_rate(self.history_corruption_max_rate, self.history_corruption_warmup_steps)

        zero = torch.tensor(0.0, device=device)
        metrics = {
            "history_corruption_rate_used": torch.tensor(hc_rate, device=device),
            "scheduled_sampling_rate_used": torch.tensor(ss_rate, device=device),
            "history_corruption_applied_ratio": zero.clone(),
            "scheduled_sampling_applied_ratio": zero.clone(),
        }
        if (ss_rate <= 0 and hc_rate <= 0) or not (self.use_history_corruption or self.use_scheduled_sampling):
            return batch, metrics

        pred_target_tokens = None
        pred_target_lengths = None
        if self.use_scheduled_sampling and ss_rate > 0:
            if teacher_outputs is None:
                teacher_batch = dict(batch)
                teacher_batch["_disable_exposure_mitigation"] = True
                with torch.no_grad():
                    teacher_outputs = self._forward_impl(teacher_batch, return_pred_target_tokens=True)
            pred_target_tokens = teacher_outputs.get("pred_target_tokens")
            pred_target_lengths = teacher_outputs.get("pred_target_lengths")

        y = batch["y"]
        y_lens = batch["y_lens"].to(device=device, dtype=torch.long)
        split_lengths = batch["split_lens"].to(device=device, dtype=torch.long)
        y_mod, total_target_frames, ss_applied, hc_applied = self._apply_target_history_perturbation(
            y=y,
            y_lens=y_lens,
            split_lengths=split_lengths,
            hc_rate=hc_rate if self.use_history_corruption else 0.0,
            ss_rate=ss_rate if self.use_scheduled_sampling else 0.0,
            pred_target_tokens=pred_target_tokens,
            pred_target_lengths=pred_target_lengths,
        )

        if total_target_frames > 0:
            metrics["scheduled_sampling_applied_ratio"] = torch.tensor(ss_applied / float(total_target_frames), device=device)
            metrics["history_corruption_applied_ratio"] = torch.tensor(hc_applied / float(total_target_frames), device=device)

        mitigated_batch = dict(batch)
        mitigated_batch["y"] = y_mod
        return mitigated_batch, metrics

    def _forward_impl(
        self,
        batch: Dict,
        return_pred_target_tokens: bool = False,
        return_interface_state: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x, x_lens = batch["x"], batch["x_lens"]
        y, y_lens = batch["y"], batch["y_lens"]
        v, v_lens = batch["v"], batch["v_lens"]
        split_lengths = batch["split_lens"]

        x = x[:, :x_lens.max()]
        y = y[:, :, :y_lens.max()]
        v = v[:, :v_lens.max()]
        assert x.ndim == 2, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.ndim == 3 and y.shape[1] == self.n_codebooks, y.shape
        assert y_lens.ndim == 1, y_lens.shape

        x_emb = self.text_embedding(x)
        x_emb = self.text_positional_embedding(x_emb)
        x_emb = self._add_text_segment_embeddings(x_emb, x, x_lens)

        if self.enable_audio_mask:
            y_masked = self.processor.apply_random_audio_mask(y, y_lens)
        else:
            y_masked = y
        data = self.processor.prepare_audio_input(
            y_masked,
            y_lens,
            split_lengths,
            self.audio_embedding,
        )

        embedded_y = data["embedded_y"]
        delayed_ref_lengths = split_lengths + self.n_codebooks
        delayed_tar_start_ids = delayed_ref_lengths
        target_shifted_lengths = data["new_y_lens"] - delayed_ref_lengths

        if v.device != x_emb.device:
            v = v.to(x_emb.device)
            v_lens = v_lens.to(x_emb.device)
        v_lens, v_upsampled = self.processor.prepare_video_input(v, v_lens)
        token_out = self.video_tokenizer(v_upsampled, v_lens)

        disable_plan_conditioning = bool(batch.get("_disable_plan_conditioning", False))
        disable_plan_supervision = bool(batch.get("_disable_plan_supervision", False))
        plan_shift_steps = int(batch.get("_plan_shift_steps", 0))

        if self.use_v4:
            y_emb, target_fusion_positions = self._prepare_v4_audio_embeddings(
                embedded_y=embedded_y,
                delayed_ref_lengths=delayed_ref_lengths,
                split_lengths=split_lengths,
                target_shifted_lengths=target_shifted_lengths,
                target_anchor_lens=v_lens,
            )
            raw_video_sync_inputs = token_out.get("pre_vq_embeddings", token_out["embeddings"])
            cross_video_embeddings, _ = self._prepare_v4_video_embeddings(
                raw_video_sync_inputs,
                v_lens,
            )
            if self.use_video_prefix:
                video_tokens_emb, _ = self._prepare_v4_video_embeddings(token_out["embeddings"], v_lens)
            else:
                video_tokens_emb = None
            if not self.use_in_decoder_cross_attention:
                y_emb = self._fuse_target_inputs(
                    y_emb=y_emb,
                    delayed_ref_lengths=delayed_ref_lengths,
                    target_shifted_lengths=target_shifted_lengths,
                    cross_video_embeddings=cross_video_embeddings,
                    target_fusion_positions=target_fusion_positions,
                    v_lens=v_lens,
                )
            target_end_token = self._build_target_end_token(
                batch_size=x_emb.shape[0],
                v_lens=v_lens,
                dim_model=x_emb.shape[-1],
                device=x_emb.device,
            )
            text_memory, text_memory_lens = self._extract_target_text_memory(
                x_emb=x_emb,
                x=x,
                x_lens=x_lens.to(device=x.device, dtype=torch.long),
            )
            text_token_ids, _ = self._extract_target_text_token_ids(
                x=x,
                x_lens=x_lens.to(device=x.device, dtype=torch.long),
            )
            text_summary = self._masked_mean(text_memory, text_memory_lens.to(device=text_memory.device, dtype=torch.long))
            ref_memory, ref_memory_lens = self._extract_ref_audio_memory(
                y_emb=y_emb,
                ref_lengths=delayed_ref_lengths,
            )
            ref_summary = None
            if ref_memory is not None and ref_memory_lens is not None:
                ref_summary = self._masked_mean(
                    ref_memory,
                    ref_memory_lens.to(device=ref_memory.device, dtype=torch.long),
                )
            plan_outputs = self._build_plan_outputs(
                video_embeddings=cross_video_embeddings,
                video_lens=v_lens,
                text_memory=text_memory,
                text_lens=text_memory_lens,
                text_token_ids=text_token_ids,
                ref_memory=ref_memory,
                ref_lens=ref_memory_lens,
                text_summary=text_summary,
                ref_summary=ref_summary,
            )
            plan_hidden = None if plan_outputs is None else plan_outputs.get("plan_hidden")
        else:
            raw_video_sync_inputs = token_out.get("pre_vq_embeddings", token_out["embeddings"])
            fused_y = embedded_y
            audio_positions = torch.arange(fused_y.shape[1], device=x_emb.device).unsqueeze(0).expand(
                fused_y.shape[0],
                -1,
            )
            y_emb = self.audio_positional_embedding(fused_y, positions=audio_positions)
            if self.use_video_prefix:
                video_tokens_emb = token_out["embeddings"]
                video_positions = self._build_v3_video_positions(
                    delayed_tar_start_ids,
                    v_lens,
                    video_tokens_emb.shape[1],
                    x_emb.device,
                )
                video_tokens_emb = self.audio_positional_embedding(video_tokens_emb, positions=video_positions)
            else:
                video_tokens_emb = None
            target_end_token = None
            plan_outputs = None
            plan_hidden = None

        audio_ref_token = self.audio_ref_token(x_emb.shape[0], device=x_emb.device)
        video_len_token = None
        if self.use_video_len_token and self.video_len_token is not None:
            video_len_token = self.video_len_token(v_lens.to(x_emb.device))
        target_start_token = self.target_start_token(x_emb.shape[0], device=x_emb.device)

        seq_data = self.processor.prepare_sequence_with_video_tokens(
            x_emb,
            x_lens,
            y_emb,
            data["new_y_lens"],
            delayed_ref_lengths,
            video_tokens_emb,
            v_lens.to(x_emb.device) if video_tokens_emb is not None else None,
            audio_ref_token,
            video_len_token,
            target_start_token,
            target_end_token=target_end_token,
        )
        xy_input = seq_data["sequence"]
        seq_lens = seq_data["sequence_lens"]

        xy_pad_mask, xy_attn_mask = self.processor.build_causal_masks(seq_lens)
        in_decoder_layer_metrics: Dict[int, Dict[str, torch.Tensor]] = {}
        plan_hidden_for_decoder = self._shift_plan_hidden(plan_hidden, plan_shift_steps) if self.use_v4 else None
        effective_plan_hidden = None
        if (
            self.use_plan_conditioning
            and plan_hidden_for_decoder is not None
            and not disable_plan_conditioning
        ):
            effective_plan_hidden = plan_hidden_for_decoder
        if self.use_in_decoder_cross_attention and self.use_v4:
            decoder_video_memory = cross_video_embeddings
            if effective_plan_hidden is not None:
                decoder_video_memory = effective_plan_hidden
            y_out, _, in_decoder_layer_metrics = self._decoder_forward_with_in_decoder_video_attention(
                xy_input=xy_input,
                xy_attn_mask=xy_attn_mask,
                xy_pad_mask=xy_pad_mask,
                video_memory=decoder_video_memory,
                target_query_positions=target_fusion_positions,
                target_offsets=seq_data["target_offsets"],
                target_lengths=seq_data["target_lengths"],
                video_lens=v_lens.to(xy_input.device),
            )
        else:
            y_out, _ = self.processor.decoder_forward(
                self.decoder,
                xy_input=xy_input,
                x_lens=x_lens,
                xy_attn_mask=xy_attn_mask,
                xy_pad_mask=xy_pad_mask,
            )

        y_out_ref = self.processor.gather_sequence_outputs(
            y_out,
            seq_data["ref_offsets"],
            seq_data["ref_lengths"],
        )
        ref_logits = self.predict_layer(y_out_ref)
        ref_aligned_logits, ref_aligned_targets = self.processor.post_process_ref_logits(
            logits=ref_logits,
            rearranged_y=data["rearranged_y"],
        )

        y_out_target = self.processor.gather_sequence_outputs(
            y_out,
            seq_data["target_offsets"],
            seq_data["target_lengths"],
        )
        target_codebook_reprs = None
        pooled_audio_repr = None
        real_time_mask = None
        routing_metrics: Dict[str, torch.Tensor] = {}
        routed_target_hidden = None
        if self.use_v4:
            routing_query_positions = target_fusion_positions
            if self.use_per_codebook_real_time_routing:
                routing_query_positions = self._build_per_codebook_routing_positions(
                    shifted_lengths=seq_data["target_lengths"],
                    anchor_lengths=v_lens,
                    max_len=y_out_target.shape[1],
                    device=y_out_target.device,
                )
            routed_target_hidden, routing_metrics = self._apply_codebook_routing(
                target_hidden=y_out_target,
                plan_hidden=effective_plan_hidden,
                query_positions=routing_query_positions,
                video_lens=v_lens,
            )
        codebook_head_inputs = routed_target_hidden
        if self.use_hierarchical_codebook_conditioning:
            if codebook_head_inputs is None:
                codebook_head_inputs = (
                    y_out_target.unsqueeze(1).expand(-1, self.n_codebooks, -1, -1).contiguous()
                )
            codebook_head_inputs, hier_metrics = self._apply_hierarchical_codebook_conditioning(
                codebook_hidden=codebook_head_inputs,
                rearranged_y=data["rearranged_y"],
                target_time_len=y_out_target.shape[1],
            )
            routing_metrics.update(hier_metrics)
        need_interface_state = (
            return_interface_state
            or self.use_sync_loss
            or self.use_ctc_loss
            or self.use_alignment_loss
            or self.use_progress_loss
            or self.use_boundary_loss
            or self.use_q0_controller
            or self.use_q0_planner_state_adapter
            or self.use_g_embedding_loss
        )
        q0_progress_logits = None
        q0_loop_logits = None
        q0_eos_logits = None
        q0_repr = None
        q0_lengths = None
        q0_target_tokens = None
        if need_interface_state and hasattr(self.predict_layer, "forward_with_reprs"):
            logits, raw_target_codebook_reprs = self.predict_layer.forward_with_reprs(
                x=y_out_target,
                codebook_inputs=codebook_head_inputs,
            )
            g_embedding_metrics = self._compute_g_embedding_losses(
                codebook_reprs=raw_target_codebook_reprs,
                rearranged_y=data["rearranged_y"],
            )
            target_codebook_reprs = raw_target_codebook_reprs.permute(0, 2, 1, 3).contiguous()
            if (
                (self.use_q0_controller and self.q0_controller is not None)
                or self.use_q0_planner_state_adapter
                or return_interface_state
            ):
                q0_repr, q0_lengths, q0_target_tokens = self.processor.decode_target_q0_reprs(
                    target_codebook_reprs,
                    data["rearranged_y"],
                )
            if self.use_q0_controller and self.q0_controller is not None and q0_repr is not None:
                q0_ctrl = self.q0_controller(q0_repr)
                q0_progress_logits = q0_ctrl["q0_progress_logits"]
                q0_loop_logits = q0_ctrl["q0_loop_logits"]
                q0_eos_logits = q0_ctrl["q0_eos_logits"]
            pooled_audio_repr, real_time_mask = self._pool_sync_codebook_reprs(
                codebook_reprs=target_codebook_reprs,
                shifted_lengths=seq_data["target_lengths"],
                anchor_lengths=v_lens,
            )
        else:
            logits = self.predict_layer(y_out_target, codebook_inputs=codebook_head_inputs)
            g_embedding_metrics = {}
        aligned_logits, aligned_targets = self.processor.post_process_target_logits(
            logits=logits,
            rearranged_y=data["rearranged_y"],
        )
        aligned_logits, q0_adapter_metrics = self._apply_q0_planner_state_adapter_to_aligned_logits(
            aligned_logits=aligned_logits,
            q0_repr=q0_repr,
            q0_lengths=q0_lengths,
            plan_hidden=plan_hidden,
            plan_outputs=plan_outputs,
        )
        sync_video_inputs = raw_video_sync_inputs
        if self.sync_video_source == "plan_hidden":
            sync_video_inputs = effective_plan_hidden
        elif self.sync_video_source != "raw_video":
            raise ValueError(f"Unsupported sync_video_source: {self.sync_video_source}")

        sync_loss, sync_pos_alignment, sync_neg_alignment = self._compute_sync_loss(
            pooled_audio=pooled_audio_repr,
            sync_mask=real_time_mask,
            video_sync_inputs=sync_video_inputs,
        )

        ctc_logits = None
        ctc_input_lengths = None
        if self.use_ctc_loss and self.ctc_head is not None and pooled_audio_repr is not None and real_time_mask is not None:
            ctc_logits = self.ctc_head(pooled_audio_repr)
            ctc_input_lengths = real_time_mask.sum(dim=1).to(dtype=torch.long)

        align_logits = None
        if self.use_alignment_loss and self.alignment_head is not None and pooled_audio_repr is not None:
            align_logits = self.alignment_head(pooled_audio_repr)

        progress_logits = None
        if self.use_progress_loss and self.progress_head is not None and pooled_audio_repr is not None:
            progress_logits = self.progress_head(pooled_audio_repr)

        boundary_logits = None
        if self.use_boundary_loss and self.boundary_head is not None and pooled_audio_repr is not None:
            boundary_logits = self.boundary_head(pooled_audio_repr)

        plan_progress_logits = None
        if (
            self.use_plan_progress_loss
            and self.plan_progress_head is not None
            and plan_hidden is not None
            and not disable_plan_supervision
        ):
            plan_progress_logits = self.plan_progress_head(plan_hidden)

        plan_boundary_logits = None
        if (
            self.use_plan_boundary_loss
            and not disable_plan_supervision
        ):
            if plan_outputs is not None and plan_outputs.get("plan_boundary_logits") is not None:
                plan_boundary_logits = plan_outputs.get("plan_boundary_logits")
            elif self.plan_boundary_head is not None and plan_hidden is not None:
                plan_boundary_logits = self.plan_boundary_head(plan_hidden)

        plan_acoustic_metrics = self._compute_plan_acoustic_loss(
            plan_hidden=plan_hidden,
            plan_lengths=v_lens.to(dtype=torch.long) if plan_hidden is not None else None,
            rearranged_y=data["rearranged_y"],
        )

        pred_target_tokens = None
        pred_target_lengths = None
        if return_pred_target_tokens:
            pred_target_tokens, pred_target_lengths = self.processor.decode_target_predictions(
                logits=logits,
                rearranged_y=data["rearranged_y"],
            )

        return {
            "logits": aligned_logits,
            "targets": aligned_targets,
            "vq_loss": token_out["loss"],
            "vq_usage": self._compute_vq_usage(token_out.get("token_ids"), v_lens),
            "ref_logits": ref_aligned_logits,
            "ref_targets": ref_aligned_targets,
            "sync_loss": sync_loss,
            "sync_pos_alignment": sync_pos_alignment,
            "sync_neg_alignment": sync_neg_alignment,
            **g_embedding_metrics,
            "ctc_logits": ctc_logits,
            "ctc_input_lengths": ctc_input_lengths,
            "align_logits": align_logits,
            "progress_logits": progress_logits,
            "boundary_logits": boundary_logits,
            "plan_progress_logits": plan_progress_logits,
            "plan_boundary_logits": plan_boundary_logits,
            **plan_acoustic_metrics,
            "plan_input_lengths": v_lens.to(dtype=torch.long) if plan_hidden is not None else None,
            "plan_text_lengths": text_memory_lens.to(dtype=torch.long) if self.use_v4 and plan_hidden is not None else None,
            "interface_input_lengths": real_time_mask.sum(dim=1).to(dtype=torch.long) if real_time_mask is not None else None,
            "pooled_audio_repr": pooled_audio_repr if return_interface_state else None,
            "pooled_audio_mask": real_time_mask if return_interface_state else None,
            "plan_hidden": plan_hidden if return_interface_state else None,
            "plan_text_expected_positions": None if plan_outputs is None else plan_outputs.get("plan_text_expected_positions"),
            "plan_text_entropy": None if plan_outputs is None else plan_outputs.get("plan_text_entropy"),
            "plan_cursor_logits": None if plan_outputs is None else plan_outputs.get("plan_cursor_logits"),
            "plan_cursor_entropy": None if plan_outputs is None else plan_outputs.get("plan_cursor_entropy"),
            "plan_monotonic_loss": None if plan_outputs is None else plan_outputs.get("plan_monotonic_loss"),
            "plan_remaining_pred": None if plan_outputs is None else plan_outputs.get("plan_remaining_pred"),
            "plan_stop_logits": None if plan_outputs is None else plan_outputs.get("plan_stop_logits"),
            "plan_activity_logits": None if plan_outputs is None else plan_outputs.get("plan_activity_logits"),
            "plan_viseme_logits": None if plan_outputs is None else plan_outputs.get("plan_viseme_logits"),
            "plan_phone_logits": None if plan_outputs is None else plan_outputs.get("plan_phone_logits"),
            "plan_occurrence_pred": None if plan_outputs is None else plan_outputs.get("plan_occurrence_pred"),
            "plan_viseme_compatibility_mean": None if plan_outputs is None else plan_outputs.get("plan_viseme_compatibility_mean"),
            "plan_gate_means": None if plan_outputs is None else plan_outputs.get("plan_gate_means"),
            "routing_lambdas": routing_metrics.get("routing_lambdas"),
            "routing_valid_mask": routing_metrics.get("routing_valid_mask"),
            "route_scale_values": routing_metrics.get("route_scale_values"),
            "hier_context_valid_mask": routing_metrics.get("hier_context_valid_mask"),
            "hier_context_norm_means": routing_metrics.get("hier_context_norm_means"),
            "hier_context_scales": routing_metrics.get("hier_context_scales"),
            "q0_progress_logits": q0_progress_logits,
            "q0_loop_logits": q0_loop_logits,
            "q0_eos_logits": q0_eos_logits,
            "q0_output_lengths": q0_lengths,
            "q0_target_tokens": q0_target_tokens,
            "q0_repr": q0_repr if return_interface_state else None,
            "q0_planner_adapter_delta_norm": q0_adapter_metrics.get("q0_planner_adapter_delta_norm"),
            "q0_planner_adapter_gate": q0_adapter_metrics.get("q0_planner_adapter_gate"),
            "pred_target_tokens": pred_target_tokens,
            "pred_target_lengths": pred_target_lengths,
            "in_decoder_gate_values": {
                layer_idx: layer_metrics["gate_value"]
                for layer_idx, layer_metrics in in_decoder_layer_metrics.items()
            },
            "in_decoder_fusion_delta_norms": {
                layer_idx: layer_metrics["fusion_delta_norm"]
                for layer_idx, layer_metrics in in_decoder_layer_metrics.items()
            },
        }

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        exposure_metrics = None
        if self.training and not bool(batch.get("_disable_exposure_mitigation", False)):
            batch, exposure_metrics = self._prepare_training_batch_with_exposure_mitigation(batch)
        outputs = self._forward_impl(batch)
        if exposure_metrics is not None:
            outputs.update(exposure_metrics)
        return outputs

    def _apply_visual_decode_bias(
        self,
        logits: torch.Tensor,
        step: int,
        plan_outputs: Optional[Dict[str, torch.Tensor]],
        silence_tokens: list[int],
    ) -> torch.Tensor:
        if (
            not self.use_visual_decode_bias
            or self.visual_decode_log_bias is None
            or plan_outputs is None
            or self.visual_decode_bias_weight == 0.0
        ):
            return logits

        viseme_logits = plan_outputs.get("plan_viseme_logits")
        if viseme_logits is None or viseme_logits.shape[1] == 0:
            return logits

        plan_idx = max(0, min(int(step), viseme_logits.shape[1] - 1))
        viseme_probs = torch.softmax(viseme_logits[:, plan_idx, :].float(), dim=-1)
        log_bias = self.visual_decode_log_bias.to(device=logits.device, dtype=logits.dtype)
        if log_bias.shape[0] != viseme_probs.shape[-1]:
            raise ValueError(
                f"Visual decode prior class mismatch: prior={log_bias.shape[0]}, "
                f"planner={viseme_probs.shape[-1]}"
            )
        if log_bias.shape[1] != logits.shape[-1]:
            if log_bias.shape[1] > logits.shape[-1]:
                log_bias = log_bias[:, : logits.shape[-1]]
            else:
                pad = log_bias.new_zeros((log_bias.shape[0], logits.shape[-1] - log_bias.shape[1]))
                log_bias = torch.cat([log_bias, pad], dim=1)

        token_bias = torch.matmul(viseme_probs.to(dtype=log_bias.dtype), log_bias).squeeze(0)
        if self.visual_decode_bias_clamp > 0:
            token_bias = token_bias.clamp(
                min=-self.visual_decode_bias_clamp,
                max=self.visual_decode_bias_clamp,
            )
        logits = logits.clone()
        logits[0] = logits[0] + token_bias * self.visual_decode_bias_weight

        if self.visual_decode_activity_silence_weight != 0.0:
            activity_logits = plan_outputs.get("plan_activity_logits")
            if activity_logits is not None and activity_logits.shape[1] > 0:
                activity_idx = max(0, min(int(step), activity_logits.shape[1] - 1))
                activity_prob = torch.sigmoid(activity_logits[:, activity_idx].float()).to(
                    device=logits.device,
                    dtype=logits.dtype,
                )[0]
                silence_shift = (0.5 - activity_prob) * self.visual_decode_activity_silence_weight
                for token_id in silence_tokens:
                    if 0 <= int(token_id) < logits.shape[-1]:
                        logits[0, int(token_id)] = logits[0, int(token_id)] + silence_shift
        return logits

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
        top_k: int = -100,
        top_p: float = 1.0,
        temperature: float = 1.0,
        stop_repetition: int = 3,
        kvcache: int = 1,
        silence_tokens: list[int] = [1388, 1898, 131],
        *kargs,
    ) -> torch.Tensor:
        assert x.shape[0] == 1, "Batch size must be 1 for this method"
        assert x.ndim == 2, x.shape
        assert y.ndim == 3, y.shape
        if x_lens.device != x.device:
            x_lens = x_lens.to(x.device)
        if y.device != x.device:
            y = y.to(x.device)
        if v.device != x.device:
            v = v.to(x.device)

        x_input = self.text_embedding(x)
        x_input = self.text_positional_embedding(x_input)
        x_input = self._add_text_segment_embeddings(x_input, x, x_lens)

        prep_data = self.processor.prepare_generate_inputs(y, v, self.audio_embedding)
        embedded_y = prep_data["embedded_y"]
        v_lens = prep_data["v_lens"]
        v_upsampled = prep_data["v_upsampled"]

        y_len = y.shape[2]
        ref_length_delayed = torch.LongTensor([y_len + self.n_codebooks]).to(x.device)
        pred_start_ids_delayed = ref_length_delayed

        token_out = self.video_tokenizer(v_upsampled, v_lens)
        if self.use_v4:
            target_shifted_lengths = prep_data["new_y_lens"] - ref_length_delayed
            y_emb, _ = self._prepare_v4_audio_embeddings(
                embedded_y=embedded_y,
                delayed_ref_lengths=ref_length_delayed,
                split_lengths=torch.LongTensor([y_len]).to(x.device),
                target_shifted_lengths=target_shifted_lengths,
                target_anchor_lens=v_lens,
            )
            cross_video_embeddings, _ = self._prepare_v4_video_embeddings(
                token_out.get("pre_vq_embeddings", token_out["embeddings"]),
                v_lens,
            )
            if self.use_video_prefix:
                video_tokens_emb, _ = self._prepare_v4_video_embeddings(token_out["embeddings"], v_lens)
            else:
                video_tokens_emb = None
            target_end_token = self._build_target_end_token(
                batch_size=1,
                v_lens=v_lens,
                dim_model=x_input.shape[-1],
                device=x.device,
            )
            blank_emb = self.processor.get_empty_embedding(self.audio_embedding)
            blank_emb = self._add_type_embedding(blank_emb, self.target_audio_segment_embedding)
            text_memory, text_memory_lens = self._extract_target_text_memory(
                x_emb=x_input,
                x=x,
                x_lens=x_lens.to(device=x.device, dtype=torch.long),
            )
            text_token_ids, _ = self._extract_target_text_token_ids(
                x=x,
                x_lens=x_lens.to(device=x.device, dtype=torch.long),
            )
            text_summary = self._masked_mean(
                text_memory,
                text_memory_lens.to(device=text_memory.device, dtype=torch.long),
            )
            ref_memory, ref_memory_lens = self._extract_ref_audio_memory(
                y_emb=y_emb,
                ref_lengths=ref_length_delayed,
            )
            ref_summary = self._masked_mean(
                ref_memory,
                ref_memory_lens.to(device=ref_memory.device, dtype=torch.long),
            ) if ref_memory is not None and ref_memory_lens is not None else None
            plan_outputs = self._build_plan_outputs(
                video_embeddings=cross_video_embeddings,
                video_lens=v_lens,
                text_memory=text_memory,
                text_lens=text_memory_lens,
                text_token_ids=text_token_ids,
                ref_memory=ref_memory,
                ref_lens=ref_memory_lens,
                text_summary=text_summary,
                ref_summary=ref_summary,
            )
            plan_hidden = None if plan_outputs is None else plan_outputs.get("plan_hidden")
        else:
            audio_positions = torch.arange(embedded_y.shape[1], device=x.device).unsqueeze(0)
            y_emb = self.audio_positional_embedding(embedded_y, positions=audio_positions)
            if self.use_video_prefix:
                video_tokens_emb = token_out["embeddings"]
                video_positions = self._build_v3_video_positions(
                    pred_start_ids_delayed,
                    v_lens,
                    video_tokens_emb.shape[1],
                    x.device,
                )
                video_tokens_emb = self.audio_positional_embedding(video_tokens_emb, positions=video_positions)
            else:
                video_tokens_emb = None
            target_end_token = None
            blank_emb = None
            plan_hidden = None

        audio_ref_token = self.audio_ref_token(x_input.shape[0], device=x.device)
        video_len_token = None
        if self.use_video_len_token and self.video_len_token is not None:
            video_len_token = self.video_len_token(v_lens.to(x.device))
        target_start_token = self.target_start_token(x_input.shape[0], device=x.device)

        seq_data = self.processor.prepare_sequence_with_video_tokens(
            x_input,
            x_lens,
            y_emb,
            prep_data["new_y_lens"],
            ref_length_delayed,
            video_tokens_emb,
            v_lens.to(x.device) if video_tokens_emb is not None else None,
            audio_ref_token,
            video_len_token,
            target_start_token,
            target_end_token=target_end_token,
        )
        curr_xy = seq_data["sequence"]
        if self.use_v4 and blank_emb is not None:
            curr_xy = torch.cat([curr_xy, blank_emb], dim=1)

        codebook_done = [False] * self.n_codebooks
        consec_silence_count = 0
        prev_token = None
        curr_generated = [[] for _ in range(self.n_codebooks)]

        num_decoder_layers = len(self.decoder.layers)
        past = torch.ones([num_decoder_layers, 2, 1, x.shape[0]], device=x.device, dtype=torch.float32) if kvcache else None
        total_generate_len = int(v_lens[0].item())
        max_anchor_pos = max(total_generate_len - 1, 0)

        for step in range(total_generate_len + self.n_codebooks + 2):
            seq_lens = torch.LongTensor([curr_xy.shape[1]]).to(x.device)
            xy_pad_mask, xy_attn_mask = self.processor.build_causal_masks(seq_lens)
            if self.use_in_decoder_cross_attention and self.use_v4:
                curr_target_len = curr_xy.shape[1] - int(seq_data["target_offsets"][0].item())
                target_lengths = torch.LongTensor([max(curr_target_len, 0)]).to(x.device)
                target_positions = self._build_generate_target_query_positions(
                    target_lengths=target_lengths,
                    anchor_lengths=v_lens.to(x.device),
                    device=x.device,
                )
                decoder_video_memory = cross_video_embeddings
                if self.use_plan_conditioning and plan_hidden is not None:
                    decoder_video_memory = plan_hidden
                out, present, _ = self._decoder_forward_with_in_decoder_video_attention(
                    xy_input=curr_xy,
                    xy_attn_mask=xy_attn_mask,
                    xy_pad_mask=xy_pad_mask,
                    video_memory=decoder_video_memory,
                    target_query_positions=target_positions,
                    target_offsets=seq_data["target_offsets"],
                    target_lengths=target_lengths,
                    video_lens=v_lens.to(x.device),
                    past=past if kvcache else None,
                )
            else:
                out, present = self.processor.decoder_forward(
                    self.decoder,
                    xy_input=curr_xy,
                    x_lens=x_lens,
                    xy_attn_mask=xy_attn_mask,
                    xy_pad_mask=xy_pad_mask,
                    past=past,
                    last_n_tokens=1,
                )

            if past is not None:
                past = torch.cat([past, present.to(past.dtype)], dim=-2) if past.ndim > 3 else present.to(past.dtype)

            last_hidden = out[:, -1:, :]
            routed_last_hidden = None
            if self.use_v4 and self.use_codebook_routing and self.plan_router is not None and self.use_plan_conditioning and plan_hidden is not None:
                if self.use_per_codebook_real_time_routing:
                    current_query_positions_all = self._build_per_codebook_routing_positions(
                        shifted_lengths=target_lengths,
                        anchor_lengths=v_lens.to(x.device),
                        max_len=int(target_lengths.max().item()) if target_lengths.numel() > 0 else 0,
                        device=x.device,
                    )
                    current_query_positions = (
                        current_query_positions_all[:, -1:, :]
                        if current_query_positions_all.shape[1] > 0
                        else None
                    )
                else:
                    current_query_positions = target_positions[:, -1:] if target_positions.shape[1] > 0 else None
                if current_query_positions is not None:
                    routed_last_hidden, _ = self._apply_codebook_routing(
                        target_hidden=last_hidden,
                        plan_hidden=plan_hidden,
                        query_positions=current_query_positions,
                        video_lens=v_lens,
                    )
            codebook_step_inputs = routed_last_hidden
            if self.use_hierarchical_codebook_conditioning and self.hierarchical_codebook_conditioner is not None:
                if codebook_step_inputs is None:
                    codebook_step_inputs = (
                        last_hidden.unsqueeze(1).expand(-1, self.n_codebooks, -1, -1).contiguous()
                    )
                codebook_step_inputs, _ = self.hierarchical_codebook_conditioner.forward_step(
                    base_hidden_step=codebook_step_inputs,
                    curr_generated=curr_generated,
                    step=step,
                    audio_embedding=self.audio_embedding,
                    device=x.device,
                )

            q0_loop_prob = None
            q0_eos_prob = None
            if hasattr(self.predict_layer, "forward_with_reprs"):
                step_logits, step_reprs = self.predict_layer.forward_with_reprs(
                    x=last_hidden,
                    codebook_inputs=codebook_step_inputs,
                )
                logits = step_logits[:, :, -1, :].squeeze(0)
                q0_step_repr = step_reprs[:, 0, :, :]
                if self.use_q0_controller and self.q0_controller is not None:
                    q0_ctrl = self.q0_controller(q0_step_repr)
                    q0_loop_prob = torch.sigmoid(q0_ctrl["q0_loop_logits"][:, -1, 0])
                    q0_eos_prob = torch.sigmoid(q0_ctrl["q0_eos_logits"][:, -1, 0])
                if self.use_q0_planner_state_adapter and self.q0_planner_state_adapter is not None:
                    q0_step_repr, _ = self._apply_q0_planner_state_adapter_step(
                        q0_step_repr=q0_step_repr,
                        step=step,
                        plan_hidden=plan_hidden,
                        plan_outputs=plan_outputs,
                    )
                    logits[0] = self.predict_layer.heads[0].classifier(q0_step_repr[:, -1, :]).squeeze(0)
            else:
                logits = self.predict_layer(last_hidden, codebook_inputs=codebook_step_inputs).view(self.n_codebooks, -1)

            if q0_loop_prob is not None and len(curr_generated[0]) > 0:
                if float(q0_loop_prob[0].item()) > self.q0_loop_threshold:
                    recent_tokens = curr_generated[0][-3:]
                    for token_id in set(recent_tokens):
                        logits[0, token_id] -= self.q0_loop_logit_penalty
            if q0_eos_prob is not None and float(q0_eos_prob[0].item()) < self.q0_eos_threshold:
                logits[0, self.processor.eog_id] -= self.q0_eog_logit_penalty
            logits = self._apply_visual_decode_bias(
                logits=logits,
                step=step,
                plan_outputs=plan_outputs,
                silence_tokens=silence_tokens,
            )
            n_eog = sum(codebook_done)
            curr_y_len = pred_start_ids_delayed + step
            samples, codebook_done, prev_token, consec_silence_count = self.processor.sample_codebooks_token(
                n_eog,
                logits,
                codebook_done,
                top_k,
                top_p,
                temperature,
                prev_token,
                consec_silence_count,
                stop_repetition,
                silence_tokens,
                curr_y_len,
                x_lens,
                step,
                total_generate_len,
            )

            if sum(codebook_done) == self.n_codebooks:
                break

            for k in range(self.n_codebooks):
                curr_generated[k].append(samples[k].item())

            samples = samples.view(1, self.n_codebooks, 1)
            next_emb = self.audio_embedding(samples).sum(dim=1)
            if self.use_v4:
                next_pos = torch.full((1, 1), min(step, max_anchor_pos), device=x.device, dtype=torch.long)
                next_emb = self.shared_audio_positional_embedding(next_emb, positions=next_pos)
                next_emb = self._add_type_embedding(next_emb, self.target_audio_segment_embedding)
            else:
                next_pos = torch.full(
                    (1, 1),
                    int(pred_start_ids_delayed[0].item()) + step,
                    device=x.device,
                    dtype=torch.long,
                )
                next_emb = self.audio_positional_embedding(next_emb, positions=next_pos)
            curr_xy = torch.cat([curr_xy, next_emb], dim=1)

        curr_generated = torch.tensor(curr_generated, dtype=torch.long, device=x.device)
        res, unshifted_generated = self.processor.reconstruct_sequence(y, curr_generated)
        return res, unshifted_generated
