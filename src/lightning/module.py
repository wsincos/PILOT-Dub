import torch
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from typing import Any, Dict, Optional, Tuple, Union
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import torch.nn as nn

from src.modeling.nets.voicecraft_dub import VoiceCraftDubModel
from src.modeling.utils import load_ckpt_from_origin
from src.modeling.losses.voicecraft_loss import VoiceCraftLoss
from src.lightning.utils.optim import ScaledAdam, Eden

import logging
logger = logging.getLogger(__name__)

class VoiceCraftDubLightningModule(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.n_codebooks = cfg.n_codebooks
        self.codebook_weights = cfg.get("codebook_weights", [1.0] * self.n_codebooks)
        self.accuracy_metrics = instantiate(cfg.accuracy_metrics)
        self.is_scaled_adam = "ScaledAdam" in self.cfg.optimizer.get("_target_", "")
        self._setup_model()
        self._apply_parameter_training_rules()
        self._align_label_to_viseme_table = None
        self.ctc_loss_fn = None
        if self.cfg.get("use_ctc_loss", False):
            self.ctc_loss_fn = nn.CTCLoss(
                blank=int(self.cfg.get("ctc_blank_id")),
                reduction="mean",
                zero_infinity=True,
            )
        self.save_hyperparameters(ignore=["model"])
        
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(batch)

    def compute_loss_and_metrics(
        self,
        logits,
        targets,
        vq_loss: Optional[torch.Tensor] = None,
        vq_usage: Optional[torch.Tensor] = None,
        ref_logits: Optional[torch.Tensor] = None,
        ref_targets: Optional[torch.Tensor] = None,
        sync_loss: Optional[torch.Tensor] = None,
        sync_pos_alignment: Optional[torch.Tensor] = None,
        sync_neg_alignment: Optional[torch.Tensor] = None,
        g_embedding_cos_loss: Optional[torch.Tensor] = None,
        g_embedding_ce_loss: Optional[torch.Tensor] = None,
        g_embedding_cos_mean: Optional[torch.Tensor] = None,
        g_embedding_top1: Optional[torch.Tensor] = None,
        g_embedding_top10: Optional[torch.Tensor] = None,
        ctc_loss: Optional[torch.Tensor] = None,
        align_loss: Optional[torch.Tensor] = None,
        align_frame_acc: Optional[torch.Tensor] = None,
        progress_loss: Optional[torch.Tensor] = None,
        progress_frame_acc: Optional[torch.Tensor] = None,
        boundary_loss: Optional[torch.Tensor] = None,
        boundary_frame_acc: Optional[torch.Tensor] = None,
        plan_progress_loss: Optional[torch.Tensor] = None,
        plan_progress_frame_acc: Optional[torch.Tensor] = None,
        plan_boundary_loss: Optional[torch.Tensor] = None,
        plan_boundary_frame_acc: Optional[torch.Tensor] = None,
        plan_acoustic_loss: Optional[torch.Tensor] = None,
        plan_acoustic_pos_alignment: Optional[torch.Tensor] = None,
        plan_acoustic_neg_alignment: Optional[torch.Tensor] = None,
        plan_cursor_loss: Optional[torch.Tensor] = None,
        plan_cursor_frame_acc: Optional[torch.Tensor] = None,
        plan_monotonic_loss: Optional[torch.Tensor] = None,
        plan_remaining_loss: Optional[torch.Tensor] = None,
        plan_stop_loss: Optional[torch.Tensor] = None,
        plan_stop_frame_acc: Optional[torch.Tensor] = None,
        plan_activity_loss: Optional[torch.Tensor] = None,
        plan_activity_frame_acc: Optional[torch.Tensor] = None,
        plan_viseme_loss: Optional[torch.Tensor] = None,
        plan_viseme_frame_acc: Optional[torch.Tensor] = None,
        plan_phone_loss: Optional[torch.Tensor] = None,
        plan_phone_frame_acc: Optional[torch.Tensor] = None,
        plan_occurrence_loss: Optional[torch.Tensor] = None,
        q0_progress_loss: Optional[torch.Tensor] = None,
        q0_progress_frame_acc: Optional[torch.Tensor] = None,
        q0_loop_loss: Optional[torch.Tensor] = None,
        q0_loop_frame_acc: Optional[torch.Tensor] = None,
        q0_eos_loss: Optional[torch.Tensor] = None,
        q0_eos_frame_acc: Optional[torch.Tensor] = None,
        state_loss: Optional[torch.Tensor] = None,
        state_huber_loss: Optional[torch.Tensor] = None,
        state_cos_loss: Optional[torch.Tensor] = None,
        state_cosine: Optional[torch.Tensor] = None,
        state_l2: Optional[torch.Tensor] = None,
        state_valid_ratio: Optional[torch.Tensor] = None,
        late_nonblank_penalty: Optional[torch.Tensor] = None,
        ctc_greedy_per: Optional[torch.Tensor] = None,
        ctc_prefix_per: Optional[torch.Tensor] = None,
        ctc_blank_ratio: Optional[torch.Tensor] = None,
        ctc_early_blank_ratio: Optional[torch.Tensor] = None,
        ctc_late_blank_ratio: Optional[torch.Tensor] = None,
        ctc_greedy_len_ratio: Optional[torch.Tensor] = None,
        ctc_tail_nonblank_ratio: Optional[torch.Tensor] = None,
        completion_before_tail_ratio: Optional[torch.Tensor] = None,
        forced_cutoff_risk: Optional[torch.Tensor] = None,
        video_ablation_gap: Optional[torch.Tensor] = None,
        **extra_metrics: Any,
    ):
        loss_list = []
        ntokens_list = []
        topk_acc_list = []

        # 1. calculate each codebook's loss and acc
        for k in range(self.n_codebooks):
            logit = logits[k]   # [T,V]
            target = targets[k] # [T]
            
            loss = F.cross_entropy(logit, target, reduction='sum', label_smoothing=float(self.cfg.get("ce_label_smoothing", 0.0)))
            if target.numel() > 0:
                acc = self.accuracy_metrics[k](logit, target)
            else:
                acc = torch.tensor(0.0, device=logit.device)
            # acc = self.accuracy_metrics[k](logit, target)
            
            loss_list.append(loss)
            ntokens_list.append(len(target))
            topk_acc_list.append(acc)

        all_ntokens = sum(ntokens_list)
        ce_loss_sum = sum([l * cw for l, cw in zip(loss_list, self.codebook_weights)])
        ce_loss = ce_loss_sum / (all_ntokens + 1e-6)
        ce_weight = float(self.cfg.get("ce_loss_weight", 1.0))
        total_loss = ce_loss * ce_weight
        if vq_loss is not None:
            vq_weight = float(self.cfg.get("vq_loss_weight", 1.0))
            total_loss = total_loss + vq_loss * vq_weight
        topk_acc_by_codebook = [topk_acc * nt for topk_acc, nt in zip(topk_acc_list, ntokens_list)]
        total_topk_acc = sum(topk_acc_by_codebook)

        ref_ce_loss = None
        ref_topk_acc = None
        if ref_logits is not None and ref_targets is not None and ref_targets.numel() > 0:
            ref_loss_list = []
            ref_ntokens_list = []
            ref_topk_acc_list = []
            for k in range(self.n_codebooks):
                ref_logit = ref_logits[k]
                ref_target = ref_targets[k]
                if ref_target.numel() > 0:
                    ref_loss = F.cross_entropy(ref_logit, ref_target, reduction='sum', label_smoothing=float(self.cfg.get("ce_label_smoothing", 0.0)))
                    ref_acc = self.accuracy_metrics[k](ref_logit, ref_target)
                    ref_ntokens = len(ref_target)
                else:
                    ref_loss = torch.tensor(0.0, device=ref_logit.device)
                    ref_acc = torch.tensor(0.0, device=ref_logit.device)
                    ref_ntokens = 0
                ref_loss_list.append(ref_loss)
                ref_ntokens_list.append(ref_ntokens)
                ref_topk_acc_list.append(ref_acc)

            ref_all_ntokens = sum(ref_ntokens_list)
            if ref_all_ntokens > 0:
                ref_loss_sum = sum([l * cw for l, cw in zip(ref_loss_list, self.codebook_weights)])
                ref_ce_loss = ref_loss_sum / (ref_all_ntokens + 1e-6)
                ref_weight = float(self.cfg.get("ref_consistency_weight", 0.0))
                total_loss = total_loss + ref_ce_loss * ref_weight
                ref_topk_acc_by_codebook = [acc * nt for acc, nt in zip(ref_topk_acc_list, ref_ntokens_list)]
                ref_topk_acc = sum(ref_topk_acc_by_codebook) / (ref_all_ntokens + 1e-6)

        if sync_loss is not None:
            sync_weight = float(self.cfg.get("sync_loss_weight", 0.0))
            total_loss = total_loss + sync_loss * sync_weight
        if g_embedding_cos_loss is not None:
            g_emb_weight = float(self.cfg.get("g_embedding_loss_weight", 0.0))
            total_loss = total_loss + g_embedding_cos_loss * g_emb_weight
        if g_embedding_ce_loss is not None:
            g_emb_ce_weight = float(self.cfg.get("g_embedding_ce_loss_weight", 0.0))
            total_loss = total_loss + g_embedding_ce_loss * g_emb_ce_weight
        if ctc_loss is not None:
            ctc_weight = float(self.cfg.get("ctc_loss_weight", 0.0))
            total_loss = total_loss + ctc_loss * ctc_weight
        if align_loss is not None:
            align_weight = float(self.cfg.get("align_loss_weight", 0.0))
            total_loss = total_loss + align_loss * align_weight
        if progress_loss is not None:
            progress_weight = float(self.cfg.get("progress_loss_weight", 0.0))
            total_loss = total_loss + progress_loss * progress_weight
        if boundary_loss is not None:
            boundary_weight = float(self.cfg.get("boundary_loss_weight", 0.0))
            total_loss = total_loss + boundary_loss * boundary_weight
        if plan_progress_loss is not None:
            plan_progress_weight = float(self.cfg.get("plan_progress_loss_weight", 0.0))
            total_loss = total_loss + plan_progress_loss * plan_progress_weight
        if plan_boundary_loss is not None:
            plan_boundary_weight = float(self.cfg.get("plan_boundary_loss_weight", 0.0))
            total_loss = total_loss + plan_boundary_loss * plan_boundary_weight
        if plan_acoustic_loss is not None:
            plan_acoustic_weight = float(self.cfg.get("plan_acoustic_loss_weight", 0.0))
            total_loss = total_loss + plan_acoustic_loss * plan_acoustic_weight
        if plan_cursor_loss is not None:
            plan_cursor_weight = float(self.cfg.get("plan_cursor_loss_weight", 0.0))
            total_loss = total_loss + plan_cursor_loss * plan_cursor_weight
        if plan_monotonic_loss is not None:
            plan_mono_weight = float(self.cfg.get("plan_monotonic_loss_weight", 0.0))
            total_loss = total_loss + plan_monotonic_loss * plan_mono_weight
        if plan_remaining_loss is not None:
            plan_remaining_weight = float(self.cfg.get("plan_remaining_loss_weight", 0.0))
            total_loss = total_loss + plan_remaining_loss * plan_remaining_weight
        if plan_stop_loss is not None:
            plan_stop_weight = float(self.cfg.get("plan_stop_loss_weight", 0.0))
            total_loss = total_loss + plan_stop_loss * plan_stop_weight
        if plan_activity_loss is not None:
            plan_activity_weight = float(self.cfg.get("plan_activity_loss_weight", 0.0))
            total_loss = total_loss + plan_activity_loss * plan_activity_weight
        if plan_viseme_loss is not None:
            plan_viseme_weight = float(self.cfg.get("plan_viseme_loss_weight", 0.0))
            total_loss = total_loss + plan_viseme_loss * plan_viseme_weight
        if plan_phone_loss is not None:
            plan_phone_weight = float(self.cfg.get("plan_phone_loss_weight", 0.0))
            total_loss = total_loss + plan_phone_loss * plan_phone_weight
        if plan_occurrence_loss is not None:
            plan_occ_weight = float(self.cfg.get("plan_occurrence_loss_weight", 0.0))
            total_loss = total_loss + plan_occurrence_loss * plan_occ_weight
        if q0_progress_loss is not None:
            q0_progress_weight = float(self.cfg.get("q0_progress_loss_weight", 0.0))
            total_loss = total_loss + q0_progress_loss * q0_progress_weight
        if q0_loop_loss is not None:
            q0_loop_weight = float(self.cfg.get("q0_loop_loss_weight", 0.0))
            total_loss = total_loss + q0_loop_loss * q0_loop_weight
        if q0_eos_loss is not None:
            q0_eos_weight = float(self.cfg.get("q0_eos_loss_weight", 0.0))
            total_loss = total_loss + q0_eos_loss * q0_eos_weight
        if state_loss is not None:
            state_weight = float(self.cfg.get("state_loss_weight", 0.0))
            total_loss = total_loss + state_loss * state_weight
        if late_nonblank_penalty is not None:
            late_weight = float(self.cfg.get("late_nonblank_weight", 0.0))
            total_loss = total_loss + late_nonblank_penalty * late_weight
        
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "ce_loss_sum": ce_loss_sum,
            "topk_acc": total_topk_acc,
            "topk_acc_by_codebook": topk_acc_by_codebook,
            "effective_ntoken": torch.tensor(all_ntokens).to(logits[0].device),
            "vq_loss": vq_loss,
            "vq_usage": vq_usage,
            "ref_ce_loss": ref_ce_loss,
            "ref_topk_acc": ref_topk_acc,
            "target_loss_step": ce_loss,
            "sync_loss": sync_loss,
            "sync_pos_alignment": sync_pos_alignment,
            "sync_neg_alignment": sync_neg_alignment,
            "g_embedding_cos_loss": g_embedding_cos_loss,
            "g_embedding_ce_loss": g_embedding_ce_loss,
            "g_embedding_cos_mean": g_embedding_cos_mean,
            "g_embedding_top1": g_embedding_top1,
            "g_embedding_top10": g_embedding_top10,
            "ctc_loss": ctc_loss,
            "align_loss": align_loss,
            "align_frame_acc": align_frame_acc,
            "progress_loss": progress_loss,
            "progress_frame_acc": progress_frame_acc,
            "boundary_loss": boundary_loss,
            "boundary_frame_acc": boundary_frame_acc,
            "plan_progress_loss": plan_progress_loss,
            "plan_progress_frame_acc": plan_progress_frame_acc,
            "plan_boundary_loss": plan_boundary_loss,
            "plan_boundary_frame_acc": plan_boundary_frame_acc,
            "plan_acoustic_loss": plan_acoustic_loss,
            "plan_acoustic_pos_alignment": plan_acoustic_pos_alignment,
            "plan_acoustic_neg_alignment": plan_acoustic_neg_alignment,
            "plan_cursor_loss": plan_cursor_loss,
            "plan_cursor_frame_acc": plan_cursor_frame_acc,
            "plan_monotonic_loss": plan_monotonic_loss,
            "plan_remaining_loss": plan_remaining_loss,
            "plan_stop_loss": plan_stop_loss,
            "plan_stop_frame_acc": plan_stop_frame_acc,
            "plan_activity_loss": plan_activity_loss,
            "plan_activity_frame_acc": plan_activity_frame_acc,
            "plan_viseme_loss": plan_viseme_loss,
            "plan_viseme_frame_acc": plan_viseme_frame_acc,
            "plan_phone_loss": plan_phone_loss,
            "plan_phone_frame_acc": plan_phone_frame_acc,
            "plan_occurrence_loss": plan_occurrence_loss,
            "q0_progress_loss": q0_progress_loss,
            "q0_progress_frame_acc": q0_progress_frame_acc,
            "q0_loop_loss": q0_loop_loss,
            "q0_loop_frame_acc": q0_loop_frame_acc,
            "q0_eos_loss": q0_eos_loss,
            "q0_eos_frame_acc": q0_eos_frame_acc,
            "state_loss": state_loss,
            "state_huber_loss": state_huber_loss,
            "state_cos_loss": state_cos_loss,
            "state_cosine": state_cosine,
            "state_l2": state_l2,
            "state_valid_ratio": state_valid_ratio,
            "late_nonblank_penalty": late_nonblank_penalty,
            "ctc_greedy_per": ctc_greedy_per,
            "ctc_prefix_per": ctc_prefix_per,
            "ctc_blank_ratio": ctc_blank_ratio,
            "ctc_early_blank_ratio": ctc_early_blank_ratio,
            "ctc_late_blank_ratio": ctc_late_blank_ratio,
            "ctc_greedy_len_ratio": ctc_greedy_len_ratio,
            "ctc_tail_nonblank_ratio": ctc_tail_nonblank_ratio,
            "completion_before_tail_ratio": completion_before_tail_ratio,
            "forced_cutoff_risk": forced_cutoff_risk,
            "video_ablation_gap": video_ablation_gap,
            **{
                k: v
                for k, v in extra_metrics.items()
                if v is None or torch.is_tensor(v)
            },
        }

    @staticmethod
    def _edit_distance(seq_a: list[int], seq_b: list[int]) -> int:
        if len(seq_a) == 0:
            return len(seq_b)
        if len(seq_b) == 0:
            return len(seq_a)
        dp = list(range(len(seq_b) + 1))
        for i, a in enumerate(seq_a, start=1):
            prev_diag = dp[0]
            dp[0] = i
            for j, b in enumerate(seq_b, start=1):
                temp = dp[j]
                cost = 0 if a == b else 1
                dp[j] = min(
                    dp[j] + 1,
                    dp[j - 1] + 1,
                    prev_diag + cost,
                )
                prev_diag = temp
        return dp[-1]

    @staticmethod
    def _lcs_length(seq_a: list[int], seq_b: list[int]) -> int:
        if len(seq_a) == 0 or len(seq_b) == 0:
            return 0
        dp = [0] * (len(seq_b) + 1)
        for a in seq_a:
            prev_diag = 0
            for j, b in enumerate(seq_b, start=1):
                temp = dp[j]
                if a == b:
                    dp[j] = prev_diag + 1
                else:
                    dp[j] = max(dp[j], dp[j - 1])
                prev_diag = temp
        return dp[-1]

    def _compute_ctc_metrics(
        self,
        ctc_logits: Optional[torch.Tensor],
        ctc_input_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if ctc_logits is None or ctc_input_lengths is None:
            return {
                "ctc_greedy_per": None,
                "ctc_prefix_per": None,
                "ctc_blank_ratio": None,
                "ctc_early_blank_ratio": None,
                "ctc_late_blank_ratio": None,
                "ctc_greedy_len_ratio": None,
                "ctc_tail_nonblank_ratio": None,
                "completion_before_tail_ratio": None,
                "forced_cutoff_risk": None,
            }
        if "ctc_labels" not in batch or "ctc_label_lens" not in batch:
            return {
                "ctc_greedy_per": None,
                "ctc_prefix_per": None,
                "ctc_blank_ratio": None,
                "ctc_early_blank_ratio": None,
                "ctc_late_blank_ratio": None,
                "ctc_greedy_len_ratio": None,
                "ctc_tail_nonblank_ratio": None,
                "completion_before_tail_ratio": None,
                "forced_cutoff_risk": None,
            }

        blank_id = int(self.cfg.get("ctc_blank_id"))
        pred_ids = ctc_logits.argmax(dim=-1)
        input_lengths = ctc_input_lengths.to(dtype=torch.long)
        target_labels = batch["ctc_labels"]
        target_lengths = batch["ctc_label_lens"].to(dtype=torch.long)

        total_per = 0.0
        total_prefix_per = 0.0
        total_blank = 0.0
        total_valid = 0
        total_early_blank = 0.0
        total_early_valid = 0
        total_late_blank = 0.0
        total_late_valid = 0
        total_len_ratio = 0.0
        total_tail_nonblank = 0.0
        total_completion_before_tail = 0.0
        total_forced_cutoff_risk = 0.0
        total_samples = 0

        for b in range(pred_ids.shape[0]):
            curr_len = int(input_lengths[b].item())
            if curr_len <= 0:
                continue
            curr_pred = pred_ids[b, :curr_len].tolist()
            curr_target_len = int(target_lengths[b].item())
            curr_target = target_labels[b, :curr_target_len].tolist()

            collapsed = []
            prev = None
            for token in curr_pred:
                if token != prev:
                    if token != blank_id:
                        collapsed.append(token)
                    prev = token

            if curr_target_len > 0:
                total_per += self._edit_distance(collapsed, curr_target) / float(curr_target_len)
                total_len_ratio += len(collapsed) / float(curr_target_len)
                prefix_target_len = max(1, int(curr_target_len * 0.3))
                early_frame_len = max(1, int(curr_len * 0.3))
                early_pred = curr_pred[:early_frame_len]
                collapsed_early = []
                prev_early = None
                for token in early_pred:
                    if token != prev_early:
                        if token != blank_id:
                            collapsed_early.append(token)
                        prev_early = token
                total_prefix_per += self._edit_distance(
                    collapsed_early,
                    curr_target[:prefix_target_len],
                ) / float(prefix_target_len)

            blank_count = sum(1 for token in curr_pred if token == blank_id)
            total_blank += blank_count
            total_valid += curr_len
            early_end = max(1, int(curr_len * 0.3))
            early_seq = curr_pred[:early_end]
            total_early_blank += sum(1 for token in early_seq if token == blank_id)
            total_early_valid += len(early_seq)

            late_start = min(curr_len - 1, max(0, int(curr_len * 0.7)))
            late_seq = curr_pred[late_start:]
            total_late_blank += sum(1 for token in late_seq if token == blank_id)
            total_late_valid += len(late_seq)

            tail_start = min(curr_len - 1, max(0, int(curr_len * 0.9)))
            tail_seq = curr_pred[tail_start:]
            tail_nonblank_ratio = (
                sum(1 for token in tail_seq if token != blank_id) / float(max(len(tail_seq), 1))
            )
            total_tail_nonblank += tail_nonblank_ratio

            pre_tail_seq = curr_pred[:tail_start]
            collapsed_pre_tail = []
            prev_pre_tail = None
            for token in pre_tail_seq:
                if token != prev_pre_tail:
                    if token != blank_id:
                        collapsed_pre_tail.append(token)
                    prev_pre_tail = token

            if curr_target_len > 0:
                completion_ratio = self._lcs_length(collapsed_pre_tail, curr_target) / float(curr_target_len)
                total_completion_before_tail += completion_ratio
                total_forced_cutoff_risk += float(
                    tail_nonblank_ratio > 0.2 and completion_ratio < 0.95
                )
            total_samples += 1

        if total_samples == 0:
            return {
                "ctc_greedy_per": None,
                "ctc_prefix_per": None,
                "ctc_blank_ratio": None,
                "ctc_early_blank_ratio": None,
                "ctc_late_blank_ratio": None,
                "ctc_greedy_len_ratio": None,
                "ctc_tail_nonblank_ratio": None,
                "completion_before_tail_ratio": None,
                "forced_cutoff_risk": None,
            }

        device = ctc_logits.device
        return {
            "ctc_greedy_per": torch.tensor(total_per / total_samples, device=device),
            "ctc_prefix_per": torch.tensor(total_prefix_per / total_samples, device=device),
            "ctc_blank_ratio": torch.tensor(total_blank / max(total_valid, 1), device=device),
            "ctc_early_blank_ratio": torch.tensor(total_early_blank / max(total_early_valid, 1), device=device),
            "ctc_late_blank_ratio": torch.tensor(total_late_blank / max(total_late_valid, 1), device=device),
            "ctc_greedy_len_ratio": torch.tensor(total_len_ratio / total_samples, device=device),
            "ctc_tail_nonblank_ratio": torch.tensor(total_tail_nonblank / total_samples, device=device),
            "completion_before_tail_ratio": torch.tensor(total_completion_before_tail / total_samples, device=device),
            "forced_cutoff_risk": torch.tensor(total_forced_cutoff_risk / total_samples, device=device),
        }

    def _compute_alignment_loss_and_metrics(
        self,
        align_logits: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_alignment_loss", False):
            return None, None
        if align_logits is None or "align_labels" not in batch:
            return None, None

        align_labels = batch["align_labels"].to(device=align_logits.device, dtype=torch.long)
        time_len = min(align_logits.shape[1], align_labels.shape[1])
        if time_len <= 0:
            return None, None

        align_logits = align_logits[:, :time_len, :]
        align_labels = align_labels[:, :time_len]
        loss = F.cross_entropy(
            align_logits.transpose(1, 2),
            align_labels,
            ignore_index=-100,
            reduction="mean",
        )

        with torch.no_grad():
            pred = align_logits.argmax(dim=-1)
            valid = align_labels != -100
            if bool(valid.any().item()):
                acc = (pred[valid] == align_labels[valid]).float().mean()
            else:
                acc = None
        return loss, acc

    @staticmethod
    def _compute_frame_classification_loss_and_metrics(
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        ignore_index: int = -100,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if logits is None or labels is None:
            return None, None
        time_len = min(logits.shape[1], labels.shape[1])
        if time_len <= 0:
            return None, None
        logits = logits[:, :time_len, :]
        labels = labels[:, :time_len].to(device=logits.device, dtype=torch.long)
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=ignore_index,
            reduction="mean",
        )
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            valid = labels != ignore_index
            acc = (pred[valid] == labels[valid]).float().mean() if bool(valid.any().item()) else None
        return loss, acc

    def _build_progress_targets(
        self,
        input_lengths: torch.Tensor,
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if input_lengths is None or max_time <= 0:
            return None
        num_buckets = int(self.cfg.get("progress_num_buckets", 0))
        if num_buckets <= 1:
            return None
        labels = torch.full(
            (input_lengths.shape[0], max_time),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(input_lengths.shape[0]):
            curr_len = min(int(input_lengths[b].item()), max_time)
            if curr_len <= 0:
                continue
            positions = torch.arange(curr_len, device=device, dtype=torch.long)
            curr = torch.div(positions * num_buckets, curr_len, rounding_mode="floor").clamp(max=num_buckets - 1)
            labels[b, :curr_len] = curr
        return labels

    def _build_boundary_targets(
        self,
        batch: Dict[str, torch.Tensor],
        input_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if "align_labels" not in batch or batch["align_labels"] is None or input_lengths is None or max_time <= 0:
            return None
        radius = int(self.cfg.get("boundary_radius", 1))
        align_labels = batch["align_labels"].to(device=device, dtype=torch.long)
        time_len = min(max_time, align_labels.shape[1])
        labels = torch.full(
            (align_labels.shape[0], time_len),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(align_labels.shape[0]):
            curr_len = min(int(input_lengths[b].item()), time_len)
            if curr_len <= 0:
                continue
            curr_align = align_labels[b, :curr_len]
            valid = curr_align != -100
            if not bool(valid.any().item()):
                continue
            curr_targets = torch.zeros(curr_len, device=device, dtype=torch.long)
            change = torch.zeros(curr_len, device=device, dtype=torch.bool)
            change[1:] = valid[1:] & valid[:-1] & (curr_align[1:] != curr_align[:-1])
            boundary_idx = change.nonzero(as_tuple=False).flatten()
            for idx in boundary_idx.tolist():
                start = max(0, idx - radius)
                end = min(curr_len, idx + radius + 1)
                curr_targets[start:end] = 1
            curr_targets = torch.where(valid, curr_targets, torch.full_like(curr_targets, -100))
            labels[b, :curr_len] = curr_targets
        return labels

    def _compute_progress_loss_and_metrics(
        self,
        progress_logits: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_progress_loss", False):
            return None, None
        if progress_logits is None or input_lengths is None:
            return None, None
        labels = self._build_progress_targets(
            input_lengths=input_lengths.to(device=progress_logits.device, dtype=torch.long),
            max_time=progress_logits.shape[1],
            device=progress_logits.device,
        )
        return self._compute_frame_classification_loss_and_metrics(progress_logits, labels)

    def _compute_boundary_loss_and_metrics(
        self,
        boundary_logits: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_boundary_loss", False):
            return None, None
        if boundary_logits is None or input_lengths is None:
            return None, None
        labels = self._build_boundary_targets(
            batch=batch,
            input_lengths=input_lengths.to(device=boundary_logits.device, dtype=torch.long),
            max_time=boundary_logits.shape[1],
            device=boundary_logits.device,
        )
        return self._compute_frame_classification_loss_and_metrics(boundary_logits, labels)

    def _compute_plan_progress_loss_and_metrics(
        self,
        progress_logits: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_progress_loss", False):
            return None, None
        if progress_logits is None or input_lengths is None:
            return None, None
        labels = self._build_progress_targets(
            input_lengths=input_lengths.to(device=progress_logits.device, dtype=torch.long),
            max_time=progress_logits.shape[1],
            device=progress_logits.device,
        )
        return self._compute_frame_classification_loss_and_metrics(progress_logits, labels)

    def _compute_plan_boundary_loss_and_metrics(
        self,
        boundary_logits: Optional[torch.Tensor],
        input_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_boundary_loss", False):
            return None, None
        if boundary_logits is None or input_lengths is None:
            return None, None
        labels = self._build_boundary_targets(
            batch=batch,
            input_lengths=input_lengths.to(device=boundary_logits.device, dtype=torch.long),
            max_time=boundary_logits.shape[1],
            device=boundary_logits.device,
        )
        return self._compute_frame_classification_loss_and_metrics(boundary_logits, labels)

    def _build_plan_cursor_targets(
        self,
        plan_lengths: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        max_time: int,
        max_cursor_class: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if plan_lengths is None or text_lengths is None or max_time <= 0:
            return None
        labels = torch.full(
            (plan_lengths.shape[0], max_time),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(plan_lengths.shape[0]):
            plan_len = min(int(plan_lengths[b].item()), max_time)
            text_len = min(int(text_lengths[b].item()), max_cursor_class)
            if plan_len <= 0 or text_len <= 0:
                continue
            pos = torch.arange(plan_len, device=device, dtype=torch.long)
            cursor = torch.div(pos * text_len, plan_len, rounding_mode="floor").clamp(max=text_len - 1)
            labels[b, :plan_len] = cursor
        return labels

    def _build_plan_remaining_targets(
        self,
        cursor_targets: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if cursor_targets is None or text_lengths is None or max_time <= 0:
            return None
        labels = torch.full(
            (cursor_targets.shape[0], max_time),
            fill_value=-100.0,
            device=device,
            dtype=torch.float32,
        )
        for b in range(cursor_targets.shape[0]):
            text_len = int(text_lengths[b].item())
            if text_len <= 0:
                continue
            valid = cursor_targets[b, :max_time] >= 0
            if not bool(valid.any().item()):
                continue
            curr = cursor_targets[b, :max_time].to(dtype=torch.float32)
            remain = (float(text_len - 1) - curr).clamp_min(0.0) / max(float(text_len - 1), 1.0)
            labels[b, :max_time] = torch.where(valid, remain, labels[b, :max_time])
        return labels

    def _build_plan_stop_targets(
        self,
        cursor_targets: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if cursor_targets is None or text_lengths is None or max_time <= 0:
            return None
        labels = torch.full(
            (cursor_targets.shape[0], max_time),
            fill_value=-100.0,
            device=device,
            dtype=torch.float32,
        )
        for b in range(cursor_targets.shape[0]):
            text_len = int(text_lengths[b].item())
            if text_len <= 0:
                continue
            valid = cursor_targets[b, :max_time] >= 0
            if not bool(valid.any().item()):
                continue
            curr = cursor_targets[b, :max_time]
            stop = (curr >= (text_len - 1)).to(dtype=torch.float32)
            labels[b, :max_time] = torch.where(valid, stop, labels[b, :max_time])
        return labels

    def _compute_plan_cursor_loss_and_metrics(
        self,
        cursor_logits: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_cursor_loss", False):
            return None, None, None
        if cursor_logits is None or plan_lengths is None or text_lengths is None:
            return None, None, None
        labels = self._build_plan_cursor_targets(
            plan_lengths=plan_lengths.to(device=cursor_logits.device, dtype=torch.long),
            text_lengths=text_lengths.to(device=cursor_logits.device, dtype=torch.long),
            max_time=cursor_logits.shape[1],
            max_cursor_class=cursor_logits.shape[-1],
            device=cursor_logits.device,
        )
        loss, acc = self._compute_frame_classification_loss_and_metrics(cursor_logits, labels)
        return loss, acc, labels

    def _compute_plan_remaining_loss(
        self,
        remaining_pred: Optional[torch.Tensor],
        cursor_targets: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.cfg.get("use_plan_remaining_loss", False):
            return None
        if remaining_pred is None or cursor_targets is None or text_lengths is None:
            return None
        labels = self._build_plan_remaining_targets(
            cursor_targets=cursor_targets,
            text_lengths=text_lengths.to(device=remaining_pred.device, dtype=torch.long),
            max_time=remaining_pred.shape[1],
            device=remaining_pred.device,
        )
        valid = labels >= 0
        if not bool(valid.any().item()):
            return None
        return F.smooth_l1_loss(remaining_pred[valid], labels[valid], reduction="mean")

    def _compute_plan_stop_loss_and_metrics(
        self,
        stop_logits: Optional[torch.Tensor],
        cursor_targets: Optional[torch.Tensor],
        text_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_stop_loss", False):
            return None, None
        if stop_logits is None or cursor_targets is None or text_lengths is None:
            return None, None
        labels = self._build_plan_stop_targets(
            cursor_targets=cursor_targets,
            text_lengths=text_lengths.to(device=stop_logits.device, dtype=torch.long),
            max_time=stop_logits.shape[1],
            device=stop_logits.device,
        )
        return self._compute_binary_frame_loss_and_metrics(stop_logits, labels)

    def _upsample_align_labels_to_plan(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        device: torch.device,
        max_time: int,
    ) -> Optional[torch.Tensor]:
        align_labels = batch.get("align_labels")
        align_frame_counts = batch.get("align_frame_counts")
        if align_labels is None or align_frame_counts is None or plan_lengths is None:
            return None
        labels = torch.full(
            (align_labels.shape[0], max_time),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(align_labels.shape[0]):
            frame_count = int(align_frame_counts[b].item())
            plan_len = min(int(plan_lengths[b].item()), max_time)
            if frame_count <= 0 or plan_len <= 0:
                continue
            seq = align_labels[b, :frame_count].to(device=device, dtype=torch.long)
            up = seq.repeat_interleave(2)[:plan_len]
            labels[b, : up.shape[0]] = up
        return labels

    def _build_plan_activity_targets(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        upsampled = self._upsample_align_labels_to_plan(batch, plan_lengths, device, max_time)
        if upsampled is None:
            return None
        labels = torch.full(
            upsampled.shape,
            fill_value=-100.0,
            device=device,
            dtype=torch.float32,
        )
        valid = upsampled != -100
        labels[upsampled >= 0] = 1.0
        labels[(upsampled == -100)] = 0.0
        for b in range(plan_lengths.shape[0]):
            plan_len = min(int(plan_lengths[b].item()), max_time)
            if plan_len < max_time:
                labels[b, plan_len:] = -100.0
        return labels

    def _build_plan_viseme_targets(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        upsampled = self._upsample_align_labels_to_plan(batch, plan_lengths, device, max_time)
        if upsampled is None:
            return None
        align_vocab_path = self.cfg.get("align_viseme_vocab_path")
        if align_vocab_path is None:
            return None
        from src.modeling.modules.viseme_utils import build_align_label_to_viseme_table

        if self._align_label_to_viseme_table is None:
            self._align_label_to_viseme_table = build_align_label_to_viseme_table(align_vocab_path)
        table = self._align_label_to_viseme_table.to(device=device)
        labels = torch.full(
            upsampled.shape,
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(plan_lengths.shape[0]):
            plan_len = min(int(plan_lengths[b].item()), max_time)
            if plan_len <= 0:
                continue
            seq = upsampled[b, :plan_len]
            safe = seq.clamp_min(0).clamp_max(table.shape[0] - 1)
            mapped = table[safe]
            mapped = torch.where(seq >= 0, mapped, torch.zeros_like(mapped))
            labels[b, :plan_len] = mapped
        return labels

    def _compute_plan_activity_loss_and_metrics(
        self,
        activity_logits: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_activity_loss", False):
            return None, None
        if activity_logits is None or plan_lengths is None:
            return None, None
        labels = self._build_plan_activity_targets(
            batch=batch,
            plan_lengths=plan_lengths.to(device=activity_logits.device, dtype=torch.long),
            max_time=activity_logits.shape[1],
            device=activity_logits.device,
        )
        if self.cfg.get("use_balanced_plan_activity_loss", False):
            return self._compute_balanced_binary_frame_loss_and_metrics(activity_logits, labels)
        return self._compute_binary_frame_loss_and_metrics(activity_logits, labels)

    def _compute_plan_viseme_loss_and_metrics(
        self,
        viseme_logits: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_viseme_loss", False):
            return None, None
        if viseme_logits is None or plan_lengths is None:
            return None, None
        labels = self._build_plan_viseme_targets(
            batch=batch,
            plan_lengths=plan_lengths.to(device=viseme_logits.device, dtype=torch.long),
            max_time=viseme_logits.shape[1],
            device=viseme_logits.device,
        )
        if labels is None:
            return None, None
        if not self.cfg.get("use_balanced_plan_viseme_loss", False):
            return self._compute_frame_classification_loss_and_metrics(viseme_logits, labels)
        valid = labels >= 0
        if not bool(valid.any().item()):
            return None, None
        flat_labels = labels[valid]
        flat_logits = viseme_logits[valid]
        num_classes = viseme_logits.shape[-1]
        counts = torch.bincount(flat_labels, minlength=num_classes).float()
        weights = counts.sum() / counts.clamp_min(1.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        loss = F.cross_entropy(flat_logits, flat_labels, weight=weights.to(flat_logits.device), reduction="mean")
        with torch.no_grad():
            pred = flat_logits.argmax(dim=-1)
            acc = (pred == flat_labels).float().mean()
        return loss, acc

    def _build_plan_phone_targets(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        align_labels = batch.get("align_labels")
        align_frame_counts = batch.get("align_frame_counts")
        if align_labels is None or align_frame_counts is None or plan_lengths is None:
            return None
        labels = torch.full(
            (align_labels.shape[0], max_time),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(align_labels.shape[0]):
            frame_count = int(align_frame_counts[b].item())
            plan_len = min(int(plan_lengths[b].item()), max_time)
            if frame_count <= 0 or plan_len <= 0:
                continue
            seq = align_labels[b, :frame_count].to(device=device, dtype=torch.long)
            up = seq.repeat_interleave(2)[:plan_len]
            labels[b, : up.shape[0]] = up
        return labels

    def _compute_plan_phone_loss_and_metrics(
        self,
        phone_logits: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_phone_loss", False):
            return None, None
        if phone_logits is None or plan_lengths is None:
            return None, None
        labels = self._build_plan_phone_targets(
            batch=batch,
            plan_lengths=plan_lengths.to(device=phone_logits.device, dtype=torch.long),
            max_time=phone_logits.shape[1],
            device=phone_logits.device,
        )
        return self._compute_frame_classification_loss_and_metrics(phone_logits, labels)

    def _upsample_align_occurrence_to_plan(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        occ = batch.get("align_occurrence_labels")
        frame_counts = batch.get("align_frame_counts")
        if occ is None or frame_counts is None or plan_lengths is None:
            return None
        labels = torch.full((occ.shape[0], max_time), fill_value=-100, device=device, dtype=torch.long)
        for b in range(occ.shape[0]):
            frame_count = int(frame_counts[b].item())
            plan_len = min(int(plan_lengths[b].item()), max_time)
            if frame_count <= 0 or plan_len <= 0:
                continue
            seq = occ[b, :frame_count].to(device=device, dtype=torch.long)
            up = seq.repeat_interleave(2)[:plan_len]
            labels[b, : up.shape[0]] = up
        return labels

    def _build_plan_occurrence_targets(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        occ = self._upsample_align_occurrence_to_plan(batch, plan_lengths, max_time, device)
        occ_counts = batch.get("align_occurrence_counts")
        if occ is None or occ_counts is None:
            return None
        labels = torch.full(occ.shape, fill_value=-100.0, device=device, dtype=torch.float32)
        for b in range(occ.shape[0]):
            valid = occ[b] >= 0
            if not bool(valid.any().item()):
                continue
            denom = max(int(occ_counts[b].item()) - 1, 1)
            labels[b, valid] = occ[b, valid].float() / float(denom)
        return labels

    def _compute_plan_occurrence_loss(
        self,
        occurrence_pred: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.cfg.get("use_plan_occurrence_loss", False):
            return None
        if occurrence_pred is None or plan_lengths is None:
            return None
        labels = self._build_plan_occurrence_targets(
            batch=batch,
            plan_lengths=plan_lengths.to(device=occurrence_pred.device, dtype=torch.long),
            max_time=occurrence_pred.shape[1],
            device=occurrence_pred.device,
        )
        valid = labels >= 0
        if not bool(valid.any().item()):
            return None
        return F.smooth_l1_loss(occurrence_pred[valid], labels[valid], reduction="mean")

    def _compute_plan_remaining_loss_from_occurrence(
        self,
        remaining_pred: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.cfg.get("use_plan_remaining_loss", False):
            return None
        if remaining_pred is None or plan_lengths is None:
            return None
        occ = self._build_plan_occurrence_targets(
            batch=batch,
            plan_lengths=plan_lengths.to(device=remaining_pred.device, dtype=torch.long),
            max_time=remaining_pred.shape[1],
            device=remaining_pred.device,
        )
        occ_counts = batch.get("align_occurrence_counts")
        if occ is None or occ_counts is None:
            return None
        labels = torch.full_like(remaining_pred, fill_value=-100.0)
        for b in range(occ.shape[0]):
            valid = occ[b] >= 0
            if not bool(valid.any().item()):
                continue
            labels[b, valid] = 1.0 - occ[b, valid].to(dtype=labels.dtype)
        valid = labels >= 0
        if not bool(valid.any().item()):
            return None
        return F.smooth_l1_loss(remaining_pred[valid], labels[valid], reduction="mean")

    def _compute_plan_stop_loss_from_occurrence_and_metrics(
        self,
        stop_logits: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_stop_loss", False):
            return None, None
        if stop_logits is None or plan_lengths is None:
            return None, None
        occ = self._upsample_align_occurrence_to_plan(
            batch=batch,
            plan_lengths=plan_lengths.to(device=stop_logits.device, dtype=torch.long),
            max_time=stop_logits.shape[1],
            device=stop_logits.device,
        )
        occ_counts = batch.get("align_occurrence_counts")
        if occ is None or occ_counts is None:
            return None, None
        labels = torch.full_like(stop_logits, fill_value=-100.0)
        for b in range(occ.shape[0]):
            valid = occ[b] >= 0
            if not bool(valid.any().item()):
                continue
            last_idx = max(int(occ_counts[b].item()) - 1, 0)
            labels[b, valid] = (occ[b, valid] >= last_idx).to(dtype=labels.dtype)
        if self.cfg.get("use_balanced_plan_stop_loss", False):
            return self._compute_balanced_binary_frame_loss_and_metrics(stop_logits, labels)
        return self._compute_binary_frame_loss_and_metrics(stop_logits, labels)

    def _build_segment_boundary_targets(
        self,
        batch: Dict[str, torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        occ = self._upsample_align_occurrence_to_plan(batch, plan_lengths, max_time, device)
        if occ is None:
            return None
        labels = torch.full((occ.shape[0], max_time), fill_value=-100.0, device=device, dtype=torch.float32)
        for b in range(occ.shape[0]):
            valid = occ[b] >= 0
            if not bool(valid.any().item()):
                continue
            labels[b, valid] = 0.0
            seq = occ[b]
            change = (seq[1:] != seq[:-1]) & valid[1:] & valid[:-1]
            labels[b, 1:][change] = 1.0
        return labels

    def _compute_segment_boundary_loss_and_metrics(
        self,
        boundary_logits: Optional[torch.Tensor],
        plan_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_plan_boundary_loss", False):
            return None, None
        if boundary_logits is None or plan_lengths is None:
            return None, None
        if boundary_logits.shape[-1] == 2:
            labels = self._build_segment_boundary_targets(
                batch=batch,
                plan_lengths=plan_lengths.to(device=boundary_logits.device, dtype=torch.long),
                max_time=boundary_logits.shape[1],
                device=boundary_logits.device,
            )
            if labels is None:
                return None, None
            hard = torch.full(labels.shape, fill_value=-100, device=labels.device, dtype=torch.long)
            valid = labels >= 0
            hard[valid] = labels[valid].long()
            return self._compute_frame_classification_loss_and_metrics(boundary_logits, hard)
        return None, None

    def _build_q0_progress_targets(
        self,
        q0_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if q0_lengths is None or max_time <= 0:
            return None
        num_buckets = int(self.cfg.get("q0_progress_num_buckets", 0))
        if num_buckets <= 1:
            return None
        labels = torch.full(
            (q0_lengths.shape[0], max_time),
            fill_value=-100,
            device=device,
            dtype=torch.long,
        )
        for b in range(q0_lengths.shape[0]):
            curr_len = min(int(q0_lengths[b].item()), max_time)
            if curr_len <= 0:
                continue
            positions = torch.arange(curr_len, device=device, dtype=torch.long)
            curr = torch.div(positions * num_buckets, curr_len, rounding_mode="floor").clamp(max=num_buckets - 1)
            labels[b, :curr_len] = curr
        return labels

    def _build_q0_eos_targets(
        self,
        q0_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if q0_lengths is None or max_time <= 0:
            return None
        labels = torch.full(
            (q0_lengths.shape[0], max_time),
            fill_value=-100.0,
            device=device,
            dtype=torch.float32,
        )
        for b in range(q0_lengths.shape[0]):
            curr_len = min(int(q0_lengths[b].item()), max_time)
            if curr_len <= 0:
                continue
            labels[b, :curr_len] = 0.0
            labels[b, curr_len - 1] = 1.0
        return labels

    def _build_q0_loop_targets(
        self,
        q0_target_tokens: Optional[torch.Tensor],
        q0_lengths: Optional[torch.Tensor],
        max_time: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if q0_target_tokens is None or q0_lengths is None or max_time <= 0:
            return None
        repeat_threshold = int(self.cfg.get("q0_loop_repeat_threshold", 3))
        labels = torch.full(
            (q0_target_tokens.shape[0], max_time),
            fill_value=-100.0,
            device=device,
            dtype=torch.float32,
        )
        q0_target_tokens = q0_target_tokens.to(device=device, dtype=torch.long)
        for b in range(q0_target_tokens.shape[0]):
            curr_len = min(int(q0_lengths[b].item()), max_time)
            if curr_len <= 0:
                continue
            labels[b, :curr_len] = 0.0
            run_len = 1
            for t in range(1, curr_len):
                if int(q0_target_tokens[b, t].item()) == int(q0_target_tokens[b, t - 1].item()):
                    run_len += 1
                else:
                    run_len = 1
                if run_len >= repeat_threshold:
                    labels[b, t] = 1.0
        return labels

    @staticmethod
    def _compute_binary_frame_loss_and_metrics(
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if logits is None or labels is None:
            return None, None
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        time_len = min(logits.shape[1], labels.shape[1])
        if time_len <= 0:
            return None, None
        logits = logits[:, :time_len]
        labels = labels[:, :time_len].to(device=logits.device, dtype=torch.float32)
        valid = labels >= 0
        if not bool(valid.any().item()):
            return None, None
        loss = F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction="mean")
        with torch.no_grad():
            pred = (torch.sigmoid(logits[valid]) > 0.5).float()
            acc = (pred == labels[valid]).float().mean()
        return loss, acc

    @staticmethod
    def _compute_balanced_binary_frame_loss_and_metrics(
        logits: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        max_pos_weight: float = 20.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if logits is None or labels is None:
            return None, None
        if logits.ndim == 3 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        time_len = min(logits.shape[1], labels.shape[1])
        if time_len <= 0:
            return None, None
        logits = logits[:, :time_len]
        labels = labels[:, :time_len].to(device=logits.device, dtype=torch.float32)
        valid = labels >= 0
        if not bool(valid.any().item()):
            return None, None
        target = labels[valid]
        pos = float((target == 1).sum().item())
        neg = float((target == 0).sum().item())
        if pos <= 0:
            pos_weight = 1.0
        else:
            pos_weight = min(max_pos_weight, neg / max(pos, 1.0))
        loss = F.binary_cross_entropy_with_logits(
            logits[valid],
            target,
            reduction="mean",
            pos_weight=torch.tensor(pos_weight, device=logits.device),
        )
        with torch.no_grad():
            pred = (torch.sigmoid(logits[valid]) > 0.5).float()
            acc = (pred == target).float().mean()
        return loss, acc

    def _compute_q0_progress_loss_and_metrics(
        self,
        q0_progress_logits: Optional[torch.Tensor],
        q0_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_q0_progress_loss", False):
            return None, None
        if q0_progress_logits is None or q0_lengths is None:
            return None, None
        labels = self._build_q0_progress_targets(
            q0_lengths=q0_lengths.to(device=q0_progress_logits.device, dtype=torch.long),
            max_time=q0_progress_logits.shape[1],
            device=q0_progress_logits.device,
        )
        return self._compute_frame_classification_loss_and_metrics(q0_progress_logits, labels)

    def _compute_q0_eos_loss_and_metrics(
        self,
        q0_eos_logits: Optional[torch.Tensor],
        q0_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_q0_eos_loss", False):
            return None, None
        if q0_eos_logits is None or q0_lengths is None:
            return None, None
        labels = self._build_q0_eos_targets(
            q0_lengths=q0_lengths.to(device=q0_eos_logits.device, dtype=torch.long),
            max_time=q0_eos_logits.shape[1],
            device=q0_eos_logits.device,
        )
        return self._compute_binary_frame_loss_and_metrics(q0_eos_logits, labels)

    def _compute_q0_loop_loss_and_metrics(
        self,
        q0_loop_logits: Optional[torch.Tensor],
        q0_target_tokens: Optional[torch.Tensor],
        q0_lengths: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.cfg.get("use_q0_loop_loss", False):
            return None, None
        if q0_loop_logits is None or q0_target_tokens is None or q0_lengths is None:
            return None, None
        labels = self._build_q0_loop_targets(
            q0_target_tokens=q0_target_tokens,
            q0_lengths=q0_lengths.to(device=q0_loop_logits.device, dtype=torch.long),
            max_time=q0_loop_logits.shape[1],
            device=q0_loop_logits.device,
        )
        return self._compute_binary_frame_loss_and_metrics(q0_loop_logits, labels)

    def _use_state_consistency_loss(self) -> bool:
        return bool(self.cfg.get("use_state_consistency_loss", False)) and float(self.cfg.get("state_loss_weight", 0.0)) > 0.0

    def _use_offpolicy_recovery(self) -> bool:
        return bool(self.cfg.get("use_offpolicy_recovery", False))

    def _use_plan_utilization_loss(self) -> bool:
        return bool(self.cfg.get("use_plan_utilization_loss", False)) and float(self.cfg.get("plan_utilization_loss_weight", 0.0)) > 0.0

    def _use_codebook_routing(self) -> bool:
        return bool(self.cfg.get("use_codebook_routing", False))

    def _compute_routing_losses_and_metrics(
        self,
        routing_lambdas: Optional[torch.Tensor],
        routing_valid_mask: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not self._use_codebook_routing():
            return {
                "routing_order_loss": None,
                "routing_smoothness_loss": None,
                "routing_order_margin_mean": None,
                "routing_smoothness": None,
                "routing_lambda_means": None,
            }
        if routing_lambdas is None or routing_valid_mask is None:
            return {
                "routing_order_loss": None,
                "routing_smoothness_loss": None,
                "routing_order_margin_mean": None,
                "routing_smoothness": None,
                "routing_lambda_means": None,
            }

        valid_mask = routing_valid_mask.to(device=routing_lambdas.device, dtype=torch.bool)
        if not bool(valid_mask.any().item()):
            return {
                "routing_order_loss": None,
                "routing_smoothness_loss": None,
                "routing_order_margin_mean": None,
                "routing_smoothness": None,
                "routing_lambda_means": None,
            }

        flat_lambdas = routing_lambdas[valid_mask]
        lambda_means = flat_lambdas.mean(dim=0)

        order_loss = None
        order_margin_mean = None
        if lambda_means.numel() > 1:
            diffs = lambda_means[:-1] - lambda_means[1:]
            order_margin = float(self.cfg.get("routing_order_margin", 0.0))
            order_loss = F.relu(lambda_means.new_tensor(order_margin) - diffs).mean()
            order_margin_mean = diffs.mean()

        smooth_loss = None
        smoothness_metric = None
        pair_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
        if bool(pair_mask.any().item()):
            delta = (routing_lambdas[:, 1:, :] - routing_lambdas[:, :-1, :]).abs()
            smooth_terms = delta[pair_mask]
            smoothness_metric = smooth_terms.mean()
            smooth_loss = smoothness_metric

        return {
            "routing_order_loss": order_loss,
            "routing_smoothness_loss": smooth_loss,
            "routing_order_margin_mean": order_margin_mean,
            "routing_smoothness": smoothness_metric,
            "routing_lambda_means": lambda_means.detach(),
        }

    def _compute_ce_loss_only(
        self,
        logits,
        targets,
    ) -> Optional[torch.Tensor]:
        if logits is None or targets is None:
            return None
        loss_terms = []
        total_tokens = 0
        label_smoothing = float(self.cfg.get("ce_label_smoothing", 0.0))
        for k in range(self.n_codebooks):
            target = targets[k]
            if target.numel() == 0:
                continue
            logit = logits[k]
            total_tokens += int(target.numel())
            loss_terms.append(
                F.cross_entropy(
                    logit,
                    target,
                    reduction="sum",
                    label_smoothing=label_smoothing,
                ) * float(self.codebook_weights[k])
            )
        if total_tokens == 0:
            return None
        return sum(loss_terms) / (float(total_tokens) + 1e-6)

    def _sample_plan_counterfactual_shift(self) -> int:
        max_shift = int(self.cfg.get("plan_counterfactual_shift", 0))
        if max_shift <= 0:
            return 0
        random_sign = bool(self.cfg.get("plan_counterfactual_random_sign", True))
        if not random_sign:
            return max_shift
        device = self.device if self.device.type != "meta" else torch.device("cpu")
        sign = -1 if bool(torch.rand(1, device=device).item() < 0.5) else 1
        return sign * max_shift

    def _compute_plan_utilization_losses(
        self,
        batch: Dict[str, torch.Tensor],
        base_ce_loss: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not self._use_plan_utilization_loss():
            return {
                "plan_utilization_loss": None,
                "plan_ablation_loss": None,
                "plan_shift_ce": None,
                "plan_shift_ce_gap": None,
                "plan_no_plan_ce": None,
                "plan_no_plan_ce_gap": None,
                "plan_counterfactual_shift": None,
            }
        if base_ce_loss is None or not bool(self.cfg.model.get("use_plan_conditioning", False)):
            return {
                "plan_utilization_loss": None,
                "plan_ablation_loss": None,
                "plan_shift_ce": None,
                "plan_shift_ce_gap": None,
                "plan_no_plan_ce": None,
                "plan_no_plan_ce_gap": None,
                "plan_counterfactual_shift": None,
            }

        shift_steps = self._sample_plan_counterfactual_shift()
        shifted_ce = None
        no_plan_ce = None
        with torch.no_grad():
            if shift_steps != 0:
                shifted_batch = dict(batch)
                shifted_batch["_plan_shift_steps"] = shift_steps
                shifted_batch["_disable_plan_supervision"] = True
                shifted_outputs = self.model._forward_impl(shifted_batch)
                shifted_ce = self._compute_ce_loss_only(
                    shifted_outputs.get("logits"),
                    shifted_outputs.get("targets"),
                )

            no_plan_batch = dict(batch)
            no_plan_batch["_disable_plan_conditioning"] = True
            no_plan_batch["_disable_plan_supervision"] = True
            no_plan_outputs = self.model._forward_impl(no_plan_batch)
            no_plan_ce = self._compute_ce_loss_only(
                no_plan_outputs.get("logits"),
                no_plan_outputs.get("targets"),
            )

        util_loss = None
        shift_gap = None
        if shifted_ce is not None:
            shift_gap = (shifted_ce - base_ce_loss.detach())
            util_margin = float(self.cfg.get("plan_utilization_margin", 0.0))
            util_loss = F.relu(base_ce_loss.new_tensor(util_margin) - (shifted_ce.detach() - base_ce_loss))

        ablation_loss = None
        no_plan_gap = None
        if no_plan_ce is not None:
            no_plan_gap = (no_plan_ce - base_ce_loss.detach())
            ablation_margin = float(self.cfg.get("plan_ablation_margin", self.cfg.get("plan_utilization_margin", 0.0)))
            ablation_loss = F.relu(base_ce_loss.new_tensor(ablation_margin) - (no_plan_ce.detach() - base_ce_loss))

        shift_tensor = None
        if shift_steps != 0:
            shift_tensor = base_ce_loss.new_tensor(float(shift_steps))

        return {
            "plan_utilization_loss": util_loss,
            "plan_ablation_loss": ablation_loss,
            "plan_shift_ce": shifted_ce.detach() if shifted_ce is not None else None,
            "plan_shift_ce_gap": shift_gap.detach() if shift_gap is not None else None,
            "plan_no_plan_ce": no_plan_ce.detach() if no_plan_ce is not None else None,
            "plan_no_plan_ce_gap": no_plan_gap.detach() if no_plan_gap is not None else None,
            "plan_counterfactual_shift": shift_tensor,
        }

    def _compute_state_consistency_metrics(
        self,
        teacher_repr: Optional[torch.Tensor],
        teacher_mask: Optional[torch.Tensor],
        student_repr: Optional[torch.Tensor],
        student_mask: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if teacher_repr is None or teacher_mask is None or student_repr is None or student_mask is None:
            return {
                "state_loss": None,
                "state_huber_loss": None,
                "state_cos_loss": None,
                "state_cosine": None,
                "state_l2": None,
                "state_valid_ratio": None,
            }

        time_len = min(teacher_repr.shape[1], student_repr.shape[1], teacher_mask.shape[1], student_mask.shape[1])
        if time_len <= 0:
            return {
                "state_loss": None,
                "state_huber_loss": None,
                "state_cos_loss": None,
                "state_cosine": None,
                "state_l2": None,
                "state_valid_ratio": None,
            }

        teacher_repr = teacher_repr[:, :time_len, :]
        student_repr = student_repr[:, :time_len, :]
        overlap_mask = teacher_mask[:, :time_len].bool() & student_mask[:, :time_len].bool()
        valid_ratio = overlap_mask.float().mean()
        if not bool(overlap_mask.any().item()):
            return {
                "state_loss": None,
                "state_huber_loss": None,
                "state_cos_loss": None,
                "state_cosine": None,
                "state_l2": None,
                "state_valid_ratio": valid_ratio,
            }

        teacher_valid = teacher_repr[overlap_mask].detach()
        student_valid = student_repr[overlap_mask]
        state_huber_loss = F.smooth_l1_loss(student_valid, teacher_valid, reduction="mean")
        cosine_sim = F.cosine_similarity(student_valid, teacher_valid, dim=-1)
        state_cos_loss = 1.0 - cosine_sim.mean()
        state_loss = (
            float(self.cfg.get("state_huber_component_weight", 1.0)) * state_huber_loss
            + float(self.cfg.get("state_cos_component_weight", 1.0)) * state_cos_loss
        )
        state_l2 = torch.norm(student_valid - teacher_valid, dim=-1).mean()
        return {
            "state_loss": state_loss,
            "state_huber_loss": state_huber_loss,
            "state_cos_loss": state_cos_loss,
            "state_cosine": cosine_sim.mean(),
            "state_l2": state_l2,
            "state_valid_ratio": valid_ratio,
        }

    def _compute_late_nonblank_penalty(
        self,
        ctc_logits: Optional[torch.Tensor],
        ctc_input_lengths: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.cfg.get("use_alignment_loss", False):
            return None
        if ctc_logits is None or ctc_input_lengths is None:
            return None

        blank_id = int(self.cfg.get("ctc_blank_id"))
        probs = F.softmax(ctc_logits, dim=-1)
        total_penalty = 0.0
        total_samples = 0
        for b in range(probs.shape[0]):
            curr_len = int(ctc_input_lengths[b].item())
            if curr_len <= 0:
                continue
            tail_start = min(curr_len - 1, max(0, int(curr_len * 0.9)))
            tail_probs = probs[b, tail_start:curr_len, blank_id]
            if tail_probs.numel() == 0:
                continue
            total_penalty += (1.0 - tail_probs).mean()
            total_samples += 1
        if total_samples == 0:
            return None
        return total_penalty / total_samples

    def _compute_target_token_metrics(
        self,
        pred_target_tokens: Optional[torch.Tensor],
        pred_target_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if pred_target_tokens is None or pred_target_lengths is None:
            return {
                "target_token_acc": None,
                "target_mean_ned": None,
                "target_unfinished_ratio": None,
                "target_coverage_ratio": None,
            }

        y = batch["y"]
        y_lens = batch["y_lens"].to(dtype=torch.long, device=pred_target_tokens.device)
        split_lengths = batch["split_lens"].to(dtype=torch.long, device=pred_target_tokens.device)

        total_acc = 0.0
        total_ned = 0.0
        total_unfinished = 0.0
        total_coverage = 0.0
        total_samples = 0

        for b in range(y.shape[0]):
            target_start = int(split_lengths[b].item())
            target_end = int(y_lens[b].item())
            gt_len = max(target_end - target_start, 0)
            if gt_len <= 0:
                continue

            pred_len = min(int(pred_target_lengths[b].item()), int(pred_target_tokens.shape[-1]))
            gt = y[b, :, target_start:target_end].to(dtype=torch.long)
            pred = pred_target_tokens[b, :, :pred_len].to(dtype=torch.long)
            overlap = min(gt_len, pred_len)

            if overlap > 0:
                acc = (pred[:, :overlap] == gt[:, :overlap]).float().mean().item()
            else:
                acc = 0.0

            neds = []
            for k in range(gt.shape[0]):
                pred_seq = pred[k].tolist()
                gt_seq = gt[k].tolist()
                dist = self._edit_distance(pred_seq, gt_seq)
                neds.append(dist / float(max(len(gt_seq), 1)))
            mean_ned = float(sum(neds) / len(neds)) if neds else 0.0

            unfinished_ratio = max(gt_len - pred_len, 0) / float(max(gt_len, 1))
            coverage_ratio = min(pred_len, gt_len) / float(max(gt_len, 1))

            total_acc += acc
            total_ned += mean_ned
            total_unfinished += unfinished_ratio
            total_coverage += coverage_ratio
            total_samples += 1

        if total_samples == 0:
            return {
                "target_token_acc": None,
                "target_mean_ned": None,
                "target_unfinished_ratio": None,
                "target_coverage_ratio": None,
            }

        device = pred_target_tokens.device
        return {
            "target_token_acc": torch.tensor(total_acc / total_samples, device=device),
            "target_mean_ned": torch.tensor(total_ned / total_samples, device=device),
            "target_unfinished_ratio": torch.tensor(total_unfinished / total_samples, device=device),
            "target_coverage_ratio": torch.tensor(total_coverage / total_samples, device=device),
        }

    def _compute_online_exposure_probe_metrics(
        self,
        batch: Dict[str, torch.Tensor],
        clean_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if not (self.cfg.get("use_history_corruption", False) or self.cfg.get("use_scheduled_sampling", False)):
            return {
                "tf_target_token_acc": None,
                "tf_target_mean_ned": None,
                "tf_target_unfinished_ratio": None,
                "tf_target_coverage_ratio": None,
                "exposure_probe_target_token_acc": None,
                "exposure_probe_target_mean_ned": None,
                "exposure_probe_target_unfinished_ratio": None,
                "exposure_probe_target_coverage_ratio": None,
                "exposure_probe_acc_gap": None,
                "exposure_probe_ned_gap": None,
                "exposure_probe_unfinished_gap": None,
                "exposure_probe_acc_retention": None,
                "exposure_probe_coverage_gap": None,
                "exposure_probe_coverage_retention": None,
                "exposure_probe_hc_rate": None,
                "exposure_probe_ss_rate": None,
            }

        teacher_metrics = self._compute_target_token_metrics(
            clean_outputs.get("pred_target_tokens"),
            clean_outputs.get("pred_target_lengths"),
            batch,
        )

        hc_rate = 0.0
        ss_rate = 0.0
        if getattr(self.model, "use_history_corruption", False):
            hc_rate = float(self.model._get_curriculum_rate(
                float(getattr(self.model, "history_corruption_max_rate", 0.0)),
                int(getattr(self.model, "history_corruption_warmup_steps", 0)),
            ))
        if getattr(self.model, "use_scheduled_sampling", False):
            ss_rate = float(self.model._get_linear_schedule_rate(
                float(getattr(self.model, "scheduled_sampling_start_rate", 0.0)),
                float(getattr(self.model, "scheduled_sampling_max_rate", 0.0)),
                int(getattr(self.model, "scheduled_sampling_warmup_steps", 0)),
            ))

        if hc_rate <= 0 and ss_rate <= 0:
            return {
                "tf_target_token_acc": teacher_metrics["target_token_acc"],
                "tf_target_mean_ned": teacher_metrics["target_mean_ned"],
                "tf_target_unfinished_ratio": teacher_metrics["target_unfinished_ratio"],
                "tf_target_coverage_ratio": teacher_metrics["target_coverage_ratio"],
                "exposure_probe_target_token_acc": teacher_metrics["target_token_acc"],
                "exposure_probe_target_mean_ned": teacher_metrics["target_mean_ned"],
                "exposure_probe_target_unfinished_ratio": teacher_metrics["target_unfinished_ratio"],
                "exposure_probe_target_coverage_ratio": teacher_metrics["target_coverage_ratio"],
                "exposure_probe_acc_gap": torch.tensor(0.0, device=batch["y"].device),
                "exposure_probe_ned_gap": torch.tensor(0.0, device=batch["y"].device),
                "exposure_probe_unfinished_gap": torch.tensor(0.0, device=batch["y"].device),
                "exposure_probe_acc_retention": torch.tensor(1.0, device=batch["y"].device),
                "exposure_probe_coverage_gap": torch.tensor(0.0, device=batch["y"].device),
                "exposure_probe_coverage_retention": torch.tensor(1.0, device=batch["y"].device),
                "exposure_probe_hc_rate": torch.tensor(hc_rate, device=batch["y"].device),
                "exposure_probe_ss_rate": torch.tensor(ss_rate, device=batch["y"].device),
            }

        y_mod, _, _, _ = self.model._apply_target_history_perturbation(
            y=batch["y"],
            y_lens=batch["y_lens"].to(device=batch["y"].device, dtype=torch.long),
            split_lengths=batch["split_lens"].to(device=batch["y"].device, dtype=torch.long),
            hc_rate=hc_rate if getattr(self.model, "use_history_corruption", False) else 0.0,
            ss_rate=ss_rate if getattr(self.model, "use_scheduled_sampling", False) else 0.0,
            pred_target_tokens=clean_outputs.get("pred_target_tokens"),
            pred_target_lengths=clean_outputs.get("pred_target_lengths"),
        )
        probe_batch = dict(batch)
        probe_batch["y"] = y_mod
        probe_outputs = self.model._forward_impl(probe_batch, return_pred_target_tokens=True)
        probe_metrics = self._compute_target_token_metrics(
            probe_outputs.get("pred_target_tokens"),
            probe_outputs.get("pred_target_lengths"),
            batch,
        )

        acc_gap = None
        ned_gap = None
        unfinished_gap = None
        acc_retention = None
        coverage_gap = None
        coverage_retention = None
        if teacher_metrics["target_token_acc"] is not None and probe_metrics["target_token_acc"] is not None:
            acc_gap = teacher_metrics["target_token_acc"] - probe_metrics["target_token_acc"]
            acc_retention = probe_metrics["target_token_acc"] / teacher_metrics["target_token_acc"].clamp_min(1e-6)
        if teacher_metrics["target_mean_ned"] is not None and probe_metrics["target_mean_ned"] is not None:
            ned_gap = probe_metrics["target_mean_ned"] - teacher_metrics["target_mean_ned"]
        if teacher_metrics["target_unfinished_ratio"] is not None and probe_metrics["target_unfinished_ratio"] is not None:
            unfinished_gap = probe_metrics["target_unfinished_ratio"] - teacher_metrics["target_unfinished_ratio"]
        if teacher_metrics["target_coverage_ratio"] is not None and probe_metrics["target_coverage_ratio"] is not None:
            coverage_gap = teacher_metrics["target_coverage_ratio"] - probe_metrics["target_coverage_ratio"]
            coverage_retention = probe_metrics["target_coverage_ratio"] / teacher_metrics["target_coverage_ratio"].clamp_min(1e-6)

        return {
            "tf_target_token_acc": teacher_metrics["target_token_acc"],
            "tf_target_mean_ned": teacher_metrics["target_mean_ned"],
            "tf_target_unfinished_ratio": teacher_metrics["target_unfinished_ratio"],
            "tf_target_coverage_ratio": teacher_metrics["target_coverage_ratio"],
            "exposure_probe_target_token_acc": probe_metrics["target_token_acc"],
            "exposure_probe_target_mean_ned": probe_metrics["target_mean_ned"],
            "exposure_probe_target_unfinished_ratio": probe_metrics["target_unfinished_ratio"],
            "exposure_probe_target_coverage_ratio": probe_metrics["target_coverage_ratio"],
            "exposure_probe_acc_gap": acc_gap,
            "exposure_probe_ned_gap": ned_gap,
            "exposure_probe_unfinished_gap": unfinished_gap,
            "exposure_probe_acc_retention": acc_retention,
            "exposure_probe_coverage_gap": coverage_gap,
            "exposure_probe_coverage_retention": coverage_retention,
            "exposure_probe_hc_rate": torch.tensor(hc_rate, device=batch["y"].device),
            "exposure_probe_ss_rate": torch.tensor(ss_rate, device=batch["y"].device),
        }

    def _compute_video_ablation_gap(
        self,
        batch: Dict[str, torch.Tensor],
        base_ce_loss: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not self.cfg.model.get("use_in_decoder_cross_attention", False):
            return None
        if "v" not in batch:
            return None

        ablated_batch = {
            key: (value.clone() if torch.is_tensor(value) else value)
            for key, value in batch.items()
        }
        ablated_batch["v"] = torch.zeros_like(batch["v"])
        outputs_no_video = self.model(ablated_batch)
        ctc_loss_no_video = self._compute_ctc_loss(
            outputs_no_video.get("ctc_logits"),
            outputs_no_video.get("ctc_input_lengths"),
            ablated_batch,
        )
        ctc_metrics_no_video = self._compute_ctc_metrics(
            outputs_no_video.get("ctc_logits"),
            outputs_no_video.get("ctc_input_lengths"),
            ablated_batch,
        )
        metrics_no_video = self.compute_loss_and_metrics(
            **outputs_no_video,
            ctc_loss=ctc_loss_no_video,
            **ctc_metrics_no_video,
        )
        return (metrics_no_video["ce_loss"] - base_ce_loss).detach()

    def _compute_ctc_loss(
        self,
        ctc_logits: Optional[torch.Tensor],
        ctc_input_lengths: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.ctc_loss_fn is None:
            return None
        if ctc_logits is None or ctc_input_lengths is None:
            return None
        if "ctc_labels" not in batch or "ctc_label_lens" not in batch:
            return None

        ctc_targets = batch["ctc_labels"]
        ctc_target_lengths = batch["ctc_label_lens"]
        log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
        return self.ctc_loss_fn(
            log_probs,
            ctc_targets,
            ctc_input_lengths.to(dtype=torch.long),
            ctc_target_lengths.to(dtype=torch.long),
        )

    def training_step(self, batch, batch_idx):
        self.model.current_train_step = int(self.global_step)
        if self._use_state_consistency_loss() or self._use_offpolicy_recovery():
            teacher_batch = dict(batch)
            teacher_batch["_disable_exposure_mitigation"] = True
            need_teacher_predictions = bool(getattr(self.model, "use_scheduled_sampling", False))
            with torch.no_grad():
                teacher_outputs = self.model._forward_impl(
                    teacher_batch,
                    return_pred_target_tokens=need_teacher_predictions,
                    return_interface_state=self._use_state_consistency_loss(),
                )
            work_batch, exposure_metrics = self.model._prepare_training_batch_with_exposure_mitigation(
                batch,
                teacher_outputs=teacher_outputs,
            )
            outputs = self.model._forward_impl(
                work_batch,
                return_pred_target_tokens=True,
                return_interface_state=self._use_state_consistency_loss(),
            )
            outputs.update(exposure_metrics)
            if self._use_state_consistency_loss():
                outputs.update(
                    self._compute_state_consistency_metrics(
                        teacher_outputs.get("pooled_audio_repr"),
                        teacher_outputs.get("pooled_audio_mask"),
                        outputs.get("pooled_audio_repr"),
                        outputs.get("pooled_audio_mask"),
                    )
                )
        else:
            work_batch = batch
            outputs = self.model(batch)
        ctc_loss = self._compute_ctc_loss(
            outputs.get("ctc_logits"),
            outputs.get("ctc_input_lengths"),
            work_batch,
        )
        align_loss, align_frame_acc = self._compute_alignment_loss_and_metrics(
            outputs.get("align_logits"),
            work_batch,
        )
        progress_loss, progress_frame_acc = self._compute_progress_loss_and_metrics(
            outputs.get("progress_logits"),
            outputs.get("interface_input_lengths"),
        )
        boundary_loss, boundary_frame_acc = self._compute_boundary_loss_and_metrics(
            outputs.get("boundary_logits"),
            outputs.get("interface_input_lengths"),
            work_batch,
        )
        plan_progress_loss, plan_progress_frame_acc = self._compute_plan_progress_loss_and_metrics(
            outputs.get("plan_progress_logits"),
            outputs.get("plan_input_lengths"),
        )
        if outputs.get("plan_occurrence_pred") is not None:
            plan_boundary_loss, plan_boundary_frame_acc = self._compute_segment_boundary_loss_and_metrics(
                outputs.get("plan_boundary_logits"),
                outputs.get("plan_input_lengths"),
                work_batch,
            )
        else:
            plan_boundary_loss, plan_boundary_frame_acc = self._compute_plan_boundary_loss_and_metrics(
                outputs.get("plan_boundary_logits"),
                outputs.get("plan_input_lengths"),
                work_batch,
            )
        plan_cursor_loss, plan_cursor_frame_acc, plan_cursor_targets = self._compute_plan_cursor_loss_and_metrics(
            outputs.get("plan_cursor_logits"),
            outputs.get("plan_input_lengths"),
            outputs.get("plan_text_lengths"),
        )
        plan_monotonic_loss = outputs.get("plan_monotonic_loss")
        if outputs.get("plan_occurrence_pred") is not None:
            plan_occurrence_loss = self._compute_plan_occurrence_loss(
                outputs.get("plan_occurrence_pred"),
                outputs.get("plan_input_lengths"),
                work_batch,
            )
            plan_remaining_loss = self._compute_plan_remaining_loss_from_occurrence(
                outputs.get("plan_remaining_pred"),
                outputs.get("plan_input_lengths"),
                work_batch,
            )
            plan_stop_loss, plan_stop_frame_acc = self._compute_plan_stop_loss_from_occurrence_and_metrics(
                outputs.get("plan_stop_logits"),
                outputs.get("plan_input_lengths"),
                work_batch,
            )
        else:
            plan_occurrence_loss = None
            plan_remaining_loss = self._compute_plan_remaining_loss(
                outputs.get("plan_remaining_pred"),
                plan_cursor_targets,
                outputs.get("plan_text_lengths"),
            )
            plan_stop_loss, plan_stop_frame_acc = self._compute_plan_stop_loss_and_metrics(
                outputs.get("plan_stop_logits"),
                plan_cursor_targets,
                outputs.get("plan_text_lengths"),
            )
        plan_activity_loss, plan_activity_frame_acc = self._compute_plan_activity_loss_and_metrics(
            outputs.get("plan_activity_logits"),
            outputs.get("plan_input_lengths"),
            work_batch,
        )
        plan_viseme_loss, plan_viseme_frame_acc = self._compute_plan_viseme_loss_and_metrics(
            outputs.get("plan_viseme_logits"),
            outputs.get("plan_input_lengths"),
            work_batch,
        )
        plan_phone_loss, plan_phone_frame_acc = self._compute_plan_phone_loss_and_metrics(
            outputs.get("plan_phone_logits"),
            outputs.get("plan_input_lengths"),
            work_batch,
        )
        q0_progress_loss, q0_progress_frame_acc = self._compute_q0_progress_loss_and_metrics(
            outputs.get("q0_progress_logits"),
            outputs.get("q0_output_lengths"),
        )
        q0_loop_loss, q0_loop_frame_acc = self._compute_q0_loop_loss_and_metrics(
            outputs.get("q0_loop_logits"),
            outputs.get("q0_target_tokens"),
            outputs.get("q0_output_lengths"),
        )
        q0_eos_loss, q0_eos_frame_acc = self._compute_q0_eos_loss_and_metrics(
            outputs.get("q0_eos_logits"),
            outputs.get("q0_output_lengths"),
        )
        late_nonblank_penalty = self._compute_late_nonblank_penalty(
            outputs.get("ctc_logits"),
            outputs.get("ctc_input_lengths"),
        )
        ctc_metrics = self._compute_ctc_metrics(
            outputs.get("ctc_logits"),
            outputs.get("ctc_input_lengths"),
            work_batch,
        )
        metrics = self.compute_loss_and_metrics(
            **outputs,
            ctc_loss=ctc_loss,
            align_loss=align_loss,
            align_frame_acc=align_frame_acc,
            progress_loss=progress_loss,
            progress_frame_acc=progress_frame_acc,
            boundary_loss=boundary_loss,
            boundary_frame_acc=boundary_frame_acc,
            plan_progress_loss=plan_progress_loss,
            plan_progress_frame_acc=plan_progress_frame_acc,
            plan_boundary_loss=plan_boundary_loss,
            plan_boundary_frame_acc=plan_boundary_frame_acc,
            plan_cursor_loss=plan_cursor_loss,
            plan_cursor_frame_acc=plan_cursor_frame_acc,
            plan_remaining_loss=plan_remaining_loss,
            plan_stop_loss=plan_stop_loss,
            plan_stop_frame_acc=plan_stop_frame_acc,
            plan_activity_loss=plan_activity_loss,
            plan_activity_frame_acc=plan_activity_frame_acc,
            plan_viseme_loss=plan_viseme_loss,
            plan_viseme_frame_acc=plan_viseme_frame_acc,
            plan_phone_loss=plan_phone_loss,
            plan_phone_frame_acc=plan_phone_frame_acc,
            plan_occurrence_loss=plan_occurrence_loss,
            q0_progress_loss=q0_progress_loss,
            q0_progress_frame_acc=q0_progress_frame_acc,
            q0_loop_loss=q0_loop_loss,
            q0_loop_frame_acc=q0_loop_frame_acc,
            q0_eos_loss=q0_eos_loss,
            q0_eos_frame_acc=q0_eos_frame_acc,
            late_nonblank_penalty=late_nonblank_penalty,
            **ctc_metrics,
        )
        util_metrics = self._compute_plan_utilization_losses(
            work_batch,
            metrics["ce_loss"],
        )
        metrics.update(util_metrics)
        routing_metrics = self._compute_routing_losses_and_metrics(
            outputs.get("routing_lambdas"),
            outputs.get("routing_valid_mask"),
        )
        metrics.update(routing_metrics)
        if metrics.get("plan_utilization_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["plan_utilization_loss"] * float(
                self.cfg.get("plan_utilization_loss_weight", 0.0)
            )
        if metrics.get("plan_ablation_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["plan_ablation_loss"] * float(
                self.cfg.get("plan_ablation_loss_weight", self.cfg.get("plan_utilization_loss_weight", 0.0))
            )
        if metrics.get("routing_order_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["routing_order_loss"] * float(
                self.cfg.get("routing_order_loss_weight", 0.0)
            )
        if metrics.get("routing_smoothness_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["routing_smoothness_loss"] * float(
                self.cfg.get("routing_smoothness_loss_weight", 0.0)
            )

        total_loss = metrics["loss"]
        effective_ntoken = metrics["effective_ntoken"]
        loss_for_backward = total_loss
        
        if torch.isnan(loss_for_backward) or torch.isinf(loss_for_backward):
            logger.warning(f"Loss is NaN/Inf at step {self.global_step}, skipping batch.")
            return None


        avg_ce_loss = metrics["ce_loss"]
        avg_loss = total_loss
        avg_acc = metrics["topk_acc"] / (effective_ntoken + 1e-6)
        
        self.log("train/total_loss", avg_loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/ce_loss", avg_ce_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/target_loss_step", avg_ce_loss, prog_bar=False, on_step=True, on_epoch=False, sync_dist=True)
        self.log("train/acc", avg_acc, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        # Extra precision logs (4dp) for easier tracking (disabled)
        
        # self.log("train/loss_4dp", avg_loss, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        # self.log("train/acc_4dp", avg_acc, prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        # if self.trainer is not None and self.trainer.log_every_n_steps:
        #     if (batch_idx + 1) % self.trainer.log_every_n_steps == 0:
        #         self.print(
        #             f"train/loss_step_4dp={avg_loss.item():.4f}, "
        #             f"train/acc_step_4dp={avg_acc.item():.4f}"
        #         )
        self.log("train/ntokens", effective_ntoken.float(), prog_bar=False, sync_dist=True)
        if metrics.get("vq_loss") is not None:
            self.log("train/vq_loss", metrics["vq_loss"], prog_bar=False, sync_dist=True)
        if metrics.get("vq_usage") is not None:
            self.log("train/vq_usage", metrics["vq_usage"], prog_bar=False, sync_dist=True)
        if metrics.get("ref_ce_loss") is not None:
            self.log("train/ref_ce_loss", metrics["ref_ce_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ref_topk_acc") is not None:
            self.log("train/ref_topk_acc", metrics["ref_topk_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("sync_loss") is not None:
            self.log("train/sync_loss", metrics["sync_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("sync_pos_alignment") is not None:
            self.log("train/sync_pos_alignment", metrics["sync_pos_alignment"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("sync_neg_alignment") is not None:
            self.log("train/sync_neg_alignment", metrics["sync_neg_alignment"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        for key in (
            "g_embedding_cos_loss",
            "g_embedding_ce_loss",
            "g_embedding_cos_mean",
            "g_embedding_top1",
            "g_embedding_top10",
        ):
            if metrics.get(key) is not None:
                self.log(f"train/{key}", metrics[key], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        for q in range(self.n_codebooks):
            for key in (
                f"g_embedding_cos_q{q}",
                f"g_embedding_ce_q{q}",
                f"g_embedding_top1_q{q}",
                f"g_embedding_top10_q{q}",
            ):
                if metrics.get(key) is not None:
                    self.log(f"train/{key}", metrics[key], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_loss") is not None:
            self.log("train/ctc_loss", metrics["ctc_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("align_loss") is not None:
            self.log("train/align_loss", metrics["align_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("align_frame_acc") is not None:
            self.log("train/align_frame_acc", metrics["align_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("progress_loss") is not None:
            self.log("train/progress_loss", metrics["progress_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("progress_frame_acc") is not None:
            self.log("train/progress_frame_acc", metrics["progress_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("boundary_loss") is not None:
            self.log("train/boundary_loss", metrics["boundary_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("boundary_frame_acc") is not None:
            self.log("train/boundary_frame_acc", metrics["boundary_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_progress_loss") is not None:
            self.log("train/plan_progress_loss", metrics["plan_progress_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_progress_frame_acc") is not None:
            self.log("train/plan_progress_frame_acc", metrics["plan_progress_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_boundary_loss") is not None:
            self.log("train/plan_boundary_loss", metrics["plan_boundary_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_boundary_frame_acc") is not None:
            self.log("train/plan_boundary_frame_acc", metrics["plan_boundary_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_acoustic_loss") is not None:
            self.log("train/plan_acoustic_loss", metrics["plan_acoustic_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_acoustic_pos_alignment") is not None:
            self.log("train/plan_acoustic_pos_alignment", metrics["plan_acoustic_pos_alignment"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_acoustic_neg_alignment") is not None:
            self.log("train/plan_acoustic_neg_alignment", metrics["plan_acoustic_neg_alignment"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_cursor_loss") is not None:
            self.log("train/plan_cursor_loss", metrics["plan_cursor_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_cursor_frame_acc") is not None:
            self.log("train/plan_cursor_frame_acc", metrics["plan_cursor_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_monotonic_loss") is not None:
            self.log("train/plan_monotonic_loss", metrics["plan_monotonic_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_remaining_loss") is not None:
            self.log("train/plan_remaining_loss", metrics["plan_remaining_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_stop_loss") is not None:
            self.log("train/plan_stop_loss", metrics["plan_stop_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_stop_frame_acc") is not None:
            self.log("train/plan_stop_frame_acc", metrics["plan_stop_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_activity_loss") is not None:
            self.log("train/plan_activity_loss", metrics["plan_activity_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_activity_frame_acc") is not None:
            self.log("train/plan_activity_frame_acc", metrics["plan_activity_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_viseme_loss") is not None:
            self.log("train/plan_viseme_loss", metrics["plan_viseme_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_viseme_frame_acc") is not None:
            self.log("train/plan_viseme_frame_acc", metrics["plan_viseme_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_phone_loss") is not None:
            self.log("train/plan_phone_loss", metrics["plan_phone_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_phone_frame_acc") is not None:
            self.log("train/plan_phone_frame_acc", metrics["plan_phone_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_occurrence_loss") is not None:
            self.log("train/plan_occurrence_loss", metrics["plan_occurrence_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if outputs.get("plan_viseme_compatibility_mean") is not None:
            self.log("train/plan_viseme_compatibility_mean", outputs["plan_viseme_compatibility_mean"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("q0_progress_loss") is not None:
            self.log("train/q0_progress_loss", metrics["q0_progress_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("q0_progress_frame_acc") is not None:
            self.log("train/q0_progress_frame_acc", metrics["q0_progress_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("q0_loop_loss") is not None:
            self.log("train/q0_loop_loss", metrics["q0_loop_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("q0_loop_frame_acc") is not None:
            self.log("train/q0_loop_frame_acc", metrics["q0_loop_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("q0_eos_loss") is not None:
            self.log("train/q0_eos_loss", metrics["q0_eos_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("q0_eos_frame_acc") is not None:
            self.log("train/q0_eos_frame_acc", metrics["q0_eos_frame_acc"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if outputs.get("q0_planner_adapter_delta_norm") is not None:
            self.log("train/q0_planner_adapter_delta_norm", outputs["q0_planner_adapter_delta_norm"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if outputs.get("q0_planner_adapter_gate") is not None:
            self.log("train/q0_planner_adapter_gate", outputs["q0_planner_adapter_gate"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_utilization_loss") is not None:
            self.log("train/plan_utilization_loss", metrics["plan_utilization_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_ablation_loss") is not None:
            self.log("train/plan_ablation_loss", metrics["plan_ablation_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("routing_order_loss") is not None:
            self.log("train/routing_order_loss", metrics["routing_order_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("routing_smoothness_loss") is not None:
            self.log("train/routing_smoothness_loss", metrics["routing_smoothness_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("routing_order_margin_mean") is not None:
            self.log("train/routing_order_margin_mean", metrics["routing_order_margin_mean"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("routing_smoothness") is not None:
            self.log("train/routing_smoothness", metrics["routing_smoothness"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_shift_ce") is not None:
            self.log("train/plan_shift_ce", metrics["plan_shift_ce"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_shift_ce_gap") is not None:
            self.log("train/plan_shift_ce_gap", metrics["plan_shift_ce_gap"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_no_plan_ce") is not None:
            self.log("train/plan_no_plan_ce", metrics["plan_no_plan_ce"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_no_plan_ce_gap") is not None:
            self.log("train/plan_no_plan_ce_gap", metrics["plan_no_plan_ce_gap"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("plan_counterfactual_shift") is not None:
            self.log("train/plan_counterfactual_shift", metrics["plan_counterfactual_shift"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("state_loss") is not None:
            self.log("train/state_loss", metrics["state_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("state_huber_loss") is not None:
            self.log("train/state_huber_loss", metrics["state_huber_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("state_cos_loss") is not None:
            self.log("train/state_cos_loss", metrics["state_cos_loss"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("state_cosine") is not None:
            self.log("train/state_cosine", metrics["state_cosine"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("state_l2") is not None:
            self.log("train/state_l2", metrics["state_l2"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("state_valid_ratio") is not None:
            self.log("train/state_valid_ratio", metrics["state_valid_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("late_nonblank_penalty") is not None:
            self.log("train/late_nonblank_penalty", metrics["late_nonblank_penalty"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_greedy_per") is not None:
            self.log("train/ctc_greedy_per", metrics["ctc_greedy_per"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_prefix_per") is not None:
            self.log("train/ctc_prefix_per", metrics["ctc_prefix_per"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_blank_ratio") is not None:
            self.log("train/ctc_blank_ratio", metrics["ctc_blank_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_early_blank_ratio") is not None:
            self.log("train/ctc_early_blank_ratio", metrics["ctc_early_blank_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_late_blank_ratio") is not None:
            self.log("train/ctc_late_blank_ratio", metrics["ctc_late_blank_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_greedy_len_ratio") is not None:
            self.log("train/ctc_greedy_len_ratio", metrics["ctc_greedy_len_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("ctc_tail_nonblank_ratio") is not None:
            self.log("train/ctc_tail_nonblank_ratio", metrics["ctc_tail_nonblank_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("completion_before_tail_ratio") is not None:
            self.log("train/completion_before_tail_ratio", metrics["completion_before_tail_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("forced_cutoff_risk") is not None:
            self.log("train/forced_cutoff_risk", metrics["forced_cutoff_risk"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if metrics.get("routing_lambda_means") is not None:
            for q, val in enumerate(metrics["routing_lambda_means"]):
                self.log(
                    f"train/routing_lambda_mean_q{q}",
                    val,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
        if outputs.get("hier_context_norm_means") is not None:
            for q, val in enumerate(outputs["hier_context_norm_means"]):
                self.log(
                    f"train/hier_context_norm_q{q}",
                    val,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
        if outputs.get("hier_context_scales") is not None:
            for q, val in enumerate(outputs["hier_context_scales"]):
                self.log(
                    f"train/hier_context_scale_q{q}",
                    val,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                )
        for layer_idx, gate_value in outputs.get("in_decoder_gate_values", {}).items():
            self.log(
                f"train/in_decoder_gate_value_layer{layer_idx}",
                gate_value,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )
        for layer_idx, delta_norm in outputs.get("in_decoder_fusion_delta_norms", {}).items():
            self.log(
                f"train/in_decoder_fusion_delta_norm_layer{layer_idx}",
                delta_norm,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

        if outputs.get("history_corruption_rate_used") is not None:
            self.log("train/history_corruption_rate_used", outputs["history_corruption_rate_used"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if outputs.get("scheduled_sampling_rate_used") is not None:
            self.log("train/scheduled_sampling_rate_used", outputs["scheduled_sampling_rate_used"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if outputs.get("history_corruption_applied_ratio") is not None:
            self.log("train/history_corruption_applied_ratio", outputs["history_corruption_applied_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)
        if outputs.get("scheduled_sampling_applied_ratio") is not None:
            self.log("train/scheduled_sampling_applied_ratio", outputs["scheduled_sampling_applied_ratio"], prog_bar=False, on_step=True, on_epoch=True, sync_dist=True)

        cb_logs = {}
        for k, acc_weighted_sum in enumerate(metrics["topk_acc_by_codebook"]):
            cb_logs[f"train/acc_cb_{k}"] = acc_weighted_sum / (effective_ntoken + 1e-6)
        self.log_dict(cb_logs, sync_dist=True)

        return loss_for_backward

    def on_train_epoch_start(self):
        scheduler = self.lr_schedulers()

        if isinstance(scheduler, Eden):
            pseudo_epoch = self.global_step // self.cfg.scheduler.get("pseudo_epoch_size", 3000) + 1
            scheduler.step_epoch(pseudo_epoch)
        
        self.train_acc_num = torch.tensor(0.0, device=self.device)
        self.train_acc_den = torch.tensor(0.0, device=self.device)

    def lr_scheduler_step(self, scheduler, metric):        
        # Only step the scheduler after a real optimizer step has occurred.
        if not hasattr(self, "_last_lr_step"):
            self._last_lr_step = -1
        if self.global_step == 0:
            return
        if self.global_step == self._last_lr_step:
            return
        self._last_lr_step = self.global_step

        if self.is_scaled_adam:
            if hasattr(scheduler, "step_batch"):
                scheduler.step_batch(self.global_step)
            else:
                # Fallback if Eden interface changes
                scheduler.step()
        else:
            scheduler.step()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch)
        
        if outputs is None:
            return None
            
        ctc_loss = self._compute_ctc_loss(
            outputs.get("ctc_logits"),
            outputs.get("ctc_input_lengths"),
            batch,
        )
        align_loss, align_frame_acc = self._compute_alignment_loss_and_metrics(
            outputs.get("align_logits"),
            batch,
        )
        progress_loss, progress_frame_acc = self._compute_progress_loss_and_metrics(
            outputs.get("progress_logits"),
            outputs.get("interface_input_lengths"),
        )
        boundary_loss, boundary_frame_acc = self._compute_boundary_loss_and_metrics(
            outputs.get("boundary_logits"),
            outputs.get("interface_input_lengths"),
            batch,
        )
        plan_progress_loss, plan_progress_frame_acc = self._compute_plan_progress_loss_and_metrics(
            outputs.get("plan_progress_logits"),
            outputs.get("plan_input_lengths"),
        )
        if outputs.get("plan_occurrence_pred") is not None:
            plan_boundary_loss, plan_boundary_frame_acc = self._compute_segment_boundary_loss_and_metrics(
                outputs.get("plan_boundary_logits"),
                outputs.get("plan_input_lengths"),
                batch,
            )
        else:
            plan_boundary_loss, plan_boundary_frame_acc = self._compute_plan_boundary_loss_and_metrics(
                outputs.get("plan_boundary_logits"),
                outputs.get("plan_input_lengths"),
                batch,
            )
        plan_cursor_loss, plan_cursor_frame_acc, plan_cursor_targets = self._compute_plan_cursor_loss_and_metrics(
            outputs.get("plan_cursor_logits"),
            outputs.get("plan_input_lengths"),
            outputs.get("plan_text_lengths"),
        )
        plan_monotonic_loss = outputs.get("plan_monotonic_loss")
        if outputs.get("plan_occurrence_pred") is not None:
            plan_occurrence_loss = self._compute_plan_occurrence_loss(
                outputs.get("plan_occurrence_pred"),
                outputs.get("plan_input_lengths"),
                batch,
            )
            plan_remaining_loss = self._compute_plan_remaining_loss_from_occurrence(
                outputs.get("plan_remaining_pred"),
                outputs.get("plan_input_lengths"),
                batch,
            )
            plan_stop_loss, plan_stop_frame_acc = self._compute_plan_stop_loss_from_occurrence_and_metrics(
                outputs.get("plan_stop_logits"),
                outputs.get("plan_input_lengths"),
                batch,
            )
        else:
            plan_occurrence_loss = None
            plan_remaining_loss = self._compute_plan_remaining_loss(
                outputs.get("plan_remaining_pred"),
                plan_cursor_targets,
                outputs.get("plan_text_lengths"),
            )
            plan_stop_loss, plan_stop_frame_acc = self._compute_plan_stop_loss_and_metrics(
                outputs.get("plan_stop_logits"),
                plan_cursor_targets,
                outputs.get("plan_text_lengths"),
            )
        plan_activity_loss, plan_activity_frame_acc = self._compute_plan_activity_loss_and_metrics(
            outputs.get("plan_activity_logits"),
            outputs.get("plan_input_lengths"),
            batch,
        )
        plan_viseme_loss, plan_viseme_frame_acc = self._compute_plan_viseme_loss_and_metrics(
            outputs.get("plan_viseme_logits"),
            outputs.get("plan_input_lengths"),
            batch,
        )
        plan_phone_loss, plan_phone_frame_acc = self._compute_plan_phone_loss_and_metrics(
            outputs.get("plan_phone_logits"),
            outputs.get("plan_input_lengths"),
            batch,
        )
        q0_progress_loss, q0_progress_frame_acc = self._compute_q0_progress_loss_and_metrics(
            outputs.get("q0_progress_logits"),
            outputs.get("q0_output_lengths"),
        )
        q0_loop_loss, q0_loop_frame_acc = self._compute_q0_loop_loss_and_metrics(
            outputs.get("q0_loop_logits"),
            outputs.get("q0_target_tokens"),
            outputs.get("q0_output_lengths"),
        )
        q0_eos_loss, q0_eos_frame_acc = self._compute_q0_eos_loss_and_metrics(
            outputs.get("q0_eos_logits"),
            outputs.get("q0_output_lengths"),
        )
        late_nonblank_penalty = self._compute_late_nonblank_penalty(
            outputs.get("ctc_logits"),
            outputs.get("ctc_input_lengths"),
        )
        ctc_metrics = self._compute_ctc_metrics(
            outputs.get("ctc_logits"),
            outputs.get("ctc_input_lengths"),
            batch,
        )
        metrics = self.compute_loss_and_metrics(
            **outputs,
            ctc_loss=ctc_loss,
            align_loss=align_loss,
            align_frame_acc=align_frame_acc,
            progress_loss=progress_loss,
            progress_frame_acc=progress_frame_acc,
            boundary_loss=boundary_loss,
            boundary_frame_acc=boundary_frame_acc,
            plan_progress_loss=plan_progress_loss,
            plan_progress_frame_acc=plan_progress_frame_acc,
            plan_boundary_loss=plan_boundary_loss,
            plan_boundary_frame_acc=plan_boundary_frame_acc,
            plan_cursor_loss=plan_cursor_loss,
            plan_cursor_frame_acc=plan_cursor_frame_acc,
            plan_remaining_loss=plan_remaining_loss,
            plan_stop_loss=plan_stop_loss,
            plan_stop_frame_acc=plan_stop_frame_acc,
            plan_activity_loss=plan_activity_loss,
            plan_activity_frame_acc=plan_activity_frame_acc,
            plan_viseme_loss=plan_viseme_loss,
            plan_viseme_frame_acc=plan_viseme_frame_acc,
            plan_phone_loss=plan_phone_loss,
            plan_phone_frame_acc=plan_phone_frame_acc,
            plan_occurrence_loss=plan_occurrence_loss,
            q0_progress_loss=q0_progress_loss,
            q0_progress_frame_acc=q0_progress_frame_acc,
            q0_loop_loss=q0_loop_loss,
            q0_loop_frame_acc=q0_loop_frame_acc,
            q0_eos_loss=q0_eos_loss,
            q0_eos_frame_acc=q0_eos_frame_acc,
            late_nonblank_penalty=late_nonblank_penalty,
            **ctc_metrics,
        )
        util_metrics = self._compute_plan_utilization_losses(
            batch,
            metrics["ce_loss"],
        )
        metrics.update(util_metrics)
        routing_metrics = self._compute_routing_losses_and_metrics(
            outputs.get("routing_lambdas"),
            outputs.get("routing_valid_mask"),
        )
        metrics.update(routing_metrics)
        if metrics.get("plan_utilization_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["plan_utilization_loss"] * float(
                self.cfg.get("plan_utilization_loss_weight", 0.0)
            )
        if metrics.get("plan_ablation_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["plan_ablation_loss"] * float(
                self.cfg.get("plan_ablation_loss_weight", self.cfg.get("plan_utilization_loss_weight", 0.0))
            )
        if metrics.get("routing_order_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["routing_order_loss"] * float(
                self.cfg.get("routing_order_loss_weight", 0.0)
            )
        if metrics.get("routing_smoothness_loss") is not None:
            metrics["loss"] = metrics["loss"] + metrics["routing_smoothness_loss"] * float(
                self.cfg.get("routing_smoothness_loss_weight", 0.0)
            )
        video_ablation_gap = self._compute_video_ablation_gap(batch, metrics["ce_loss"])
        metrics["video_ablation_gap"] = video_ablation_gap
        exposure_probe_metrics = self._compute_online_exposure_probe_metrics(batch, outputs)
        metrics.update(exposure_probe_metrics)

        all_nt = metrics["effective_ntoken"]
        ce_weight = float(self.cfg.get("ce_loss_weight", 1.0))
        vq_weight = float(self.cfg.get("vq_loss_weight", 1.0))

        avg_ce_loss = metrics["ce_loss"]
        avg_loss = metrics["loss"]
        if torch.isnan(avg_loss) or torch.isinf(avg_loss):
            return None
        avg_acc = metrics["topk_acc"] / (all_nt + 1e-6)
        
        self.log("val/total_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/ce_loss", avg_ce_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/topk_acc", avg_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if metrics.get("vq_loss") is not None:
            self.log("val/vq_loss", metrics["vq_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("vq_usage") is not None:
            self.log("val/vq_usage", metrics["vq_usage"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ref_ce_loss") is not None:
            self.log("val/ref_ce_loss", metrics["ref_ce_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ref_topk_acc") is not None:
            self.log("val/ref_topk_acc", metrics["ref_topk_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("sync_loss") is not None:
            self.log("val/sync_loss", metrics["sync_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("sync_pos_alignment") is not None:
            self.log("val/sync_pos_alignment", metrics["sync_pos_alignment"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("sync_neg_alignment") is not None:
            self.log("val/sync_neg_alignment", metrics["sync_neg_alignment"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for key in (
            "g_embedding_cos_loss",
            "g_embedding_ce_loss",
            "g_embedding_cos_mean",
            "g_embedding_top1",
            "g_embedding_top10",
        ):
            if metrics.get(key) is not None:
                self.log(f"val/{key}", metrics[key], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for q in range(self.n_codebooks):
            for key in (
                f"g_embedding_cos_q{q}",
                f"g_embedding_ce_q{q}",
                f"g_embedding_top1_q{q}",
                f"g_embedding_top10_q{q}",
            ):
                if metrics.get(key) is not None:
                    self.log(f"val/{key}", metrics[key], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_loss") is not None:
            self.log("val/ctc_loss", metrics["ctc_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("align_loss") is not None:
            self.log("val/align_loss", metrics["align_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("align_frame_acc") is not None:
            self.log("val/align_frame_acc", metrics["align_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("progress_loss") is not None:
            self.log("val/progress_loss", metrics["progress_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("progress_frame_acc") is not None:
            self.log("val/progress_frame_acc", metrics["progress_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("boundary_loss") is not None:
            self.log("val/boundary_loss", metrics["boundary_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("boundary_frame_acc") is not None:
            self.log("val/boundary_frame_acc", metrics["boundary_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_progress_loss") is not None:
            self.log("val/plan_progress_loss", metrics["plan_progress_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_progress_frame_acc") is not None:
            self.log("val/plan_progress_frame_acc", metrics["plan_progress_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_boundary_loss") is not None:
            self.log("val/plan_boundary_loss", metrics["plan_boundary_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_boundary_frame_acc") is not None:
            self.log("val/plan_boundary_frame_acc", metrics["plan_boundary_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_acoustic_loss") is not None:
            self.log("val/plan_acoustic_loss", metrics["plan_acoustic_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_acoustic_pos_alignment") is not None:
            self.log("val/plan_acoustic_pos_alignment", metrics["plan_acoustic_pos_alignment"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_acoustic_neg_alignment") is not None:
            self.log("val/plan_acoustic_neg_alignment", metrics["plan_acoustic_neg_alignment"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_cursor_loss") is not None:
            self.log("val/plan_cursor_loss", metrics["plan_cursor_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_cursor_frame_acc") is not None:
            self.log("val/plan_cursor_frame_acc", metrics["plan_cursor_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_monotonic_loss") is not None:
            self.log("val/plan_monotonic_loss", metrics["plan_monotonic_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_remaining_loss") is not None:
            self.log("val/plan_remaining_loss", metrics["plan_remaining_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_stop_loss") is not None:
            self.log("val/plan_stop_loss", metrics["plan_stop_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_stop_frame_acc") is not None:
            self.log("val/plan_stop_frame_acc", metrics["plan_stop_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_activity_loss") is not None:
            self.log("val/plan_activity_loss", metrics["plan_activity_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_activity_frame_acc") is not None:
            self.log("val/plan_activity_frame_acc", metrics["plan_activity_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_viseme_loss") is not None:
            self.log("val/plan_viseme_loss", metrics["plan_viseme_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_viseme_frame_acc") is not None:
            self.log("val/plan_viseme_frame_acc", metrics["plan_viseme_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_phone_loss") is not None:
            self.log("val/plan_phone_loss", metrics["plan_phone_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_phone_frame_acc") is not None:
            self.log("val/plan_phone_frame_acc", metrics["plan_phone_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_occurrence_loss") is not None:
            self.log("val/plan_occurrence_loss", metrics["plan_occurrence_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if outputs.get("plan_viseme_compatibility_mean") is not None:
            self.log("val/plan_viseme_compatibility_mean", outputs["plan_viseme_compatibility_mean"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("q0_progress_loss") is not None:
            self.log("val/q0_progress_loss", metrics["q0_progress_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("q0_progress_frame_acc") is not None:
            self.log("val/q0_progress_frame_acc", metrics["q0_progress_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("q0_loop_loss") is not None:
            self.log("val/q0_loop_loss", metrics["q0_loop_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("q0_loop_frame_acc") is not None:
            self.log("val/q0_loop_frame_acc", metrics["q0_loop_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("q0_eos_loss") is not None:
            self.log("val/q0_eos_loss", metrics["q0_eos_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("q0_eos_frame_acc") is not None:
            self.log("val/q0_eos_frame_acc", metrics["q0_eos_frame_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if outputs.get("q0_planner_adapter_delta_norm") is not None:
            self.log("val/q0_planner_adapter_delta_norm", outputs["q0_planner_adapter_delta_norm"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if outputs.get("q0_planner_adapter_gate") is not None:
            self.log("val/q0_planner_adapter_gate", outputs["q0_planner_adapter_gate"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_utilization_loss") is not None:
            self.log("val/plan_utilization_loss", metrics["plan_utilization_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_ablation_loss") is not None:
            self.log("val/plan_ablation_loss", metrics["plan_ablation_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("routing_order_loss") is not None:
            self.log("val/routing_order_loss", metrics["routing_order_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("routing_smoothness_loss") is not None:
            self.log("val/routing_smoothness_loss", metrics["routing_smoothness_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("routing_order_margin_mean") is not None:
            self.log("val/routing_order_margin_mean", metrics["routing_order_margin_mean"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("routing_smoothness") is not None:
            self.log("val/routing_smoothness", metrics["routing_smoothness"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_shift_ce") is not None:
            self.log("val/plan_shift_ce", metrics["plan_shift_ce"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_shift_ce_gap") is not None:
            self.log("val/plan_shift_ce_gap", metrics["plan_shift_ce_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_no_plan_ce") is not None:
            self.log("val/plan_no_plan_ce", metrics["plan_no_plan_ce"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_no_plan_ce_gap") is not None:
            self.log("val/plan_no_plan_ce_gap", metrics["plan_no_plan_ce_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("plan_counterfactual_shift") is not None:
            self.log("val/plan_counterfactual_shift", metrics["plan_counterfactual_shift"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("state_loss") is not None:
            self.log("val/state_loss", metrics["state_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("state_huber_loss") is not None:
            self.log("val/state_huber_loss", metrics["state_huber_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("state_cos_loss") is not None:
            self.log("val/state_cos_loss", metrics["state_cos_loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("state_cosine") is not None:
            self.log("val/state_cosine", metrics["state_cosine"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("state_l2") is not None:
            self.log("val/state_l2", metrics["state_l2"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("state_valid_ratio") is not None:
            self.log("val/state_valid_ratio", metrics["state_valid_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("late_nonblank_penalty") is not None:
            self.log("val/late_nonblank_penalty", metrics["late_nonblank_penalty"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_greedy_per") is not None:
            self.log("val/ctc_greedy_per", metrics["ctc_greedy_per"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_prefix_per") is not None:
            self.log("val/ctc_prefix_per", metrics["ctc_prefix_per"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_blank_ratio") is not None:
            self.log("val/ctc_blank_ratio", metrics["ctc_blank_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_early_blank_ratio") is not None:
            self.log("val/ctc_early_blank_ratio", metrics["ctc_early_blank_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_late_blank_ratio") is not None:
            self.log("val/ctc_late_blank_ratio", metrics["ctc_late_blank_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_greedy_len_ratio") is not None:
            self.log("val/ctc_greedy_len_ratio", metrics["ctc_greedy_len_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("ctc_tail_nonblank_ratio") is not None:
            self.log("val/ctc_tail_nonblank_ratio", metrics["ctc_tail_nonblank_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("completion_before_tail_ratio") is not None:
            self.log("val/completion_before_tail_ratio", metrics["completion_before_tail_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("forced_cutoff_risk") is not None:
            self.log("val/forced_cutoff_risk", metrics["forced_cutoff_risk"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("video_ablation_gap") is not None:
            self.log("val/video_ablation_gap", metrics["video_ablation_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("routing_lambda_means") is not None:
            for q, val in enumerate(metrics["routing_lambda_means"]):
                self.log(
                    f"val/routing_lambda_mean_q{q}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
        if outputs.get("hier_context_norm_means") is not None:
            for q, val in enumerate(outputs["hier_context_norm_means"]):
                self.log(
                    f"val/hier_context_norm_q{q}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
        if outputs.get("hier_context_scales") is not None:
            for q, val in enumerate(outputs["hier_context_scales"]):
                self.log(
                    f"val/hier_context_scale_q{q}",
                    val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=True,
                )
        if metrics.get("tf_target_token_acc") is not None:
            self.log("val/tf_target_token_acc", metrics["tf_target_token_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("tf_target_mean_ned") is not None:
            self.log("val/tf_target_mean_ned", metrics["tf_target_mean_ned"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("tf_target_unfinished_ratio") is not None:
            self.log("val/tf_target_unfinished_ratio", metrics["tf_target_unfinished_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("tf_target_coverage_ratio") is not None:
            self.log("val/tf_target_coverage_ratio", metrics["tf_target_coverage_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_target_token_acc") is not None:
            self.log("val/exposure_probe_target_token_acc", metrics["exposure_probe_target_token_acc"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_target_mean_ned") is not None:
            self.log("val/exposure_probe_target_mean_ned", metrics["exposure_probe_target_mean_ned"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_target_unfinished_ratio") is not None:
            self.log("val/exposure_probe_target_unfinished_ratio", metrics["exposure_probe_target_unfinished_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_target_coverage_ratio") is not None:
            self.log("val/exposure_probe_target_coverage_ratio", metrics["exposure_probe_target_coverage_ratio"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_acc_gap") is not None:
            self.log("val/exposure_probe_acc_gap", metrics["exposure_probe_acc_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_ned_gap") is not None:
            self.log("val/exposure_probe_ned_gap", metrics["exposure_probe_ned_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_unfinished_gap") is not None:
            self.log("val/exposure_probe_unfinished_gap", metrics["exposure_probe_unfinished_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_acc_retention") is not None:
            self.log("val/exposure_probe_acc_retention", metrics["exposure_probe_acc_retention"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_coverage_gap") is not None:
            self.log("val/exposure_probe_coverage_gap", metrics["exposure_probe_coverage_gap"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_coverage_retention") is not None:
            self.log("val/exposure_probe_coverage_retention", metrics["exposure_probe_coverage_retention"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_hc_rate") is not None:
            self.log("val/exposure_probe_hc_rate", metrics["exposure_probe_hc_rate"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if metrics.get("exposure_probe_ss_rate") is not None:
            self.log("val/exposure_probe_ss_rate", metrics["exposure_probe_ss_rate"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        for layer_idx, gate_value in outputs.get("in_decoder_gate_values", {}).items():
            self.log(
                f"val/in_decoder_gate_value_layer{layer_idx}",
                gate_value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )
        for layer_idx, delta_norm in outputs.get("in_decoder_fusion_delta_norms", {}).items():
            self.log(
                f"val/in_decoder_fusion_delta_norm_layer{layer_idx}",
                delta_norm,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        cb_logs = {}
        for k, acc_weighted_sum in enumerate(metrics["topk_acc_by_codebook"]):
            cb_logs[f"val/acc_cb_{k}"] = acc_weighted_sum / (all_nt + 1e-6)
        
        self.log_dict(cb_logs, sync_dist=True)
        return avg_loss

    def on_validation_epoch_end(self):
        # Retrieve aggregated metrics from trainer.callback_metrics
        # Note: keys must match what you used in self.log() inside validation_step
        avg_loss = self.trainer.callback_metrics.get("val/total_loss")
        avg_acc = self.trainer.callback_metrics.get("val/topk_acc")

        if avg_loss is not None and avg_acc is not None:
            logger.info(f"Validation Finish: Loss={avg_loss.item():.4f}, Acc={avg_acc.item():.4f})")
        return avg_loss

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["config"] = self.cfg
        # # Save phn2num if available in datamodule
        # if self.trainer and self.trainer.datamodule and hasattr(self.trainer.datamodule, "train_dataset"):
        #     dataset = self.trainer.datamodule.train_dataset
        #     if hasattr(dataset, "phn2num"):
        #         checkpoint["phn2num"] = dataset.phn2num


    def _setup_model(self):
        self.model = instantiate(self.cfg.model)

        logger.info("VoiceCraftDub Model Summary:")
        logger.info(self.model)

        if self.cfg.get("ckpt_path", None) is not None:
            logger.info(f"Resume model weights from {self.cfg.ckpt_path}")
            return
        else:
            load_from = self.cfg.get("load_original_model_from", None)
            if load_from is not None:
                logger.info(f"Loading model weights from {load_from}")
                ckpt = load_ckpt_from_origin(load_from)
                model_state = self.model.state_dict()
                load_state = {}
                skipped = []
                for k, v in ckpt["state_dict"].items():
                    if k in model_state and model_state[k].shape == v.shape:
                        load_state[k] = v
                    else:
                        skipped.append(k)

                missing, unexpected = self.model.load_state_dict(
                    load_state, strict=False
                )
                self._load_partial_pretrained_tensors(
                    ckpt_state=ckpt["state_dict"],
                    model_state=model_state,
                )
                if skipped:
                    logger.info(
                        "Skipped %d keys due to missing/mismatch shape when loading pretrained weights.",
                        len(skipped),
                    )
                if missing:
                    logger.info(
                        "Missing %d keys after loading pretrained weights.",
                        len(missing),
                    )
                if unexpected:
                    logger.info(
                        "Unexpected %d keys after loading pretrained weights.",
                        len(unexpected),
                    )

    def _apply_parameter_training_rules(self) -> None:
        trainable_patterns = self.cfg.get("trainable_patterns", None)
        freeze_patterns = self.cfg.get("freeze_patterns", None)

        if not trainable_patterns and not freeze_patterns:
            return

        if trainable_patterns:
            for param in self.model.parameters():
                param.requires_grad = False

            matched_names = []
            trainable_params = 0
            for name, param in self.model.named_parameters():
                full_name = f"model.{name}"
                if any(pattern in name or pattern in full_name for pattern in trainable_patterns):
                    param.requires_grad = True
                    matched_names.append(full_name)
                    trainable_params += param.numel()

            logger.info(
                "Applied trainable_patterns. Matched %d parameter tensors, %d trainable parameters.",
                len(matched_names),
                trainable_params,
            )
            if matched_names:
                logger.info("Trainable parameter names:\n%s", "\n".join(matched_names))
            return

        frozen_names = []
        frozen_params = 0
        for name, param in self.model.named_parameters():
            full_name = f"model.{name}"
            if any(pattern in name or pattern in full_name for pattern in freeze_patterns):
                param.requires_grad = False
                frozen_names.append(full_name)
                frozen_params += param.numel()

        logger.info(
            "Applied freeze_patterns. Matched %d parameter tensors, %d frozen parameters.",
            len(frozen_names),
            frozen_params,
        )
        if frozen_names:
            logger.info("Frozen parameter names:\n%s", "\n".join(frozen_names))

    def _load_partial_pretrained_tensors(
        self,
        ckpt_state: Dict[str, torch.Tensor],
        model_state: Dict[str, torch.Tensor],
    ) -> None:
        text_key = "text_embedding.word_embeddings.weight"
        if text_key in ckpt_state and text_key in model_state:
            pretrained = ckpt_state[text_key]
            current = model_state[text_key]
            if (
                pretrained.ndim == 2
                and current.ndim == 2
                and pretrained.shape != current.shape
                and pretrained.shape[1] == current.shape[1]
                and pretrained.shape[0] <= current.shape[0]
            ):
                with torch.no_grad():
                    self.model.text_embedding.word_embeddings.weight[: pretrained.shape[0]].copy_(pretrained)
                logger.info(
                    "Partially loaded %s rows for expanded text embedding (%d -> %d).",
                    text_key,
                    pretrained.shape[0],
                    current.shape[0],
                )

        # Continue loading partially compatible output heads when repr_dim expands
        # from older 512-dim g space to newer 2048-dim embedding-space g heads.
        for q in range(self.n_codebooks):
            repr_w_key = f"predict_layer.heads.{q}.repr_proj.weight"
            repr_b_key = f"predict_layer.heads.{q}.repr_proj.bias"
            cls_w_key = f"predict_layer.heads.{q}.classifier.weight"
            cls_b_key = f"predict_layer.heads.{q}.classifier.bias"

            if repr_w_key in ckpt_state and repr_w_key in model_state:
                old_w = ckpt_state[repr_w_key]
                new_w = model_state[repr_w_key]
                if (
                    old_w.ndim == 2
                    and new_w.ndim == 2
                    and old_w.shape[1] == new_w.shape[1]
                    and old_w.shape[0] < new_w.shape[0]
                ):
                    with torch.no_grad():
                        self.model.state_dict()[repr_w_key][: old_w.shape[0]].copy_(old_w)
                    logger.info(
                        "Partially loaded %s rows for expanded repr_proj (%d -> %d).",
                        repr_w_key,
                        old_w.shape[0],
                        new_w.shape[0],
                    )

            if repr_b_key in ckpt_state and repr_b_key in model_state:
                old_b = ckpt_state[repr_b_key]
                new_b = model_state[repr_b_key]
                if (
                    old_b.ndim == 1
                    and new_b.ndim == 1
                    and old_b.shape[0] < new_b.shape[0]
                ):
                    with torch.no_grad():
                        self.model.state_dict()[repr_b_key][: old_b.shape[0]].copy_(old_b)

            if cls_w_key in ckpt_state and cls_w_key in model_state:
                old_w = ckpt_state[cls_w_key]
                new_w = model_state[cls_w_key]
                if (
                    old_w.ndim == 2
                    and new_w.ndim == 2
                    and old_w.shape[0] == new_w.shape[0]
                    and old_w.shape[1] < new_w.shape[1]
                ):
                    with torch.no_grad():
                        self.model.state_dict()[cls_w_key][:, : old_w.shape[1]].copy_(old_w)
                    logger.info(
                        "Partially loaded %s columns for expanded classifier (%d -> %d).",
                        cls_w_key,
                        old_w.shape[1],
                        new_w.shape[1],
                    )

            if cls_b_key in ckpt_state and cls_b_key in model_state:
                old_b = ckpt_state[cls_b_key]
                new_b = model_state[cls_b_key]
                if old_b.shape == new_b.shape:
                    with torch.no_grad():
                        self.model.state_dict()[cls_b_key].copy_(old_b)
    
    
    def configure_optimizers(self):
        optimizer_cfg = self.cfg.get("optimizer", None)
        scheduler_cfg = self.cfg.get("scheduler", None)
        if optimizer_cfg is None:
            raise ValueError("Optimizer configuration is missing in cfg.optimizer")
        if isinstance(optimizer_cfg, DictConfig):
            optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
        else:
            optimizer_cfg = dict(optimizer_cfg)
        if isinstance(scheduler_cfg, DictConfig):
            scheduler_cfg = OmegaConf.to_container(scheduler_cfg, resolve=True)
        elif scheduler_cfg is not None:
            scheduler_cfg = dict(scheduler_cfg)
        
        if self.trainer.max_steps and self.trainer.max_steps > 0:
            total_steps = self.trainer.max_steps
        else:
            total_steps = self.trainer.estimated_stepping_batches
        
        optimizer_name = optimizer_cfg.pop("_target_")
        if optimizer_name == "ScaledAdam":
            trainables = [p for p in self.parameters() if p.requires_grad]
            parameters_names = [[n for n, p in self.named_parameters() if p.requires_grad]]
            
            optimizer = ScaledAdam(
                trainables,
                parameters_names=parameters_names,
                **optimizer_cfg,
            )
            
            # Eden Scheduler
            scheduler = Eden(
                optimizer, 
                lr_batches=scheduler_cfg.get("reduce_lr_start_step"),
                lr_epochs=scheduler_cfg.get("reduce_lr_start_epoch"),
                warmup_batches=int(total_steps * self.cfg.scheduler.get("warmup_fraction", 0.1))
            )

        else: # Default: AdamW
            base_lr = optimizer_cfg.get("lr", 1e-5)
            weight_decay = optimizer_cfg.get("weight_decay", 0.01)

            no_decay = [
                ".bias",
                ".audio_embeddings.",
                ".text_embeddings.weight",
                ".norm.weight",
                ".norm1.weight",
                ".norm2.weight",
            ]
            param_group_rules = optimizer_cfg.pop("param_groups", None)

            if param_group_rules:
                named_params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
                used = set()
                optim_groups = []

                for rule in param_group_rules:
                    group_name = rule.get("name", "group")
                    include = rule.get("include", [])
                    exclude = rule.get("exclude", [])
                    group_lr = float(rule.get("lr", base_lr))
                    group_weight_decay = float(rule.get("weight_decay", weight_decay))
                    group_no_decay_weight = float(rule.get("no_decay_weight_decay", 0.0))

                    matched = [
                        (n, p) for n, p in named_params
                        if n not in used
                        and any(pattern in n for pattern in include)
                        and not any(pattern in n for pattern in exclude)
                    ]

                    decay_params = []
                    no_decay_params = []
                    for n, p in matched:
                        used.add(n)
                        if any(nd in n for nd in no_decay):
                            no_decay_params.append(p)
                        else:
                            decay_params.append(p)

                    if decay_params:
                        optim_groups.append(
                            {
                                "params": decay_params,
                                "weight_decay": group_weight_decay,
                                "lr": group_lr,
                                "group_name": f"{group_name}_decay",
                            }
                        )
                    if no_decay_params:
                        optim_groups.append(
                            {
                                "params": no_decay_params,
                                "weight_decay": group_no_decay_weight,
                                "lr": group_lr,
                                "group_name": f"{group_name}_no_decay",
                            }
                        )

                fallback_decay = []
                fallback_no_decay = []
                for n, p in named_params:
                    if n in used:
                        continue
                    if any(nd in n for nd in no_decay):
                        fallback_no_decay.append(p)
                    else:
                        fallback_decay.append(p)

                if fallback_decay:
                    optim_groups.append(
                        {
                            "params": fallback_decay,
                            "weight_decay": weight_decay,
                            "lr": base_lr,
                            "group_name": "base_decay",
                        }
                    )
                if fallback_no_decay:
                    optim_groups.append(
                        {
                            "params": fallback_no_decay,
                            "weight_decay": 0.0,
                            "lr": base_lr,
                            "group_name": "base_no_decay",
                        }
                    )
            else:
                params_decay = []
                params_no_decay = []

                for n, p in self.named_parameters():
                    if not p.requires_grad:
                        continue
                    if any(nd in n for nd in no_decay):
                        params_no_decay.append(p)
                    else:
                        params_decay.append(p)

                optim_groups = [
                    {
                        "params": params_decay,
                        "weight_decay": weight_decay,
                        "lr": base_lr,
                        "group_name": "base_decay",
                    },
                    {
                        "params": params_no_decay,
                        "weight_decay": 0.0,
                        "lr": base_lr,
                        "group_name": "base_no_decay",
                    },
                ]

            optimizer_kwargs = {
                k: v for k, v in optimizer_cfg.items()
                if k not in {"lr", "weight_decay"}
            }
            optimizer = torch.optim.AdamW(optim_groups, lr=base_lr, weight_decay=weight_decay, **optimizer_kwargs)
            warmup_fraction = self.cfg.scheduler.get("warmup_fraction", 0.1)
            warmup_steps = int(total_steps * warmup_fraction)

            def lr_lambda(current_step: int):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",
                "interval": self.cfg.get("scheduler_interval", "step"),
                "frequency": 1
            }
        }
    
    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        if not self.is_scaled_adam:
            self.clip_gradients(optimizer, gradient_clip_val=gradient_clip_val, gradient_clip_algorithm=gradient_clip_algorithm)
