#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def read_manifest(path: Path, limit: int) -> list[str]:
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ids.append(parts[1])
            if limit > 0 and len(ids) >= limit:
                break
    return ids


def load_audio_ffmpeg(path: Path, sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-",
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class ShiftedRealAVDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        limit: int,
        window_frames: int,
        offsets: list[int],
        sample_rate: int = 16000,
    ):
        self.dataset_dir = dataset_dir
        self.ids = read_manifest(dataset_dir / "manifest" / f"{split}.txt", limit=limit)
        if not self.ids:
            raise ValueError(f"No ids found for split={split}")
        self.window_frames = int(window_frames)
        self.offsets = offsets
        self.sample_rate = int(sample_rate)
        self.frame_samples = self.sample_rate // 25
        self.max_offset = max(abs(x) for x in offsets)
        self.lip_dir = dataset_dir / "lip_feature"
        self.video_dir = dataset_dir / "video_origin"

    def __len__(self) -> int:
        return len(self.ids)

    def _load_pair(self, pair_id: str) -> tuple[torch.Tensor, torch.Tensor]:
        speaker, _, target_utt = pair_id.split("__")
        base = f"{speaker}_{target_utt}"
        lip = np.load(self.lip_dir / f"{base}.npy")
        if lip.ndim == 2 and lip.shape[0] == 1024:
            lip = lip.T
        video = torch.from_numpy(lip).float()
        audio = torch.from_numpy(load_audio_ffmpeg(self.video_dir / f"{base}.mp4", sr=self.sample_rate)).float()
        return audio, video

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        for _ in range(96):
            pair_id = self.ids[idx % len(self.ids)]
            try:
                audio, video = self._load_pair(pair_id)
                audio_frames = audio.shape[0] // self.frame_samples
                length = min(int(video.shape[0]), int(audio_frames))
                min_len = self.window_frames + 2 * self.max_offset + 2
                if length < min_len:
                    raise ValueError("short clip")
                label_idx = random.randrange(len(self.offsets))
                offset = int(self.offsets[label_idx])
                start = random.randint(self.max_offset, length - self.window_frames - self.max_offset - 1)
                audio_start_frame = start + offset
                audio_start = audio_start_frame * self.frame_samples
                audio_end = audio_start + self.window_frames * self.frame_samples
                audio_win = audio[audio_start:audio_end]
                expected_len = self.window_frames * self.frame_samples
                if audio_win.numel() != expected_len:
                    raise ValueError("bad audio window")
                video_win = video[start:start + self.window_frames]
                return {
                    "audio": audio_win,
                    "video": video_win,
                    "label": torch.tensor(label_idx, dtype=torch.long),
                }
            except Exception:
                idx = random.randrange(len(self.ids))
        raise RuntimeError("failed to load a valid shifted AV sample")


class WavLMAVSyncScorer(nn.Module):
    def __init__(
        self,
        num_offsets: int,
        hidden_dim: int = 256,
        audio_ssl: str = "WAVLM_BASE_PLUS",
    ):
        super().__init__()
        bundle = getattr(torchaudio.pipelines, audio_ssl)
        self.ssl_sample_rate = int(bundle.sample_rate)
        self.ssl = bundle.get_model()
        for param in self.ssl.parameters():
            param.requires_grad = False
        self.ssl.eval()
        ssl_dim = int(getattr(bundle, "_params", {}).get("encoder_embed_dim", 768))
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(ssl_dim),
            nn.Linear(ssl_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.video_proj = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden_dim * 4, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_dim, num_offsets)

    def _extract_audio_features(self, audio: torch.Tensor, target_len: int) -> torch.Tensor:
        self.ssl.eval()
        with torch.no_grad():
            features, _ = self.ssl(audio)
        features = features.transpose(1, 2)
        features = F.interpolate(features, size=target_len, mode="linear", align_corners=False)
        return features.transpose(1, 2)

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:
        audio_features = self._extract_audio_features(audio, target_len=video.shape[1])
        a = F.normalize(self.audio_proj(audio_features), dim=-1)
        v = F.normalize(self.video_proj(video), dim=-1)
        fused = torch.cat([a, v, (a - v).abs(), a * v], dim=-1).transpose(1, 2)
        pooled = self.temporal(fused).mean(dim=-1)
        return self.head(pooled)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, zero_idx: int) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    zero_total = 0
    zero_correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device, non_blocking=True)
            video = batch["video"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            logits = model(audio, video)
            loss = F.cross_entropy(logits, label)
            pred = logits.argmax(dim=-1)
            total += int(label.numel())
            correct += int((pred == label).sum().item())
            loss_sum += float(loss.item()) * int(label.numel())
            mask = label == zero_idx
            if bool(mask.any().item()):
                zero_total += int(mask.sum().item())
                zero_correct += int((pred[mask] == label[mask]).sum().item())
    return {
        "val_loss": loss_sum / max(total, 1),
        "val_acc": correct / max(total, 1),
        "val_zero_acc": zero_correct / max(zero_total, 1),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    offsets: list[int],
    args: argparse.Namespace,
    wandb_id: Optional[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "global_step": global_step,
            "offsets": offsets,
            "window_frames": args.window_frames,
            "hidden_dim": args.hidden_dim,
            "audio_ssl": args.audio_ssl,
            "wandb_id": wandb_id,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, optimizer: Optional[torch.optim.Optimizer]) -> tuple[int, Optional[str]]:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("global_step", 0)), ckpt.get("wandb_id")


def init_wandb(args: argparse.Namespace, resume_wandb_id: Optional[str]):
    if not args.wandb:
        return None
    import wandb

    run_id = args.wandb_id or resume_wandb_id or None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=run_id,
        resume="allow" if run_id else None,
        config=vars(args),
    )
    if args.wandb_id_file:
        Path(args.wandb_id_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.wandb_id_file).write_text(run.id + "\n", encoding="utf-8")
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="/data1/jinyu_wang/projects/PILOT-Dub/data/dataset/LRS3_Dataset/mp4/trainval_preprocess")
    parser.add_argument("--output-dir", default="/data1/jinyu_wang/projects/PILOT-Dub/artifacts/PILOT-Dub/avsync_scorer_training")
    parser.add_argument("--resume", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audio-ssl", default="WAVLM_BASE_PLUS")
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--val-limit", type=int, default=2000)
    parser.add_argument("--window-frames", type=int, default=48)
    parser.add_argument("--offsets", default="-8,-6,-4,-2,0,2,4,6,8")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="PILOT-Dub")
    parser.add_argument("--wandb-name", default="wavlm-avsync-scorer")
    parser.add_argument("--wandb-id", default="")
    parser.add_argument("--wandb-id-file", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    offsets = [int(x) for x in args.offsets.split(",")]
    zero_idx = offsets.index(0)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")

    train_ds = ShiftedRealAVDataset(Path(args.dataset_dir), "train", args.train_limit, args.window_frames, offsets)
    val_ds = ShiftedRealAVDataset(Path(args.dataset_dir), "validation", args.val_limit, args.window_frames, offsets)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = WavLMAVSyncScorer(
        num_offsets=len(offsets),
        hidden_dim=args.hidden_dim,
        audio_ssl=args.audio_ssl,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    global_step = 0
    resume_wandb_id = None
    if args.resume:
        global_step, resume_wandb_id = load_checkpoint(Path(args.resume), model, optimizer)
        print(f"resumed from {args.resume} at step={global_step}", flush=True)
    run = init_wandb(args, resume_wandb_id)
    wandb_id = getattr(run, "id", None)

    output_dir = Path(args.output_dir)
    best_acc = -1.0
    train_iter = itertools.cycle(train_loader)
    while global_step < args.max_steps:
        model.train()
        batch = next(train_iter)
        audio = batch["audio"].to(device, non_blocking=True)
        video = batch["video"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(audio, video)
        loss = F.cross_entropy(logits, label)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        global_step += 1

        pred = logits.detach().argmax(dim=-1)
        train_acc = float((pred == label).float().mean().item())
        if global_step % args.log_every == 0 or global_step == 1:
            payload = {"train/loss": float(loss.item()), "train/acc": train_acc, "step": global_step}
            print(json.dumps(payload, ensure_ascii=False), flush=True)
            if run is not None:
                run.log(payload, step=global_step)

        if global_step % args.val_every == 0 or global_step == args.max_steps:
            metrics = evaluate(model, val_loader, device, zero_idx)
            metrics["step"] = global_step
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if run is not None:
                run.log({f"val/{k[4:] if k.startswith('val_') else k}": v for k, v in metrics.items()}, step=global_step)
            if metrics["val_acc"] > best_acc:
                best_acc = metrics["val_acc"]
                save_checkpoint(output_dir / "best.pt", model, optimizer, global_step, offsets, args, wandb_id)

        if global_step % args.save_every == 0 or global_step == args.max_steps:
            save_checkpoint(
                output_dir / f"resume_step={global_step}.pt",
                model,
                optimizer,
                global_step,
                offsets,
                args,
                wandb_id,
            )

    save_checkpoint(output_dir / "last.pt", model, optimizer, global_step, offsets, args, wandb_id)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
