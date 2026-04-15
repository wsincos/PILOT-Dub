#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_wavlm_avsync_scorer import WavLMAVSyncScorer


def load_summary(model_root: Path, model_name: str, system: str) -> dict[str, dict]:
    summary_path = model_root / model_name / system / "summary.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    rows: dict[str, dict] = {}
    with summary_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            sample_id = f"{row['speaker']}/{row['target_id']}"
            rows[sample_id] = row
    return rows


def copy_candidate_files(src_dir: Path, dst_dir: Path, target_id: str) -> list[str]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for src in sorted(src_dir.glob(f"{target_id}*")):
        if src.is_file() and src.suffix.lower() in {".wav", ".mp4", ".npy"}:
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            copied.append(str(dst))
    if not copied:
        raise FileNotFoundError(f"No candidate files found for {src_dir}/{target_id}*")
    return copied


def load_scorer(checkpoint_path: Path, device: torch.device) -> tuple[WavLMAVSyncScorer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = WavLMAVSyncScorer(
        num_offsets=len(checkpoint["offsets"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        audio_ssl=str(checkpoint.get("audio_ssl", "WAVLM_BASE_PLUS")),
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def read_wav_16k(path: Path) -> torch.Tensor:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"Expected 16k wav, got sr={sr}: {path}")
    return torch.from_numpy(audio.astype(np.float32))


def score_candidate(
    model: WavLMAVSyncScorer,
    checkpoint: dict,
    wav_path: Path,
    feature_path: Path,
    device: torch.device,
    num_windows: int,
) -> float:
    window = int(checkpoint.get("window_frames", checkpoint.get("window", 48)))
    offsets = list(checkpoint["offsets"])
    zero_idx = offsets.index(0)
    frame_samples = 16000 // 25

    audio = read_wav_16k(wav_path)
    video_np = np.load(feature_path)
    if video_np.ndim == 2 and video_np.shape[0] == 1024:
        video_np = video_np.T
    video = torch.from_numpy(video_np.astype(np.float32))
    length = min(int(audio.numel() // frame_samples), int(video.shape[0]))
    if length < window:
        return -1e9
    if num_windows <= 1:
        starts = [max(0, (length - window) // 2)]
    else:
        starts = np.linspace(0, length - window, num=num_windows).round().astype(int).tolist()

    scores = []
    with torch.no_grad():
        for start in starts:
            a0 = start * frame_samples
            a1 = a0 + window * frame_samples
            audio_win = audio[a0:a1].unsqueeze(0).to(device)
            video_win = video[start:start + window].unsqueeze(0).to(device)
            logits = model(audio_win, video_win)
            probs = F.softmax(logits, dim=-1)[0]
            other = torch.cat([probs[:zero_idx], probs[zero_idx + 1 :]]).max()
            margin = probs[zero_idx] - other
            scores.append(float(margin.item()))
    return float(sum(scores) / len(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Select generated candidates with an independent AV sync scorer.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="LRS3-mini50")
    parser.add_argument("--model-out-root", default="/data1/jinyu_wang/projects/metrics/model_out")
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate model_out names.")
    parser.add_argument("--selected-model-name", required=True)
    parser.add_argument("--systems", default="system1,system2")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-windows", type=int, default=5)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--fallback-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    if not names:
        raise ValueError("No candidate names provided")
    if not (0 <= args.fallback_index < len(names)):
        raise ValueError(f"Invalid fallback index: {args.fallback_index}")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model, checkpoint = load_scorer(Path(args.checkpoint), device)

    model_root = Path(args.model_out_root) / args.dataset
    selected_root = model_root / args.selected_model_name
    selected_root.mkdir(parents=True, exist_ok=True)

    report = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "candidate_names": names,
        "selected_model_name": args.selected_model_name,
        "num_windows": args.num_windows,
        "margin_threshold": args.margin_threshold,
        "fallback_index": args.fallback_index,
        "systems": {},
    }

    for system in [s.strip() for s in args.systems.split(",") if s.strip()]:
        summaries = [load_summary(model_root, name, system) for name in names]
        sample_ids = sorted(summaries[args.fallback_index].keys())
        system_dir = selected_root / system
        if system_dir.exists():
            shutil.rmtree(system_dir)
        system_dir.mkdir(parents=True, exist_ok=True)

        choices = {name: 0 for name in names}
        selected_rows = []
        per_sample = {}

        for sample_id in sample_ids:
            speaker, target_id = sample_id.split("/")
            scores = []
            for idx, name in enumerate(names):
                row = summaries[idx].get(sample_id)
                if row is None:
                    scores.append(float("-inf"))
                    continue
                wav_path = model_root / name / system / speaker / f"{target_id}.wav"
                feature_path = Path(row["feature_path"])
                score = score_candidate(
                    model=model,
                    checkpoint=checkpoint,
                    wav_path=wav_path,
                    feature_path=feature_path,
                    device=device,
                    num_windows=args.num_windows,
                )
                scores.append(score)

            best_idx = int(np.argmax(scores))
            fallback_score = scores[args.fallback_index]
            if not np.isfinite(scores[best_idx]):
                best_idx = args.fallback_index
            elif args.margin_threshold > 0 and (scores[best_idx] - fallback_score) < args.margin_threshold:
                best_idx = args.fallback_index

            selected_name = names[best_idx]
            choices[selected_name] += 1
            selected_row = dict(summaries[best_idx][sample_id])
            src_dir = model_root / selected_name / system / speaker
            dst_dir = system_dir / speaker
            copied = copy_candidate_files(src_dir, dst_dir, target_id)
            selected_row["source_candidate_name"] = selected_name
            selected_row["source_candidate_index"] = best_idx
            selected_row["source_candidate_scores"] = {names[i]: scores[i] for i in range(len(names))}
            selected_row["output_dir"] = str(dst_dir)
            selected_row["copied_files"] = copied
            selected_rows.append(selected_row)
            per_sample[sample_id] = selected_row

        summary_path = system_dir / "summary.jsonl"
        with summary_path.open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")

        report["systems"][system] = {
            "num_samples": len(selected_rows),
            "choices": choices,
            "summary_path": str(summary_path),
            "per_sample": per_sample,
        }
        print(f"[{system}] choices={choices}")

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved selection report: {output_json}")
    print(f"Selected model_out root: {selected_root}")


if __name__ == "__main__":
    main()
