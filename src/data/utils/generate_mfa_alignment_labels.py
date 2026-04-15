import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


DEFAULT_SILENCE_LABELS = {"sil", "sp", "spn", "<eps>", "silence"}


@dataclass(frozen=True)
class UtteranceKey:
    speaker: str
    utt: str

    @property
    def stem(self) -> str:
        return f"{self.speaker}_{self.utt}"


@dataclass
class PhoneInterval:
    begin: float
    end: float
    label: str
    speaker: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MFA target-side corpus and build frame-level phoneme alignment labels."
    )
    parser.add_argument(
        "--raw_root_dir",
        type=str,
        default="/data1/jinyu_wang/datasets/LRS3_Dataset/mp4",
        help="Root of the raw LRS3 dataset that still contains trainval/<speaker>/<utt>.mp4 and .txt.",
    )
    parser.add_argument(
        "--preprocess_dir",
        type=str,
        default="/data1/jinyu_wang/datasets/LRS3_Dataset/mp4/trainval_preprocess",
        help="Existing preprocess directory containing manifest/ and encodec_16khz_4codebooks_.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Manifest splits to scan for pair ids.",
    )
    parser.add_argument(
        "--manifest_name",
        type=str,
        default="manifest",
        help="Manifest folder name inside preprocess_dir.",
    )
    parser.add_argument(
        "--encodec_folder_name",
        type=str,
        default="encodec_16khz_4codebooks_",
        help="Single-utterance encodec folder used to derive exact frame counts.",
    )
    parser.add_argument(
        "--frame_hz",
        type=float,
        default=25.0,
        help="Target frame rate for frame-level alignment labels.",
    )
    parser.add_argument(
        "--encodec_code_sr",
        type=float,
        default=50.0,
        help="Encodec code rate. Used to map single-utterance encodec lengths to frame counts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output root. Defaults to <preprocess_dir>/mfa_target_alignment.",
    )
    parser.add_argument(
        "--mfa_cmd",
        type=str,
        default="mfa",
        help="MFA executable.",
    )
    parser.add_argument(
        "--mfa_dictionary",
        type=str,
        default="english_us_arpa",
        help="MFA dictionary name/path.",
    )
    parser.add_argument(
        "--mfa_acoustic_model",
        type=str,
        default="english_us_arpa",
        help="MFA acoustic model name/path.",
    )
    parser.add_argument(
        "--beam",
        type=int,
        default=50,
        help="MFA beam size.",
    )
    parser.add_argument(
        "--retry_beam",
        type=int,
        default=200,
        help="MFA retry beam size.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="MFA worker count.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Reuse existing wav/txt/csv/frame-label outputs when possible.",
    )
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Only prepare the MFA corpus; do not call MFA or build labels.",
    )
    parser.add_argument(
        "--max_utts",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help="Ignore index used for unlabeled/silence frames in frame-level CE.",
    )
    parser.add_argument(
        "--silence_labels",
        nargs="*",
        default=sorted(DEFAULT_SILENCE_LABELS),
        help="Phone labels that should become ignore frames instead of supervised labels.",
    )
    return parser.parse_args()


def load_manifest_targets(preprocess_dir: Path, manifest_name: str, splits: Sequence[str]) -> List[UtteranceKey]:
    targets: Dict[str, UtteranceKey] = {}
    for split in splits:
        manifest_path = preprocess_dir / manifest_name / f"{split}.txt"
        with open(manifest_path, "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                pair_id = parts[1]
                speaker, _, target_utt = pair_id.split("__")
                key = UtteranceKey(speaker=speaker, utt=target_utt)
                targets[key.stem] = key
    return sorted(targets.values(), key=lambda item: item.stem)


def load_raw_transcript(raw_txt_path: Path) -> str:
    with open(raw_txt_path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    if "Text:" not in first_line:
        raise ValueError(f"Unexpected transcript format in {raw_txt_path}")
    return first_line.split("Text:", 1)[1].strip()


def load_single_encodec_length(preprocess_dir: Path, encodec_folder_name: str, utt_key: UtteranceKey) -> int:
    encodec_path = preprocess_dir / encodec_folder_name / f"{utt_key.stem}.txt"
    with open(encodec_path, "r", encoding="utf-8") as handle:
        first_line = handle.readline().strip()
    return len(first_line.split())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_wav_from_mp4(mp4_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_mfa_corpus(
    raw_root_dir: Path,
    preprocess_dir: Path,
    encodec_folder_name: str,
    utterances: Sequence[UtteranceKey],
    corpus_dir: Path,
    skip_existing: bool,
) -> Dict[str, Dict[str, object]]:
    ensure_dir(corpus_dir)
    metadata: Dict[str, Dict[str, object]] = {}
    for utt_key in utterances:
        raw_dir = raw_root_dir / "trainval" / utt_key.speaker
        raw_txt = raw_dir / f"{utt_key.utt}.txt"
        raw_mp4 = raw_dir / f"{utt_key.utt}.mp4"
        if not raw_txt.exists():
            raise FileNotFoundError(raw_txt)
        if not raw_mp4.exists():
            raise FileNotFoundError(raw_mp4)

        transcript = load_raw_transcript(raw_txt)
        wav_path = corpus_dir / f"{utt_key.stem}.wav"
        txt_path = corpus_dir / f"{utt_key.stem}.txt"

        if not (skip_existing and wav_path.exists()):
            extract_wav_from_mp4(raw_mp4, wav_path)
        if not (skip_existing and txt_path.exists()):
            txt_path.write_text(transcript, encoding="utf-8")

        encodec_len = load_single_encodec_length(preprocess_dir, encodec_folder_name, utt_key)
        metadata[utt_key.stem] = {
            "speaker": utt_key.speaker,
            "utt": utt_key.utt,
            "raw_mp4": str(raw_mp4),
            "raw_txt": str(raw_txt),
            "wav_path": str(wav_path),
            "txt_path": str(txt_path),
            "transcript": transcript,
            "encodec_len": int(encodec_len),
        }
    return metadata


def resolve_mfa_binary(mfa_cmd: str) -> Optional[str]:
    return shutil.which(mfa_cmd)


def run_mfa_align(
    mfa_binary: str,
    corpus_dir: Path,
    mfa_output_dir: Path,
    dictionary: str,
    acoustic_model: str,
    beam: int,
    retry_beam: int,
    jobs: int,
) -> None:
    ensure_dir(mfa_output_dir)
    cmd = [
        mfa_binary,
        "align",
        "-v",
        "--clean",
        "-j",
        str(jobs),
        "--output_format",
        "csv",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(mfa_output_dir),
        "--beam",
        str(beam),
        "--retry_beam",
        str(retry_beam),
    ]
    subprocess.run(cmd, check=True)


def parse_mfa_phone_csv(csv_path: Path) -> List[PhoneInterval]:
    intervals: List[PhoneInterval] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_type = row.get("Type") or row.get("type")
            if row_type != "phones":
                continue
            intervals.append(
                PhoneInterval(
                    begin=float(row["Begin"]),
                    end=float(row["End"]),
                    label=row["Label"].strip(),
                    speaker=(row.get("Speaker") or "").strip(),
                )
            )
    return intervals


def build_phone_vocab(csv_paths: Iterable[Path], silence_labels: set[str]) -> Dict[str, int]:
    phones = set()
    for csv_path in csv_paths:
        for interval in parse_mfa_phone_csv(csv_path):
            label_lower = interval.label.lower()
            if label_lower in silence_labels:
                continue
            phones.add(interval.label)
    return {label: idx for idx, label in enumerate(sorted(phones))}


def save_phone_vocab(vocab_path: Path, phone_vocab: Dict[str, int]) -> None:
    with open(vocab_path, "w", encoding="utf-8") as handle:
        for label, idx in sorted(phone_vocab.items(), key=lambda item: item[1]):
            handle.write(f"{idx} {label}\n")


def compute_frame_count(encodec_len: int, encodec_code_sr: float, frame_hz: float) -> int:
    ratio = encodec_code_sr / frame_hz
    return max(1, int(round(encodec_len / ratio)))


def build_frame_labels(
    intervals: Sequence[PhoneInterval],
    frame_count: int,
    frame_hz: float,
    phone_vocab: Dict[str, int],
    silence_labels: set[str],
    ignore_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.full((frame_count,), ignore_index, dtype=np.int32)
    valid_mask = np.zeros((frame_count,), dtype=np.uint8)

    for interval in intervals:
        label_lower = interval.label.lower()
        begin_idx = max(0, int(math.floor(interval.begin * frame_hz)))
        end_idx = max(begin_idx + 1, int(math.ceil(interval.end * frame_hz)))
        end_idx = min(frame_count, end_idx)
        if begin_idx >= frame_count or end_idx <= 0:
            continue

        if label_lower in silence_labels:
            continue
        if interval.label not in phone_vocab:
            continue
        labels[begin_idx:end_idx] = phone_vocab[interval.label]
        valid_mask[begin_idx:end_idx] = 1

    return labels, valid_mask


def save_phone_intervals(json_path: Path, intervals: Sequence[PhoneInterval]) -> None:
    payload = [
        {
            "begin": interval.begin,
            "end": interval.end,
            "label": interval.label,
            "speaker": interval.speaker,
        }
        for interval in intervals
    ]
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    raw_root_dir = Path(args.raw_root_dir)
    preprocess_dir = Path(args.preprocess_dir)
    output_dir = Path(args.output_dir) if args.output_dir else preprocess_dir / "mfa_target_alignment"
    corpus_dir = output_dir / "mfa_corpus"
    csv_dir = output_dir / "mfa_csv"
    intervals_dir = output_dir / "phone_intervals"
    frame_labels_dir = output_dir / f"frame_labels_{int(args.frame_hz)}hz"
    metadata_dir = output_dir / "metadata"
    vocab_path = output_dir / "align_phn2num.txt"

    ensure_dir(output_dir)
    ensure_dir(csv_dir)
    ensure_dir(intervals_dir)
    ensure_dir(frame_labels_dir)
    ensure_dir(metadata_dir)

    utterances = load_manifest_targets(preprocess_dir, args.manifest_name, args.splits)
    if args.max_utts is not None:
        utterances = utterances[: args.max_utts]

    metadata = prepare_mfa_corpus(
        raw_root_dir=raw_root_dir,
        preprocess_dir=preprocess_dir,
        encodec_folder_name=args.encodec_folder_name,
        utterances=utterances,
        corpus_dir=corpus_dir,
        skip_existing=args.skip_existing,
    )

    metadata_path = metadata_dir / "prepared_targets.json"
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    if args.prepare_only:
        print(f"Prepared MFA corpus for {len(utterances)} target utterances at {corpus_dir}")
        print(f"Metadata written to {metadata_path}")
        return

    mfa_binary = resolve_mfa_binary(args.mfa_cmd)
    if mfa_binary is None:
        raise RuntimeError(
            "MFA executable not found. Install Montreal Forced Aligner or rerun with --prepare_only."
        )

    if not (args.skip_existing and any(csv_dir.glob("*.csv"))):
        run_mfa_align(
            mfa_binary=mfa_binary,
            corpus_dir=corpus_dir,
            mfa_output_dir=csv_dir,
            dictionary=args.mfa_dictionary,
            acoustic_model=args.mfa_acoustic_model,
            beam=args.beam,
            retry_beam=args.retry_beam,
            jobs=args.jobs,
        )

    csv_paths = []
    for utt_key in utterances:
        csv_path = csv_dir / f"{utt_key.stem}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing MFA csv for {utt_key.stem}: {csv_path}")
        csv_paths.append(csv_path)

    silence_labels = {label.lower() for label in args.silence_labels}
    phone_vocab = build_phone_vocab(csv_paths, silence_labels=silence_labels)
    save_phone_vocab(vocab_path, phone_vocab)

    summary = defaultdict(int)
    for utt_key in utterances:
        stem = utt_key.stem
        csv_path = csv_dir / f"{stem}.csv"
        intervals = parse_mfa_phone_csv(csv_path)
        save_phone_intervals(intervals_dir / f"{stem}.json", intervals)

        encodec_len = int(metadata[stem]["encodec_len"])
        frame_count = compute_frame_count(
            encodec_len=encodec_len,
            encodec_code_sr=args.encodec_code_sr,
            frame_hz=args.frame_hz,
        )
        labels, valid_mask = build_frame_labels(
            intervals=intervals,
            frame_count=frame_count,
            frame_hz=args.frame_hz,
            phone_vocab=phone_vocab,
            silence_labels=silence_labels,
            ignore_index=args.ignore_index,
        )
        np.savez_compressed(
            frame_labels_dir / f"{stem}.npz",
            labels=labels,
            valid_mask=valid_mask,
            frame_count=np.int32(frame_count),
            encodec_len=np.int32(encodec_len),
        )
        summary["utterances"] += 1
        summary["frames"] += int(frame_count)
        summary["valid_frames"] += int(valid_mask.sum())

    summary_path = metadata_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "num_utterances": summary["utterances"],
                "num_phone_vocab": len(phone_vocab),
                "num_frames": summary["frames"],
                "num_valid_frames": summary["valid_frames"],
                "valid_frame_ratio": float(summary["valid_frames"] / max(summary["frames"], 1)),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Prepared and aligned {summary['utterances']} utterances.")
    print(f"MFA csv dir: {csv_dir}")
    print(f"Frame-label dir: {frame_labels_dir}")
    print(f"Alignment vocab: {vocab_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
