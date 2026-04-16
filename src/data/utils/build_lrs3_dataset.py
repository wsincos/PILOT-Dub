import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_ROOT = str(PROJECT_ROOT / "data" / "dataset" / "LRS3_Dataset" / "mp4")


def parse_args():
    parser = argparse.ArgumentParser(
        description="One-shot builder for the final PILOT-Dub LRS3 dataset."
    )
    parser.add_argument("--raw_root_dir", type=str, default=DEFAULT_RAW_ROOT)
    parser.add_argument(
        "--raw_splits",
        nargs="+",
        default=["pretrain", "trainval"],
        help="Raw LRS3 splits under raw_root_dir to process into one final output folder.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--encodec_model_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "pretrained_models" / "tokenizers" / "encodec.th"),
    )
    parser.add_argument(
        "--encodec_device",
        type=str,
        default="cuda",
        help="Device for EnCodec tokenization. Set to cpu for smoke tests on busy GPUs.",
    )
    parser.add_argument(
        "--phn2num_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "pretrained_models" / "tokenizers" / "phn2num.txt"),
    )
    parser.add_argument(
        "--face_preprocess_dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "pretrained_models" / "face_preprocess" / "landmarks"),
    )
    parser.add_argument(
        "--avhubert_ckpt_path",
        type=str,
        default=str(PROJECT_ROOT / "artifacts" / "pretrained_models" / "large_lrs3_iter5.pt"),
    )
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--mega_batch_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--skip_video_origin",
        action="store_true",
        help="Do not restore video_origin or require it for target-side integrity.",
    )
    parser.add_argument(
        "--build_mfa_alignment",
        action="store_true",
        help="Also build MFA target alignment labels after the final manifests are written.",
    )
    parser.add_argument(
        "--mfa_python",
        type=str,
        default=None,
        help="Optional python executable for MFA stage when it must run in another environment.",
    )
    parser.add_argument("--mfa_cmd", type=str, default="mfa")
    parser.add_argument("--mfa_jobs", type=int, default=1)
    parser.add_argument("--clean_intermediate", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str], desc: str):
    print(f"[RUN] {desc}")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def directory_has_files(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def load_split_len_keys(split_len_path: Path):
    keys = []
    with open(split_len_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            keys.append(line.split(",", 1)[0])
    return keys


def load_manifest_entries(manifest_path: Path):
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                entries.append((parts[1], int(parts[2])))
    return entries


def write_integrity_report(output_dir: Path, required_target_dirs: list[str], check_mfa: bool):
    manifest_dir = output_dir / "manifest"
    split_len_path = output_dir / "split_len.txt"
    pair_phoneme_dir = output_dir / "phonemes"
    pair_encodec_dir = output_dir / "encodec_16khz_4codebooks"

    split_len_keys = load_split_len_keys(split_len_path)
    split_len_key_set = set(split_len_keys)
    duplicate_split_len = len(split_len_keys) - len(split_len_key_set)

    manifest_targets = {}
    missing_by_split = {}
    for split in ("train", "validation", "test"):
        entries = load_manifest_entries(manifest_dir / f"{split}.txt")
        missing = {
            "pair_phonemes": [],
            "pair_encodec": [],
        }
        targets = set()
        for pair_id, _ in entries:
            pair_file = f"{pair_id}.txt"
            speaker, _, target_utt = pair_id.split("__")
            target_stem = f"{speaker}_{target_utt}"
            targets.add(target_stem)
            if not (pair_phoneme_dir / pair_file).exists():
                missing["pair_phonemes"].append(pair_id)
            if not (pair_encodec_dir / pair_file).exists():
                missing["pair_encodec"].append(pair_id)
        manifest_targets[split] = targets
        missing_by_split[split] = missing

    target_missing_assets = {}
    for split, targets in manifest_targets.items():
        target_missing_assets[split] = {}
        for dir_name in required_target_dirs:
            asset_dir = output_dir / dir_name
            missing = sorted(
                target for target in targets if not (asset_dir / f"{target}{asset_dir_suffix(dir_name)}").exists()
            )
            target_missing_assets[split][dir_name] = missing

    missing_split_len = sorted(
        pair_id
        for split in ("train", "validation", "test")
        for pair_id, _ in load_manifest_entries(manifest_dir / f"{split}.txt")
        if f"{pair_id}.txt" not in split_len_key_set
    )

    report = {
        "duplicate_split_len_entries": duplicate_split_len,
        "missing_split_len_for_manifest_pairs": missing_split_len[:100],
        "missing_pair_files": {
            split: {key: value[:100] for key, value in payload.items()}
            for split, payload in missing_by_split.items()
        },
        "missing_target_assets": {
            split: {dir_name: values[:100] for dir_name, values in payload.items()}
            for split, payload in target_missing_assets.items()
        },
    }

    if check_mfa:
        mfa_dir = output_dir / "mfa_target_alignment" / "frame_labels_25hz"
        train_val_targets = manifest_targets["train"] | manifest_targets["validation"]
        mfa_targets = {path.stem for path in mfa_dir.glob("*.npz")}
        report["mfa_missing_for_train_val"] = sorted(train_val_targets - mfa_targets)[:100]
        report["mfa_extra_over_train_val"] = sorted(mfa_targets - train_val_targets)[:100]

    report_path = output_dir / "integrity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report, report_path


def asset_dir_suffix(dir_name: str) -> str:
    if dir_name == "lip_feature":
        return ".npy"
    if dir_name in {"video", "video_origin"}:
        return ".mp4"
    return ".txt"


def assert_clean_integrity(report: dict):
    if report["duplicate_split_len_entries"] != 0:
        raise RuntimeError("Found duplicate entries in split_len.txt")
    if report["missing_split_len_for_manifest_pairs"]:
        raise RuntimeError("Found manifest pairs missing split_len entries")
    for split_payload in report["missing_pair_files"].values():
        if any(split_payload.values()):
            raise RuntimeError("Found missing pair phoneme / encodec files referenced by manifest")
    for split_payload in report["missing_target_assets"].values():
        if any(split_payload.values()):
            raise RuntimeError("Found manifest targets missing required target-side assets")
    if report.get("mfa_missing_for_train_val"):
        raise RuntimeError("Found train/validation targets missing MFA frame labels")


def maybe_remove_intermediate(output_dir: Path):
    for dir_name in ("landmark", "video"):
        path = output_dir / dir_name
        if path.exists():
            shutil.rmtree(path)


def list_split_entries(raw_root_dir: Path, split_name: str) -> list[str]:
    split_dir = raw_root_dir / split_name
    if not split_dir.exists():
        raise FileNotFoundError(split_dir)
    entries = []
    for text_path in sorted(split_dir.glob("*/*.txt")):
        rel = text_path.relative_to(raw_root_dir).with_suffix("")
        entries.append(str(rel))
    return entries


def assert_no_single_utterance_collisions(raw_root_dir: Path, raw_splits: list[str]):
    owners = {}
    collisions = []
    for split_name in raw_splits:
        for text_path in (raw_root_dir / split_name).glob("*/*.txt"):
            stem = f"{text_path.parent.name}_{text_path.stem}"
            owner = owners.get(stem)
            if owner is None:
                owners[stem] = split_name
            elif owner != split_name:
                collisions.append((stem, owner, split_name))
    if collisions:
        preview = ", ".join(f"{stem} ({a},{b})" for stem, a, b in collisions[:20])
        raise RuntimeError(
            "Found single-utterance naming collisions across requested raw splits. "
            "The current flat output naming would overwrite files. "
            f"Examples: {preview}"
        )


def write_combined_file_list(raw_root_dir: Path, raw_splits: list[str], output_dir: Path) -> Path:
    entries = []
    for split_name in raw_splits:
        entries.extend(list_split_entries(raw_root_dir, split_name))
    file_list_path = output_dir / "combined_file.list"
    with open(file_list_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry + "\n")
    return file_list_path


def main():
    args = parse_args()
    raw_root_dir = Path(args.raw_root_dir).resolve()
    raw_splits = list(dict.fromkeys(args.raw_splits))
    default_output_name = (
        f"{'_'.join(raw_splits)}_preprocess" if len(raw_splits) == 1 else "pretrain_trainval_preprocess"
    )
    output_dir = Path(args.output_dir).resolve() if args.output_dir else raw_root_dir / default_output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{PROJECT_ROOT}:{env.get('PYTHONPATH', '')}".rstrip(":")

    assert_no_single_utterance_collisions(raw_root_dir, raw_splits)
    file_list_path = write_combined_file_list(raw_root_dir, raw_splits, output_dir)
    landmark_dir = output_dir / "landmark"
    video_dir = output_dir / "video"
    lip_feature_dir = output_dir / "lip_feature"
    video_origin_dir = output_dir / "video_origin"

    for split_name in raw_splits:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data" / "utils" / "save_wav.py"),
                "--root_dir",
                str(raw_root_dir),
                "--split_name",
                split_name,
                "--ffmpeg",
                args.ffmpeg,
            ],
            env=env,
            desc=f"Generate wav files for raw split {split_name}",
        )

    for split_name in raw_splits:
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data" / "utils" / "phonemize_lrs.py"),
                "--save_dir",
                str(output_dir),
                "--root_dir",
                str(raw_root_dir),
                "--split_name",
                split_name,
                "--encodec_model_path",
                args.encodec_model_path,
                "--encodec_device",
                args.encodec_device,
                "--phn2num_path",
                args.phn2num_path,
                "--n_workers",
                str(args.n_workers),
                "--mega_batch_size",
                str(args.mega_batch_size),
                "--batch_size",
                str(args.batch_size),
            ]
            + (["--skip_existing"] if args.skip_existing else []),
            env=env,
            desc=f"Phonemize transcripts and extract EnCodec tokens for raw split {split_name}",
        )

    if not (args.skip_existing and directory_has_files(landmark_dir)):
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data" / "utils" / "detect_landmark.py"),
                "--root",
                str(raw_root_dir),
                "--landmark",
                str(landmark_dir),
                "--manifest",
                str(file_list_path),
                "--ffmpeg",
                args.ffmpeg,
                "--face_preprocess_dir",
                args.face_preprocess_dir,
            ],
            env=env,
            desc=f"Detect facial landmarks for raw splits: {', '.join(raw_splits)}",
        )
    else:
        print("[SKIP] landmark directory already populated")

    if not (args.skip_existing and directory_has_files(video_dir)):
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data" / "utils" / "align_mouth.py"),
                "--video-direc",
                str(raw_root_dir),
                "--landmark",
                str(landmark_dir),
                "--filename-path",
                str(file_list_path),
                "--save-direc",
                str(video_dir),
                "--ffmpeg",
                args.ffmpeg,
                "--face_preprocess_dir",
                args.face_preprocess_dir,
            ],
            env=env,
            desc=f"Align mouth crops for raw splits: {', '.join(raw_splits)}",
        )
    else:
        print("[SKIP] aligned lip video directory already populated")

    if not args.skip_video_origin:
        if not (args.skip_existing and directory_has_files(video_origin_dir)):
            run_command(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "src" / "data" / "utils" / "restore_original_videos.py"),
                    "--source_root_dir",
                    str(raw_root_dir),
                    "--file_list",
                    str(file_list_path),
                    "--lip_video_dir",
                    str(video_dir),
                    "--output_dir",
                    str(video_origin_dir),
                ],
                env=env,
                desc=f"Restore full original target videos for raw splits: {', '.join(raw_splits)}",
            )
        else:
            print("[SKIP] video_origin directory already populated")

    if not (args.skip_existing and directory_has_files(lip_feature_dir)):
        run_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "src" / "data" / "utils" / "extract_avhubert_features.py"),
                "--ckpt_path",
                args.avhubert_ckpt_path,
                "--input",
                str(video_dir),
                "--output_dir",
                str(lip_feature_dir),
                "--ext",
                ".mp4",
                "--device",
                args.device,
            ]
            + (["--skip_existing"] if args.skip_existing else []),
            env=env,
            desc=f"Extract AV-HuBERT lip features for raw splits: {', '.join(raw_splits)}",
        )
    else:
        print("[SKIP] lip_feature directory already populated")

    required_target_dirs = ["lip_feature"]
    if not args.skip_video_origin:
        required_target_dirs.append("video_origin")
    run_command(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "data" / "utils" / "construct_dataset.py"),
            "--root_dir",
            str(output_dir),
            "--seed",
            str(args.seed),
            "--train_ratio",
            str(args.train_ratio),
            "--val_ratio",
            str(args.val_ratio),
            "--overwrite",
            "--required_target_dirs",
            *required_target_dirs,
        ],
        env=env,
        desc="Construct final pair dataset and manifests",
    )

    if args.build_mfa_alignment:
        mfa_python = args.mfa_python or sys.executable
        run_command(
            [
                mfa_python,
                str(PROJECT_ROOT / "src" / "data" / "utils" / "generate_mfa_alignment_labels.py"),
                "--raw_root_dir",
                str(raw_root_dir),
                "--preprocess_dir",
                str(output_dir),
                "--output_dir",
                str(output_dir / "mfa_target_alignment"),
                "--splits",
                "train",
                "validation",
                "--mfa_cmd",
                args.mfa_cmd,
                "--jobs",
                str(args.mfa_jobs),
                "--skip_existing",
            ],
            env=env,
            desc="Build MFA target-side alignments",
        )

    report, report_path = write_integrity_report(
        output_dir=output_dir,
        required_target_dirs=required_target_dirs,
        check_mfa=args.build_mfa_alignment,
    )
    print(f"Integrity report written to {report_path}")
    assert_clean_integrity(report)

    if args.clean_intermediate:
        maybe_remove_intermediate(output_dir)

    print(f"Done. Final dataset is ready at {output_dir}")


if __name__ == "__main__":
    main()
