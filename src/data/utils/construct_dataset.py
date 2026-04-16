import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm


def count_tokens_in_line(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def count_tokens_in_file(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return count_tokens_in_line(f.readline())


def read_single_line(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.readline().strip()


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_lines(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def split_manifest_entries(
    lines: list[str],
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
) -> tuple[list[str], list[str], list[str]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must satisfy 0 < train_ratio and train_ratio + val_ratio < 1")

    shuffled = list(lines)
    random.Random(seed).shuffle(shuffled)

    total_lines = len(shuffled)
    if total_lines < 10:
        return shuffled, shuffled, shuffled

    train_end = int(total_lines * train_ratio)
    val_end = train_end + int(total_lines * val_ratio)
    train_lines = shuffled[:train_end]
    val_lines = shuffled[train_end:val_end]
    test_lines = shuffled[val_end:]

    if len(val_lines) == 0 and len(train_lines) > 1:
        val_lines = [train_lines.pop()]
    if len(test_lines) == 0 and len(train_lines) > 1:
        test_lines = [train_lines.pop()]

    return train_lines, val_lines, test_lines


def split_existing_manifest(
    root_dir: Path,
    source_manifest_name: str = "train_origin.txt",
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
):
    manifest_dir = root_dir / "manifest"
    source_file = manifest_dir / source_manifest_name
    if not source_file.exists():
        raise FileNotFoundError(source_file)

    with open(source_file, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    train_lines, val_lines, test_lines = split_manifest_entries(
        lines=lines,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    write_lines(manifest_dir / "train.txt", train_lines)
    write_lines(manifest_dir / "validation.txt", val_lines)
    write_lines(manifest_dir / "test.txt", test_lines)
    return train_lines, val_lines, test_lines


def load_single_utterances(root_dir: Path):
    phoneme_dir = root_dir / "phonemes_"
    encodec_dir = root_dir / "encodec_16khz_4codebooks_"
    if not phoneme_dir.exists():
        raise FileNotFoundError(phoneme_dir)
    if not encodec_dir.exists():
        raise FileNotFoundError(encodec_dir)

    phoneme_files = {path.stem: path for path in phoneme_dir.glob("*.txt")}
    encodec_files = {path.stem: path for path in encodec_dir.glob("*.txt")}
    common_stems = sorted(phoneme_files.keys() & encodec_files.keys())
    if not common_stems:
        raise RuntimeError("No common single-utterance phoneme / encodec files were found.")

    speaker_utts: dict[str, list[str]] = defaultdict(list)
    for stem in common_stems:
        speaker, utt = stem.rsplit("_", 1)
        speaker_utts[speaker].append(utt)

    for utts in speaker_utts.values():
        utts.sort()

    return phoneme_files, encodec_files, speaker_utts


def collect_available_target_stems(root_dir: Path, required_target_dirs: list[str]) -> dict[str, set[str]]:
    available_by_dir: dict[str, set[str]] = {}
    for dir_name in required_target_dirs:
        dir_path = root_dir / dir_name
        if not dir_path.exists():
            raise FileNotFoundError(dir_path)
        available_by_dir[dir_name] = {path.stem for path in dir_path.iterdir() if path.is_file()}
    return available_by_dir


def construct_dataset(
    root_dir: Path,
    required_target_dirs: list[str],
    seed: int = 42,
    train_ratio: float = 0.90,
    val_ratio: float = 0.05,
    overwrite: bool = False,
):
    phoneme_files, encodec_files, speaker_utts = load_single_utterances(root_dir)
    available_by_dir = collect_available_target_stems(root_dir, required_target_dirs)

    pair_phoneme_dir = root_dir / "phonemes"
    pair_encodec_dir = root_dir / "encodec_16khz_4codebooks"
    manifest_dir = root_dir / "manifest"
    split_len_path = root_dir / "split_len.txt"
    report_path = root_dir / "dataset_build_report.json"

    ensure_clean_dir(pair_phoneme_dir, overwrite=overwrite)
    ensure_clean_dir(pair_encodec_dir, overwrite=overwrite)
    ensure_clean_dir(manifest_dir, overwrite=overwrite)
    if split_len_path.exists() and overwrite:
        split_len_path.unlink()
    if report_path.exists() and overwrite:
        report_path.unlink()

    manifest_lines: list[str] = []
    split_len_lines: list[str] = []
    skipped_pairs = 0

    missing_targets_by_dir = {dir_name: set() for dir_name in required_target_dirs}
    all_stems = set(phoneme_files.keys())
    for dir_name, stems in available_by_dir.items():
        missing_targets_by_dir[dir_name] = all_stems - stems

    for speaker, utts in tqdm(sorted(speaker_utts.items()), desc="Constructing pairs"):
        for ref_utt in utts:
            ref_stem = f"{speaker}_{ref_utt}"
            ref_phoneme = read_single_line(phoneme_files[ref_stem])
            ref_encodec_lines = encodec_files[ref_stem].read_text(encoding="utf-8").splitlines()
            ref_len = count_tokens_in_line(ref_encodec_lines[0])

            for target_utt in utts:
                if target_utt == ref_utt:
                    continue
                target_stem = f"{speaker}_{target_utt}"
                if any(target_stem not in stems for stems in available_by_dir.values()):
                    skipped_pairs += 1
                    continue

                target_phoneme = read_single_line(phoneme_files[target_stem])
                target_encodec_lines = encodec_files[target_stem].read_text(encoding="utf-8").splitlines()
                if len(ref_encodec_lines) != len(target_encodec_lines):
                    raise ValueError(
                        f"Codebook count mismatch between {ref_stem} and {target_stem}: "
                        f"{len(ref_encodec_lines)} vs {len(target_encodec_lines)}"
                    )

                pair_id = f"{speaker}__{ref_utt}__{target_utt}"
                merged_phoneme = f"{ref_phoneme} {target_phoneme}\n"
                merged_encodec_lines = [
                    f"{ref_line.strip()} {target_line.strip()}\n"
                    for ref_line, target_line in zip(ref_encodec_lines, target_encodec_lines)
                ]
                total_len = count_tokens_in_line(merged_encodec_lines[0])

                write_lines(pair_phoneme_dir / f"{pair_id}.txt", [merged_phoneme])
                write_lines(pair_encodec_dir / f"{pair_id}.txt", merged_encodec_lines)
                manifest_lines.append(f"0\t{pair_id}\t{total_len}\n")
                split_len_lines.append(f"{pair_id}.txt,{ref_len}\n")

    write_lines(manifest_dir / "train_origin.txt", manifest_lines)
    write_lines(split_len_path, split_len_lines)
    train_lines, val_lines, test_lines = split_existing_manifest(
        root_dir=root_dir,
        source_manifest_name="train_origin.txt",
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    report = {
        "num_single_utterances": len(all_stems),
        "num_speakers": len(speaker_utts),
        "required_target_dirs": required_target_dirs,
        "num_pairs_total": len(manifest_lines),
        "num_pairs_train": len(train_lines),
        "num_pairs_validation": len(val_lines),
        "num_pairs_test": len(test_lines),
        "num_pairs_skipped_missing_target_assets": skipped_pairs,
        "missing_target_counts_by_dir": {
            dir_name: len(stems) for dir_name, stems in missing_targets_by_dir.items()
        },
        "missing_target_samples_by_dir": {
            dir_name: sorted(stems)[:20] for dir_name, stems in missing_targets_by_dir.items()
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct pair dataset and final manifest files from single-utterance preprocess outputs."
    )
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument(
        "--required_target_dirs",
        nargs="*",
        default=["lip_feature"],
        help="Directories under root_dir that must contain a target-side file for a pair to be emitted.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true", help="Remove existing pair dataset outputs before rebuild.")
    return parser.parse_args()


def main():
    args = parse_args()
    report = construct_dataset(
        root_dir=Path(args.root_dir),
        required_target_dirs=args.required_target_dirs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
