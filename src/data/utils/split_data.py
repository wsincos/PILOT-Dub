import argparse
from pathlib import Path

from src.data.utils.construct_dataset import split_existing_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Dataset root directory containing the manifest folder",
    )
    parser.add_argument(
        "--source_manifest_name",
        type=str,
        default="train_origin.txt",
        help="Manifest filename inside manifest/ used as the split source.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    train_lines, val_lines, test_lines = split_existing_manifest(
        root_dir=Path(args.root_dir),
        source_manifest_name=args.source_manifest_name,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(f"Saved train.txt: {len(train_lines)} samples")
    print(f"Saved validation.txt: {len(val_lines)} samples")
    print(f"Saved test.txt: {len(test_lines)} samples")


if __name__ == "__main__":
    main()
