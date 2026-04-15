import os
import random
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Dataset root directory containing the 'manifest' folder")
    args = parser.parse_args()

    manifest_dir = os.path.join(args.root_dir, "manifest")
    source_file = os.path.join(manifest_dir, "train_origin.txt")

    if not os.path.exists(source_file):
        print(f"Error: {source_file} not found. Please run construct_dataset.py first.")
        return

    print(f"Reading manifest from: {source_file}")
    with open(source_file, 'r') as f:
        lines = [line for line in f.readlines() if line.strip()]
    
    total_lines = len(lines)
    print(f"Total samples: {total_lines}")

    random.seed(42)
    random.shuffle(lines)
    if total_lines < 10:
        print("Warning: Dataset is too small (<10). Copying full dataset to validation and test to avoid errors.")
        train_lines = lines
        val_lines = lines
        test_lines = lines
    else:
        # 90% / 5% / 5%
        train_end = int(total_lines * 0.90)
        val_end = train_end + int(total_lines * 0.05)

        train_lines = lines[:train_end]
        val_lines = lines[train_end:val_end]
        test_lines = lines[val_end:]

        if len(val_lines) == 0 and len(train_lines) > 1:
            val_lines = [train_lines.pop()]
        if len(test_lines) == 0 and len(train_lines) > 1:
            test_lines = [train_lines.pop()]

    def write_file(filename, data):
        path = os.path.join(manifest_dir, filename)
        with open(path, 'w') as f:
            f.writelines(data)
        print(f"Saved {filename}: {len(data)} samples")

    write_file("train.txt", train_lines)
    write_file("validation.txt", val_lines)
    write_file("test.txt", test_lines)

if __name__ == "__main__":
    main()