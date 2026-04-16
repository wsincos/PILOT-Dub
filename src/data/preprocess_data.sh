#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

RAW_ROOT_DIR="${PROJECT_DIR}/data/dataset/LRS3_Dataset/mp4"
OUTPUT_DIR="${RAW_ROOT_DIR}/pretrain_trainval_preprocess"

python src/data/utils/build_lrs3_dataset.py \
  --raw_root_dir "$RAW_ROOT_DIR" \
  --raw_splits pretrain trainval \
  --output_dir "$OUTPUT_DIR"

# To also build MFA target alignments in one shot, rerun with:
# python src/data/utils/build_lrs3_dataset.py \
#   --raw_root_dir "$RAW_ROOT_DIR" \
#   --raw_splits pretrain trainval \
#   --output_dir "$OUTPUT_DIR" \
#   --build_mfa_alignment \
#   --mfa_python "$(which python)"
