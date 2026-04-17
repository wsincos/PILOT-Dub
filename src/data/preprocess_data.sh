#!/bin/bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Please initialize conda before running." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vcdub

RAW_ROOT_DIR="${PROJECT_DIR}/data/dataset/LRS3_Dataset/mp4"
OUTPUT_DIR="${RAW_ROOT_DIR}/pretrain_trainval_preprocess"

python src/data/utils/build_lrs3_dataset.py \
  --raw_root_dir "$RAW_ROOT_DIR" \
  --raw_splits pretrain trainval \
  --output_dir "$OUTPUT_DIR" \
  --skip_existing

# To also build MFA target alignments in one shot, rerun with:
# python src/data/utils/build_lrs3_dataset.py \
#   --raw_root_dir "$RAW_ROOT_DIR" \
#   --raw_splits pretrain trainval \
#   --output_dir "$OUTPUT_DIR" \
#   --build_mfa_alignment \
#   --mfa_python "$(which python)"
