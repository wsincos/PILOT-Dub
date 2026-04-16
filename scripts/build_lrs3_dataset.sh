#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEFAULT_LRS3_ROOT_DIR="${PROJECT_DIR}/data/dataset/LRS3_Dataset/mp4"

if [[ $# -gt 2 ]]; then
  echo "Usage: $0 [LRS3_ROOT_DIR] [OUTPUT_DIR]" >&2
  echo "  Default LRS3_ROOT_DIR: $DEFAULT_LRS3_ROOT_DIR" >&2
  exit 1
fi

LRS3_ROOT_DIR="${1:-$DEFAULT_LRS3_ROOT_DIR}"
LRS3_ROOT_DIR="$(realpath "$LRS3_ROOT_DIR")"
OUTPUT_DIR="${2:-${LRS3_ROOT_DIR}/pretrain_trainval_preprocess}"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"

if [[ ! -d "$LRS3_ROOT_DIR" ]]; then
  echo "Error: LRS3 root directory not found: $LRS3_ROOT_DIR" >&2
  exit 1
fi

if [[ ! -d "$LRS3_ROOT_DIR/pretrain" ]]; then
  echo "Error: Missing $LRS3_ROOT_DIR/pretrain" >&2
  exit 1
fi

if [[ ! -d "$LRS3_ROOT_DIR/trainval" ]]; then
  echo "Error: Missing $LRS3_ROOT_DIR/trainval" >&2
  exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

python src/data/utils/build_lrs3_dataset.py \
  --raw_root_dir "$LRS3_ROOT_DIR" \
  --raw_splits pretrain trainval \
  --output_dir "$OUTPUT_DIR"

echo
echo "Done."
echo "Final dataset directory: $OUTPUT_DIR"
