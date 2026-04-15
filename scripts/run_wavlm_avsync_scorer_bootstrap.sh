#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/jinyu_wang/projects/PILOT-Dub"
cd "${ROOT_DIR}"

GPU_ID="${1:-0}"
OUT_DIR="${ROOT_DIR}/artifacts/PILOT-Dub/avsync_scorer_training"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vcdub

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${OUT_DIR}"

python scripts/train_wavlm_avsync_scorer.py \
  --device cuda:0 \
  --output-dir "${OUT_DIR}" \
  --train-limit 2000 \
  --val-limit 200 \
  --window-frames 48 \
  --batch-size 4 \
  --num-workers 4 \
  --max-steps 5 \
  --save-every 5 \
  --val-every 5 \
  --wandb \
  --wandb-project PILOT-Dub \
  --wandb-name wavlm-avsync-scorer-rerank2 \
  --wandb-id-file "${OUT_DIR}/wandb_id.txt"
