#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/jinyu_wang/projects/PILOT-Dub"
cd "${ROOT_DIR}"

GPU_ID="${1:-0}"
OUT_DIR="${ROOT_DIR}/artifacts/PILOT-Dub/avsync_scorer_training"
RESUME_CKPT="${OUT_DIR}/resume_step=5.pt"
WANDB_ID="edq0qy5e"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "resume checkpoint not found: ${RESUME_CKPT}" >&2
  exit 1
fi
if [[ "${WANDB_ID}" == "TO_BE_FILLED" ]]; then
  echo "WANDB_ID is not filled in scripts/run_wavlm_avsync_scorer_resume.sh" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vcdub

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python scripts/train_wavlm_avsync_scorer.py \
  --device cuda:0 \
  --output-dir "${OUT_DIR}" \
  --resume "${RESUME_CKPT}" \
  --train-limit 20000 \
  --val-limit 2000 \
  --window-frames 48 \
  --batch-size 8 \
  --num-workers 6 \
  --max-steps 6000 \
  --save-every 500 \
  --val-every 500 \
  --wandb \
  --wandb-project PILOT-Dub \
  --wandb-name wavlm-avsync-scorer-rerank2 \
  --wandb-id "${WANDB_ID}" \
  --wandb-id-file "${OUT_DIR}/wandb_id.txt"
