#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/jinyu_wang/projects/PILOT-Dub"
METRICS_DIR="/data1/jinyu_wang/projects/metrics"
cd "${ROOT_DIR}"

if [[ $# -lt 8 ]]; then
  cat >&2 <<'EOF'
Usage:
  run_candidate_metrics_mini.sh <mini20|mini50> <model_cfg> <model_name> <ckpt_path> <gpu_id> <seed> <top_p> <temperature> [top_k]
EOF
  exit 1
fi

SUBSET="$1"
MODEL_CFG="$2"
MODEL_NAME="$3"
CKPT_PATH="$4"
GPU_ID="$5"
SEED="$6"
TOP_P="$7"
TEMPERATURE="$8"
TOP_K="${9:-0}"

if [[ "${SUBSET}" != "mini20" && "${SUBSET}" != "mini50" ]]; then
  echo "subset must be mini20 or mini50: ${SUBSET}" >&2
  exit 1
fi
if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please initialize conda before running." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

DATASET_TAG="LRS3-${SUBSET}"
EVAL_CONFIG="evaluate/LRS3_Test_${SUBSET}"
METRICS_CONFIG="evaluate-LRS3-${SUBSET}"
RESULT_ROOT="${METRICS_DIR}/model_out/${DATASET_TAG}/${MODEL_NAME}"

MODEL_NAME_ESCAPED="${MODEL_NAME//=/\\=}"
CKPT_PATH_ESCAPED="${CKPT_PATH//=/\\=}"
MODEL_CFG_ESCAPED="${MODEL_CFG//=/\\=}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

echo "[0/3] Ensuring mini eval lists exist..."
conda activate vcdub
python scripts/build_mini_eval_sets.py --datasets lrs3

echo "[1/3] Generating ${MODEL_NAME} seed=${SEED} top_p=${TOP_P} temperature=${TEMPERATURE} top_k=${TOP_K}"
python scripts/evaluate_npy.py \
  --config-name "${EVAL_CONFIG}" \
  result_root="${RESULT_ROOT}" \
  ckpt_path="${CKPT_PATH_ESCAPED}" \
  "model=${MODEL_CFG_ESCAPED}" \
  device="${GPU_ID}" \
  seed="${SEED}" \
  top_p="${TOP_P}" \
  temperature="${TEMPERATURE}" \
  top_k="${TOP_K}" \
  skip_existing=false

echo "[2/3] Merging ${MODEL_NAME}"
cd "${METRICS_DIR}"
conda activate metrics
python merge.py --gen_dir="${RESULT_ROOT}"

echo "[3/3] Metrics ${MODEL_NAME}"
python evaluate.py \
  --config-name "${METRICS_CONFIG}" \
  model_name="${MODEL_NAME_ESCAPED}" \
  device=cuda \
  gpu_id="${GPU_ID}"

echo "Done ${MODEL_NAME}"
