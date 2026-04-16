#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
METRICS_DIR="${METRICS_DIR:-${ROOT_DIR}/../metrics}"
cd "${ROOT_DIR}"

DEFAULT_SUBSET="mini50"
DEFAULT_MODEL_NAME=""
DEFAULT_CKPT_PATH=""
DEFAULT_CUDA_VISIBLE_DEVICES=""
DEFAULT_MODEL_CFG="pilot-dub/final"

SUBSET="${DEFAULT_SUBSET}"
if [[ $# -ge 1 && ( "$1" == "mini20" || "$1" == "mini50" ) ]]; then
  SUBSET="$1"
  shift
fi

MODEL_CFG="$DEFAULT_MODEL_CFG"
MODEL_NAME=""
CKPT_PATH=""
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"

if [[ $# -ge 1 && ( "$1" == glo-var/* || "$1" == pilot-dub/* ) ]]; then
  MODEL_CFG="${1}"
  MODEL_NAME="${2:-$DEFAULT_MODEL_NAME}"
  CKPT_PATH="${3:-$DEFAULT_CKPT_PATH}"
  CUDA_VISIBLE_DEVICES_VALUE="${4:-$CUDA_VISIBLE_DEVICES_VALUE}"
else
  MODEL_NAME="${1:-$DEFAULT_MODEL_NAME}"
  CKPT_PATH="${2:-$DEFAULT_CKPT_PATH}"
  CUDA_VISIBLE_DEVICES_VALUE="${3:-$CUDA_VISIBLE_DEVICES_VALUE}"
fi

if [[ -z "${MODEL_NAME}" || -z "${CKPT_PATH}" ]]; then
  cat >&2 <<'EOF'
Usage:
  run_lrs3_metrics_mini.sh [mini20|mini50] [model_cfg] <model_name> <ckpt_path> [cuda_visible_devices]
Examples:
  run_lrs3_metrics_mini.sh mini50 exp_name /path/to/model.ckpt 0
  run_lrs3_metrics_mini.sh mini20 pilot-dub/final exp_name /path/to/model.ckpt 0
EOF
  exit 1
fi

USE_CPU=false
if [[ "${CUDA_VISIBLE_DEVICES_VALUE}" =~ ^-[0-9]+$ ]]; then
  USE_CPU=true
fi

if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" && "${USE_CPU}" == false ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
fi

EVAL_ARGS=()
METRICS_ARGS=()
if [[ "${USE_CPU}" == true ]]; then
  export CUDA_VISIBLE_DEVICES=""
  echo "Using CPU mode because gpu id '${CUDA_VISIBLE_DEVICES_VALUE}' is negative."
  EVAL_ARGS+=(device=cpu)
  METRICS_ARGS+=(device=cpu)
elif [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
  EVAL_ARGS+=(device="${CUDA_VISIBLE_DEVICES_VALUE}")
  METRICS_ARGS+=(device=cuda gpu_id="${CUDA_VISIBLE_DEVICES_VALUE}")
fi

DATASET_TAG="LRS3-${SUBSET}"
EVAL_CONFIG="evaluate/LRS3_Test_${SUBSET}"
METRICS_CONFIG="evaluate-LRS3-${SUBSET}"
RESULT_ROOT="${METRICS_DIR}/model_out/${DATASET_TAG}/${MODEL_NAME}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Please initialize conda before running." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

MODEL_NAME_ESCAPED="${MODEL_NAME//=/\\=}"
CKPT_PATH_ESCAPED="${CKPT_PATH//=/\\=}"
MODEL_CFG_ESCAPED="${MODEL_CFG//=/\\=}"

echo "[0/3] Ensuring mini eval lists exist..."
conda activate vcdub
python scripts/build_mini_eval_sets.py --datasets lrs3

echo "[1/3] Running evaluate_npy.py in vcdub env..."
python scripts/evaluate_npy.py \
  --config-name "${EVAL_CONFIG}" \
  result_root="${RESULT_ROOT}" \
  ckpt_path="${CKPT_PATH_ESCAPED}" \
  "model=${MODEL_CFG_ESCAPED}" \
  "${EVAL_ARGS[@]}"

echo "[2/3] Running merge.py in metrics env..."
cd "${METRICS_DIR}"
conda activate metrics
python merge.py --gen_dir="${RESULT_ROOT}"

echo "[3/3] Running metrics evaluate.py in metrics env..."
python evaluate.py \
  --config-name "${METRICS_CONFIG}" \
  model_name="${MODEL_NAME_ESCAPED}" \
  "${METRICS_ARGS[@]}"

echo "Done."
