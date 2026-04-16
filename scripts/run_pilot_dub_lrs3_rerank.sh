#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
METRICS_DIR="${METRICS_DIR:-${ROOT_DIR}/../metrics}"
cd "${ROOT_DIR}"

SUBSET="${1:-mini50}"       # mini20 | mini50 | full
EXP_NAME="${2:-pilotdub-rerank}"
GEN_CKPT="${3:-${ROOT_DIR}/artifacts/PILOT-Dub/generator_epoch00.ckpt}"
GPU_ID="${4:-0}"
SCORER_CKPT="${5:-${ROOT_DIR}/artifacts/PILOT-Dub/avsync_scorer_best.pt}"
GEN_MODEL_CFG="${6:-pilot-dub/final}"

case "${SUBSET}" in
  mini20)
    DATASET_TAG="LRS3-mini20"
    EVAL_CONFIG="evaluate/LRS3_Test_mini20"
    METRICS_CONFIG="evaluate-LRS3-mini20"
    ;;
  mini50)
    DATASET_TAG="LRS3-mini50"
    EVAL_CONFIG="evaluate/LRS3_Test_mini50"
    METRICS_CONFIG="evaluate-LRS3-mini50"
    ;;
  full|LRS3)
    DATASET_TAG="LRS3"
    EVAL_CONFIG="evaluate/LRS3_Test_True"
    METRICS_CONFIG=""
    ;;
  *)
    echo "Unknown subset: ${SUBSET}. Expected mini20, mini50, or full." >&2
    exit 1
    ;;
esac

if [[ ! -f "${GEN_CKPT}" ]]; then
  echo "generator checkpoint not found: ${GEN_CKPT}" >&2
  exit 1
fi
if [[ ! -f "${SCORER_CKPT}" ]]; then
  echo "scorer checkpoint not found: ${SCORER_CKPT}" >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

declare -a CAND_SUFFIXES=("cand0-default" "cand1-seed2" "cand2-t09-p09")
declare -a SEEDS=(1 2 3)
declare -a TOPPS=(0.8 0.8 0.9)
declare -a TEMPS=(1.0 1.0 0.9)
declare -a TOPKS=(0 0 0)

MODEL_OUT_ROOT="${METRICS_DIR}/model_out/${DATASET_TAG}"
SELECT_REPORT="${ROOT_DIR}/reports/${EXP_NAME}_${DATASET_TAG}_selection.json"
SELECTED_ROOT="${MODEL_OUT_ROOT}/${EXP_NAME}"

GEN_CKPT_ESCAPED="${GEN_CKPT//=/\\=}"
GEN_MODEL_CFG_ESCAPED="${GEN_MODEL_CFG//=/\\=}"

echo "[0/5] Ensuring mini eval lists exist when needed..."
conda activate vcdub
if [[ "${SUBSET}" == mini20 || "${SUBSET}" == mini50 ]]; then
  python scripts/build_mini_eval_sets.py --datasets lrs3
fi

CANDIDATE_NAMES=()
for idx in "${!CAND_SUFFIXES[@]}"; do
  cand_name="${EXP_NAME}-${CAND_SUFFIXES[$idx]}"
  CANDIDATE_NAMES+=("${cand_name}")
  cand_root="${MODEL_OUT_ROOT}/${cand_name}"
  echo "[1/5] Generating candidate ${idx}: ${cand_name}"
  python scripts/evaluate_npy.py \
    --config-name "${EVAL_CONFIG}" \
    result_root="${cand_root}" \
    ckpt_path="${GEN_CKPT_ESCAPED}" \
    "model=${GEN_MODEL_CFG_ESCAPED}" \
    device="${GPU_ID}" \
    seed="${SEEDS[$idx]}" \
    top_p="${TOPPS[$idx]}" \
    temperature="${TEMPS[$idx]}" \
    top_k="${TOPKS[$idx]}" \
    skip_existing=false
done

CANDIDATE_CSV="$(IFS=,; echo "${CANDIDATE_NAMES[*]}")"

echo "[2/5] Selecting candidates with WavLM-AVSync scorer..."
python scripts/select_candidates_with_avsync_scorer.py \
  --checkpoint "${SCORER_CKPT}" \
  --dataset "${DATASET_TAG}" \
  --candidates "${CANDIDATE_CSV}" \
  --selected-model-name "${EXP_NAME}" \
  --device cuda:0 \
  --num-windows 5 \
  --output-json "${SELECT_REPORT}"

echo "[3/5] Merging selected outputs..."
cd "${METRICS_DIR}"
conda activate metrics
python merge.py --gen_dir="${SELECTED_ROOT}"

echo "[4/5] Evaluating selected PILOT-Dub outputs..."
if [[ -n "${METRICS_CONFIG}" ]]; then
  python evaluate.py --config-name "${METRICS_CONFIG}" model_name="${EXP_NAME}" device=cuda gpu_id="${GPU_ID}"
else
  python evaluate.py model_name="${EXP_NAME}" device=cuda gpu_id="${GPU_ID}"
fi

echo "[5/5] Done. Selected output: ${SELECTED_ROOT}"
echo "Selection report: ${SELECT_REPORT}"
