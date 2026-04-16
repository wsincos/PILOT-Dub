#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-v19_strong_real_time_acoustic_interface_formal}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/formal/pilot_dub_v19_strong_run6}"
BOOTSTRAP_RUN_DIR="${BOOTSTRAP_RUN_DIR:-${RUN_ROOT}/bootstrap}"
BOOTSTRAP_STDOUT="${BOOTSTRAP_STDOUT:-${BOOTSTRAP_RUN_DIR}/launcher.log}"

CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/resume_steps}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT_ROOT}/artifacts/PILOT-Dub-strong/v0}"
WANDB_ID_FILE="${WANDB_ID_FILE:-${RUN_ROOT}/wandb_id.txt}"
SOURCE_CKPT="${SOURCE_CKPT:-${PROJECT_ROOT}/artifacts/PILOT-Dub/generator_epoch00.ckpt}"

VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2,3,4,5}"
NUM_DEVICES="${NUM_DEVICES:-6}"

RUNS="${RUNS:-100}"
SLEEP_SEC="${SLEEP_SEC:-15}"
TRAIN_MAX_TOKENS="${TRAIN_MAX_TOKENS:-1200}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-1200}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
STEP_CKPT_INTERVAL="${STEP_CKPT_INTERVAL:-5}"

mkdir -p "${RUN_ROOT}" "${BOOTSTRAP_RUN_DIR}" "${CKPT_DIR}" "${ARCHIVE_ROOT}"

if [[ ! -f "${SOURCE_CKPT}" ]]; then
  echo "[pilot_dub_strong_bootstrap] missing source checkpoint: ${SOURCE_CKPT}" >&2
  exit 1
fi

generate_wandb_id() {
  python - <<'PY'
import random
import string
alphabet = string.ascii_lowercase + string.digits
print(''.join(random.choice(alphabet) for _ in range(8)))
PY
}

archive_resume_steps() {
  shopt -s nullglob
  local files=("${CKPT_DIR}"/*)
  if (( ${#files[@]} == 0 )); then
    return 0
  fi
  local archive_dir="${ARCHIVE_ROOT}/resume_steps_archive_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${archive_dir}"
  mv "${CKPT_DIR}"/* "${archive_dir}/"
  echo "[pilot_dub_strong_bootstrap] archived previous resume_steps to ${archive_dir}"
}

cd "${PROJECT_ROOT}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vcdub
export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

WANDB_ID="${WANDB_ID:-}"
if [[ -f "${WANDB_ID_FILE}" ]]; then
  WANDB_ID="$(<"${WANDB_ID_FILE}")"
elif [[ -z "${WANDB_ID}" ]]; then
  WANDB_ID="$(generate_wandb_id)"
  printf '%s\n' "${WANDB_ID}" > "${WANDB_ID_FILE}"
fi

archive_resume_steps

CMD=(
  bash "${PROJECT_ROOT}/scripts/loop_train.sh"
  --config-name "${CONFIG_NAME}"
  hydra.run.dir="${BOOTSTRAP_RUN_DIR}"
  model_type=pilot-dub-strong-bootstrap
  ckpt_dir=null
  load_original_model_from="${SOURCE_CKPT}"
  trainer.devices="${NUM_DEVICES}"
  trainer.accelerator=gpu
  trainer.strategy=ddp_find_unused_parameters_true
  trainer.accumulate_grad_batches="${GRAD_ACCUM}"
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  dataloader.max_num_tokens="${TRAIN_MAX_TOKENS}"
  dataloader.val_max_num_tokens="${VAL_MAX_TOKENS}"
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${STEP_CKPT_INTERVAL}"
  +logger.wandb.id="${WANDB_ID}"
  +logger.wandb.resume=allow
)

echo "[pilot_dub_strong_bootstrap] wandb id: ${WANDB_ID}"
echo "[pilot_dub_strong_bootstrap] bootstrap outputs: ${BOOTSTRAP_RUN_DIR}"
echo "[pilot_dub_strong_bootstrap] resume steps dir: ${CKPT_DIR}"
echo "[pilot_dub_strong_bootstrap] when resume_step=5.ckpt appears, stop this job manually"

RUNS="${RUNS}" SLEEP_SEC="${SLEEP_SEC}" "${CMD[@]}" "$@" 2>&1 | tee "${BOOTSTRAP_STDOUT}"
