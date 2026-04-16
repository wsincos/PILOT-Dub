#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_NAME="${CONFIG_NAME:-v19_strong_real_time_acoustic_interface_formal}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/outputs/formal/pilot_dub_v19_strong_run6}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-${RUN_ROOT}/resume}"
RESUME_STDOUT="${RESUME_STDOUT:-${RESUME_RUN_DIR}/launcher.log}"

CKPT_DIR="${CKPT_DIR:-${PROJECT_ROOT}/resume_steps}"
WANDB_ID_FILE="${WANDB_ID_FILE:-${RUN_ROOT}/wandb_id.txt}"

VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2,3,4,5}"
NUM_DEVICES="${NUM_DEVICES:-6}"

RUNS="${RUNS:-200}"
SLEEP_SEC="${SLEEP_SEC:-15}"
TRAIN_MAX_TOKENS="${TRAIN_MAX_TOKENS:-1200}"
VAL_MAX_TOKENS="${VAL_MAX_TOKENS:-1200}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-1000}"
STEP_CKPT_INTERVAL="${STEP_CKPT_INTERVAL:-150}"

mkdir -p "${RUN_ROOT}" "${RESUME_RUN_DIR}" "${CKPT_DIR}"

latest_valid_ckpt() {
  find "${CKPT_DIR}" -maxdepth 1 -type f -name "*.ckpt" -size +0c | sort | tail -n 1 || true
}

if [[ ! -f "${WANDB_ID_FILE}" ]]; then
  echo "[pilot_dub_strong_resume] missing wandb id file: ${WANDB_ID_FILE}" >&2
  echo "[pilot_dub_strong_resume] run bootstrap script first" >&2
  exit 1
fi

LATEST_CKPT="$(latest_valid_ckpt)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "[pilot_dub_strong_resume] no checkpoint found in ${CKPT_DIR}" >&2
  echo "[pilot_dub_strong_resume] run bootstrap script first and wait for resume_step=5.ckpt" >&2
  exit 1
fi

WANDB_ID="$(<"${WANDB_ID_FILE}")"

cd "${PROJECT_ROOT}"
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vcdub
export PYTHONPATH="${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${VISIBLE_GPUS}"

CMD=(
  bash "${PROJECT_ROOT}/scripts/loop_train.sh"
  --config-name "${CONFIG_NAME}"
  hydra.run.dir="${RESUME_RUN_DIR}"
  model_type=pilot-dub-strong-resume
  ckpt_dir="${CKPT_DIR}"
  load_original_model_from=null
  trainer.devices="${NUM_DEVICES}"
  trainer.accelerator=gpu
  trainer.strategy=ddp_find_unused_parameters_true
  trainer.accumulate_grad_batches="${GRAD_ACCUM}"
  trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  dataloader.max_num_tokens="${TRAIN_MAX_TOKENS}"
  dataloader.val_max_num_tokens="${VAL_MAX_TOKENS}"
  callbacks.checkpoint_every_n_steps.every_n_train_steps="${STEP_CKPT_INTERVAL}"
  +logger.wandb.id="${WANDB_ID}"
  +logger.wandb.resume=must
)

echo "[pilot_dub_strong_resume] wandb id: ${WANDB_ID}"
echo "[pilot_dub_strong_resume] resume run dir: ${RESUME_RUN_DIR}"
echo "[pilot_dub_strong_resume] latest checkpoint: ${LATEST_CKPT}"

RUNS="${RUNS}" SLEEP_SEC="${SLEEP_SEC}" "${CMD[@]}" "$@" 2>&1 | tee "${RESUME_STDOUT}"
