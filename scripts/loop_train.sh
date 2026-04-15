#!/usr/bin/env bash
set -u

RUNS="${RUNS:-50}"
SLEEP_SEC="${SLEEP_SEC:-10}"
AUTO_KILL_OOM="${AUTO_KILL_OOM:-1}"
OOM_REGEX="${OOM_REGEX:-CUDA out of memory|out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDNN_STATUS_ALLOC_FAILED}"

echo "[loop_train] runs=${RUNS} sleep=${SLEEP_SEC}s args=$*"

cleanup_on_signal() {
  local group_pid="${pid:-}"
  if [ -n "${group_pid}" ]; then
    echo "[loop_train] received signal, killing process group ${group_pid}"
    kill -- "-${group_pid}" 2>/dev/null || true
    sleep 2
    kill -9 -- "-${group_pid}" 2>/dev/null || true
  fi
  exit 130
}

trap cleanup_on_signal INT TERM

for i in $(seq 1 "${RUNS}"); do
  echo "[loop_train] === run ${i}/${RUNS} ==="
  if [ "${AUTO_KILL_OOM}" -eq 1 ]; then
    PYTHONUNBUFFERED=1 OOM_REGEX="${OOM_REGEX}" setsid bash -c '
      set -o pipefail
      pgid=$(ps -o pgid= $$ | tr -d " ")
      python scripts/train.py "$@" 2>&1 | tee >(grep -m1 -E "$OOM_REGEX" >/dev/null && kill -- "-$pgid")
    ' bash "$@" &
  else
    setsid python scripts/train.py "$@" &
  fi
  pid=$!
  wait "${pid}"
  exit_code=$?
  if [ "${exit_code}" -ne 0 ]; then
    echo "[loop_train] run ${i} exited with code ${exit_code} (continuing)"
    echo "[loop_train] cleaning up process group ${pid}"
    kill -- "-${pid}" 2>/dev/null || true
    sleep 5
    kill -9 -- "-${pid}" 2>/dev/null || true
  else
    echo "[loop_train] run ${i} finished successfully"
  fi
  if [ "${i}" -lt "${RUNS}" ]; then
    echo "[loop_train] sleeping ${SLEEP_SEC}s before next run"
    sleep "${SLEEP_SEC}"
  fi
done
