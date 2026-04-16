#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/../metrics/results/LRS3}"
LOG_PATH="${LOG_PATH:-${ROOT_DIR}/logs/watch_pilot_dub_threshold_sweep.log}"
SELECTION_JSON="${SELECTION_JSON:-${ROOT_DIR}/reports/glovarv16-strong-rerank2-full_LRS3_selection.json}"

declare -a NAMES=(
  "glovarv16-strong-rerank2-full-cand0-default"
  "glovarv16-strong-rerank2-full-cand1-seed2"
  "glovarv16-strong-rerank2-full-cand2-t09-p09"
)
M02_NAME="pilotdub-strong-rerank-full-m02"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

mkdir -p "$(dirname "${LOG_PATH}")"
export ROOT_DIR RESULTS_ROOT SELECTION_JSON LOG_PATH

latest_run() {
  local name="$1"
  find "${RESULTS_ROOT}/${name}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

has_metrics() {
  local name="$1"
  local run
  run="$(latest_run "${name}")"
  [[ -n "${run}" && -f "${run}/system1/metrics.json" && -f "${run}/system2/metrics.json" ]]
}

while true; do
  all_done=true
  for name in "${NAMES[@]}" "${M02_NAME}"; do
    if has_metrics "${name}"; then
      log "${name}=done"
    else
      log "${name}=pending"
      all_done=false
    fi
  done

  if [[ "${all_done}" == true ]]; then
    break
  fi
  sleep 600
done

cd "${ROOT_DIR}"
python scripts/sweep_selection_thresholds.py \
  --selection-json "${SELECTION_JSON}" \
  --output-json "${ROOT_DIR}/reports/154.strong_threshold_sweep.json" \
  --output-md "${ROOT_DIR}/reports/154.strong_threshold_sweep.md"

python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["RESULTS_ROOT"])
names = ["glovarv14.6b-epoch0", "glovarv16-strong-rerank2-full", "pilotdub-strong-rerank-full-m02"]

def latest(name):
    runs = sorted([p for p in (root / name).glob("*") if p.is_dir() and p.name != "syncnet_tmp"])
    if not runs:
        return None
    return runs[-1]

def stats(name, system):
    run = latest(name)
    if run is None:
        return None
    f = run / system / "metrics.json"
    if not f.exists():
        return None
    d = json.load(open(f))["metrics"]
    n = len(d)
    wers = [float(v["wer"]["score"]) for v in d.values()]
    return {
        "wer": sum(wers) / n,
        "lse_d": sum(float(v["lse_d"]) for v in d.values()) / n,
        "lse_c": sum(float(v["lse_c"]) for v in d.values()) / n,
        "utmos": sum(float(v["utmos"]) for v in d.values()) / n,
        "spk": sum(float(v.get("spk_sim", 0.0)) for v in d.values()) / n,
        "gt25": sum(w > 25 for w in wers) / n,
        "gt50": sum(w > 50 for w in wers) / n,
    }

lines = ["# PILOT-Dub Strong Threshold Follow-up Summary", ""]
lines += ["| model | macro WER | macro LSE-D | macro LSE-C | macro UTMOS | macro SPK | macro >25 | macro >50 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
for name in names:
    vals = [stats(name, s) for s in ["system1", "system2"]]
    if any(v is None for v in vals):
        continue
    macro = {k: (vals[0][k] + vals[1][k]) / 2 for k in vals[0]}
    lines.append(
        f"| {name} | {macro['wer']:.4f} | {macro['lse_d']:.4f} | {macro['lse_c']:.4f} | "
        f"{macro['utmos']:.4f} | {macro['spk']:.4f} | {macro['gt25']:.4f} | {macro['gt50']:.4f} |"
    )
lines += ["", "See `154.strong_threshold_sweep.md` for the offline threshold sweep over exact candidate metrics."]
Path(os.environ["ROOT_DIR"]).joinpath("reports/155.strong_threshold_followup_summary.md").write_text("\n".join(lines), encoding="utf-8")
PY

log "threshold sweep completed"
