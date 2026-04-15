#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data1/jinyu_wang/projects/PILOT-Dub"
METRICS_ROOT="/data1/jinyu_wang/projects/metrics/results/LRS3"
REPORT_PATH="${ROOT_DIR}/reports/153.pilot_dub_full_rerank_results_summary.md"
LOG_PATH="/tmp/watch_pilot_dub_full_results.log"

STRONG_NAME="pilotdub-strong-rerank-full"
CLEAN_NAME="pilotdub-base-rerank-full"

cd "${ROOT_DIR}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_PATH}"
}

latest_run() {
  local name="$1"
  find "${METRICS_ROOT}/${name}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

has_metrics() {
  local name="$1"
  local run
  run="$(latest_run "${name}")"
  [[ -n "${run}" && -f "${run}/system1/metrics.json" && -f "${run}/system2/metrics.json" ]]
}

while true; do
  strong_status="pending"
  clean_status="pending"
  if has_metrics "${STRONG_NAME}"; then
    strong_status="done"
  fi
  if has_metrics "${CLEAN_NAME}"; then
    clean_status="done"
  fi
  log "strong=${strong_status} clean=${clean_status}"

  if [[ "${strong_status}" == "done" && "${clean_status}" == "done" ]]; then
    break
  fi

  for session in eval_pilotdub_strong_full eval_pilotdub_base_full; do
    if tmux has-session -t "${session}" 2>/dev/null; then
      pane="$(tmux capture-pane -pt "${session}" -S -8 2>/dev/null | tr '\n' ' ' | tail -c 500)"
      log "${session}: ${pane}"
    else
      log "${session}: no tmux session"
    fi
  done
  sleep 600
done

python - <<'PY'
import json
from pathlib import Path

root = Path("/data1/jinyu_wang/projects/metrics/results/LRS3")
report = Path("/data1/jinyu_wang/projects/PILOT-Dub/reports/153.pilot_dub_full_rerank_results_summary.md")
names = [
    ("PILOT-Dub strong", "pilotdub-strong-rerank-full"),
    ("PILOT-Dub base", "pilotdub-base-rerank-full"),
]

def latest(name):
    runs = sorted([p for p in (root / name).glob("*") if p.is_dir() and p.name != "syncnet_tmp"])
    if not runs:
        raise FileNotFoundError(root / name)
    return runs[-1]

def stats(run, system):
    d = json.load(open(run / system / "metrics.json", "r", encoding="utf-8"))["metrics"]
    n = len(d)
    wers = [float(v["wer"]["score"]) for v in d.values()]
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "n": n,
        "wer": mean(wers),
        "zero": sum(w == 0 for w in wers) / n,
        "gt25": sum(w > 25 for w in wers) / n,
        "gt50": sum(w > 50 for w in wers) / n,
        "lse_d": mean([float(v["lse_d"]) for v in d.values()]),
        "lse_c": mean([float(v["lse_c"]) for v in d.values()]),
        "utmos": mean([float(v["utmos"]) for v in d.values()]),
        "spk_sim": mean([float(v.get("spk_sim", 0.0)) for v in d.values()]),
    }

lines = ["# PILOT-Dub Full LRS3 Rerank Results Summary", ""]
rows = []
for label, name in names:
    run = latest(name)
    lines += [f"## {label}", "", f"- result dir: `{run}`", ""]
    sys_stats = {}
    for system in ["system1", "system2"]:
        s = stats(run, system)
        sys_stats[system] = s
        lines += [
            f"### {system}",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| WER | {s['wer']:.4f} |",
            f"| zero_rate | {s['zero']:.4f} |",
            f"| >25 | {s['gt25']:.4f} |",
            f"| >50 | {s['gt50']:.4f} |",
            f"| LSE-D | {s['lse_d']:.4f} |",
            f"| LSE-C | {s['lse_c']:.4f} |",
            f"| UTMOS | {s['utmos']:.4f} |",
            f"| spk_sim | {s['spk_sim']:.4f} |",
            "",
        ]
    macro = {
        k: (sys_stats["system1"][k] + sys_stats["system2"][k]) / 2
        for k in ["wer", "zero", "gt25", "gt50", "lse_d", "lse_c", "utmos", "spk_sim"]
    }
    rows.append((label, macro))

lines += [
    "## Macro Comparison",
    "",
    "| model | WER | zero | >25 | >50 | LSE-D | LSE-C | UTMOS | spk_sim |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for label, m in rows:
    lines.append(
        f"| {label} | {m['wer']:.4f} | {m['zero']:.4f} | {m['gt25']:.4f} | {m['gt50']:.4f} | "
        f"{m['lse_d']:.4f} | {m['lse_c']:.4f} | {m['utmos']:.4f} | {m['spk_sim']:.4f} |"
    )
lines.append("")
report.write_text("\n".join(lines), encoding="utf-8")
print(report)
PY

log "completed; report=${REPORT_PATH}"
