#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def latest_run(root: Path, name: str) -> Path:
    runs = sorted([p for p in (root / name).glob("*") if p.is_dir() and p.name != "syncnet_tmp"])
    if not runs:
        raise FileNotFoundError(root / name)
    return runs[-1]


def load_metrics(root: Path, name: str, system: str) -> dict[str, dict]:
    path = latest_run(root, name) / system / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(open(path, "r", encoding="utf-8"))["metrics"]


def aggregate(rows: dict[str, dict]) -> dict[str, float]:
    vals = list(rows.values())
    wers = [float(v["wer"]["score"]) for v in vals]
    n = len(vals)
    return {
        "wer": sum(wers) / n,
        "zero": sum(w == 0 for w in wers) / n,
        "gt25": sum(w > 25 for w in wers) / n,
        "gt50": sum(w > 50 for w in wers) / n,
        "lse_d": sum(float(v["lse_d"]) for v in vals) / n,
        "lse_c": sum(float(v["lse_c"]) for v in vals) / n,
        "utmos": sum(float(v["utmos"]) for v in vals) / n,
        "spk_sim": sum(float(v.get("spk_sim", 0.0)) for v in vals) / n,
    }


def select_for_threshold(selection: dict, candidate_metrics: list[dict[str, dict]], system: str, threshold: float, fallback_index: int) -> tuple[dict[str, dict], dict[str, int]]:
    names = selection["candidate_names"]
    per_sample = selection["systems"][system]["per_sample"]
    out = {}
    choices = {name: 0 for name in names}
    for sample_id, row in per_sample.items():
        scores = row["source_candidate_scores"]
        vals = [float(scores.get(name, float("-inf"))) for name in names]
        best_idx = max(range(len(vals)), key=lambda i: vals[i])
        if vals[best_idx] - vals[fallback_index] < threshold:
            best_idx = fallback_index
        out[sample_id] = candidate_metrics[best_idx][sample_id]
        choices[names[best_idx]] += 1
    return out, choices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--results-root", default="/data1/jinyu_wang/projects/metrics/results/LRS3")
    parser.add_argument("--thresholds", default="0,0.02,0.05,0.075,0.1,0.125,0.15,0.2,0.25,0.3,0.4,0.5")
    parser.add_argument("--systems", default="system1,system2")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    selection = json.load(open(args.selection_json, "r", encoding="utf-8"))
    names = selection["candidate_names"]
    fallback_index = int(selection.get("fallback_index", 0))
    root = Path(args.results_root)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    thresholds = [float(x) for x in args.thresholds.split(",")]

    candidate_by_system = {
        system: [load_metrics(root, name, system) for name in names]
        for system in systems
    }

    result = {
        "selection_json": args.selection_json,
        "candidate_names": names,
        "fallback_index": fallback_index,
        "thresholds": [],
    }

    for th in thresholds:
        entry = {"threshold": th, "systems": {}}
        macro_acc = []
        for system in systems:
            selected, choices = select_for_threshold(
                selection=selection,
                candidate_metrics=candidate_by_system[system],
                system=system,
                threshold=th,
                fallback_index=fallback_index,
            )
            stats = aggregate(selected)
            entry["systems"][system] = {"metrics": stats, "choices": choices}
            macro_acc.append(stats)
        entry["macro"] = {
            key: sum(x[key] for x in macro_acc) / len(macro_acc)
            for key in macro_acc[0]
        }
        result["thresholds"].append(entry)

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Selection Threshold Sweep", ""]
    lines.append(f"- selection: `{args.selection_json}`")
    lines.append("")
    lines.append("| threshold | macro WER | macro LSE-D | macro LSE-C | macro UTMOS | macro SPK | macro >25 | macro >50 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for entry in result["thresholds"]:
        m = entry["macro"]
        lines.append(
            f"| {entry['threshold']:.3f} | {m['wer']:.4f} | {m['lse_d']:.4f} | {m['lse_c']:.4f} | {m['utmos']:.4f} | {m['spk_sim']:.4f} | {m['gt25']:.4f} | {m['gt50']:.4f} |"
        )
    lines.append("")
    for entry in result["thresholds"]:
        lines.append(f"## threshold={entry['threshold']:.3f}")
        for system, payload in entry["systems"].items():
            m = payload["metrics"]
            lines.append(
                f"- {system}: WER={m['wer']:.4f}, LSE-D={m['lse_d']:.4f}, LSE-C={m['lse_c']:.4f}, >25={m['gt25']:.4f}, >50={m['gt50']:.4f}, choices={payload['choices']}"
            )
        lines.append("")
    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(out_md)


if __name__ == "__main__":
    main()
