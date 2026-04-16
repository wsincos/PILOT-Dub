#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT.parent / "metrics" / "results" / "LRS3-mini20"

def latest_run(result_root: Path) -> Path:
    runs = sorted([p for p in result_root.glob("*") if p.is_dir() and p.name != "syncnet_tmp"])
    if not runs:
        raise FileNotFoundError(f"No timestamped metrics run under {result_root}")
    return runs[-1]


def load_metrics(result_root: Path, system: str) -> dict[str, dict]:
    run = latest_run(result_root)
    metrics_path = run / system / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    return json.load(metrics_path.open("r", encoding="utf-8"))["metrics"]


def sample_stats(records: dict[str, dict]) -> dict[str, float]:
    wers = [float(v["wer"]["score"]) for v in records.values()]
    return {
        "wer": mean(wers),
        "lse_d": mean(float(v["lse_d"]) for v in records.values()),
        "lse_c": mean(float(v["lse_c"]) for v in records.values()),
        "utmos": mean(float(v["utmos"]) for v in records.values()),
        "spk_sim": mean(float(v.get("spk_sim", 0.0)) for v in records.values()),
        "zero_rate": sum(w == 0 for w in wers) / len(wers),
        "gt25": sum(w > 25 for w in wers) / len(wers),
        "gt50": sum(w > 50 for w in wers) / len(wers),
    }


def wer_threshold(base_wer: float, abs_delta: float, rel_delta: float) -> float:
    abs_bound = base_wer + abs_delta
    rel_bound = base_wer * (1.0 + rel_delta) if base_wer > 0 else abs_bound
    return min(abs_bound, rel_bound)


def choose_oracle(
    sample_id: str,
    candidate_records: list[dict[str, dict]],
    baseline_record: dict,
    mode: str,
    abs_delta: float,
    rel_delta: float,
) -> tuple[int, dict]:
    base_wer = float(baseline_record["wer"]["score"])
    max_wer = wer_threshold(base_wer, abs_delta=abs_delta, rel_delta=rel_delta)
    eligible = []
    for idx, records in enumerate(candidate_records):
        record = records[sample_id]
        wer = float(record["wer"]["score"])
        if wer <= max_wer:
            eligible.append((idx, record))
    if not eligible:
        return 0, baseline_record
    if mode == "lse_c":
        return max(eligible, key=lambda item: float(item[1]["lse_c"]))
    if mode == "lse_d":
        return min(eligible, key=lambda item: float(item[1]["lse_d"]))
    raise ValueError(f"unknown oracle mode: {mode}")


def aggregate_selected(selected: dict[str, dict]) -> dict[str, float]:
    return sample_stats(selected)


def analyze(args: argparse.Namespace) -> dict:
    root = Path(args.results_root)
    candidate_names = [x.strip() for x in args.candidates.split(",") if x.strip()]
    if len(candidate_names) < 2:
        raise ValueError("Need at least two candidates, with candidate 0 as baseline.")

    output: dict = {
        "dataset": args.dataset,
        "candidate_names": candidate_names,
        "baseline": candidate_names[0],
        "abs_delta": args.abs_delta,
        "rel_delta": args.rel_delta,
        "systems": {},
    }

    for system in args.systems.split(","):
        system = system.strip()
        if not system:
            continue
        candidate_records = [
            load_metrics(root / name, system=system)
            for name in candidate_names
        ]
        sample_ids = sorted(set(candidate_records[0].keys()))
        for records in candidate_records[1:]:
            missing = set(sample_ids) - set(records.keys())
            if missing:
                raise ValueError(f"{system}: candidate missing samples: {sorted(missing)[:5]}")

        candidate_stats = {
            name: sample_stats(records)
            for name, records in zip(candidate_names, candidate_records)
        }

        oracle_outputs = {}
        choice_counts = {}
        for mode in ("lse_c", "lse_d"):
            selected = {}
            choices = {name: 0 for name in candidate_names}
            for sample_id in sample_ids:
                idx, record = choose_oracle(
                    sample_id=sample_id,
                    candidate_records=candidate_records,
                    baseline_record=candidate_records[0][sample_id],
                    mode=mode,
                    abs_delta=args.abs_delta,
                    rel_delta=args.rel_delta,
                )
                selected[sample_id] = record
                choices[candidate_names[idx]] += 1
            oracle_outputs[mode] = aggregate_selected(selected)
            choice_counts[mode] = choices

        output["systems"][system] = {
            "candidate_stats": candidate_stats,
            "oracle": oracle_outputs,
            "oracle_choice_counts": choice_counts,
        }

    return output


def print_summary(result: dict) -> None:
    for system, data in result["systems"].items():
        print(f"\n[{system}] baseline={result['baseline']}")
        base = data["candidate_stats"][result["baseline"]]
        print(
            "baseline "
            f"WER={base['wer']:.4f} LSE-D={base['lse_d']:.4f} LSE-C={base['lse_c']:.4f} "
            f">25={base['gt25']:.4f} >50={base['gt50']:.4f}"
        )
        for mode, stats in data["oracle"].items():
            print(
                f"oracle_{mode} "
                f"WER={stats['wer']:.4f} LSE-D={stats['lse_d']:.4f} LSE-C={stats['lse_c']:.4f} "
                f">25={stats['gt25']:.4f} >50={stats['gt50']:.4f} "
                f"choices={data['oracle_choice_counts'][mode]}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze K-candidate oracle reranking from metrics outputs.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset", default="LRS3-mini20")
    parser.add_argument("--candidates", required=True, help="Comma-separated candidate result names. First is baseline.")
    parser.add_argument("--systems", default="system1,system2")
    parser.add_argument("--abs-delta", type=float, default=0.5)
    parser.add_argument("--rel-delta", type=float, default=0.10)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    result = analyze(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print_summary(result)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
