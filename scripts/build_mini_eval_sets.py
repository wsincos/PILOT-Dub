#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "refine-logs"


@dataclass
class DatasetSpec:
    name: str
    source_list: Path
    output_prefix: str


@dataclass
class Entry:
    raw_line: str
    speaker: str
    target_id: str
    target_text: str
    target_avhubert_num: str
    ref_speaker: str
    ref_id: str
    ref_text: str
    sample_id: str
    word_count: int
    line_idx: int


DATASETS = {
    "lrs3": DatasetSpec(
        name="LRS3",
        source_list=PROJECT_ROOT / "data/dataset/LRS3_Test_True/LRS3_Test.txt",
        output_prefix="LRS3_Test",
    ),
    "celebvdub": DatasetSpec(
        name="CelebV-Dub",
        source_list=PROJECT_ROOT / "data/CelebVDubTest/CelebVDubTest.txt",
        output_prefix="CelebVDubTest",
    ),
}


def stable_int(key: str, seed: int) -> int:
    digest = hashlib.md5(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def parse_entries(path: Path) -> List[Entry]:
    entries: List[Entry] = []
    for line_idx, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 6:
            continue
        if len(parts) == 6:
            speaker, target_id, target_text, target_avhubert_num, ref_id, ref_text = parts
            ref_speaker = speaker
        else:
            if len(parts) > 7:
                parts = parts[:6] + [" | ".join(parts[6:])]
            speaker, target_id, target_text, target_avhubert_num, ref_speaker, ref_id, ref_text = parts
        sample_id = f"{speaker}/{target_id}"
        entries.append(
            Entry(
                raw_line=line,
                speaker=speaker,
                target_id=target_id,
                target_text=target_text,
                target_avhubert_num=target_avhubert_num,
                ref_speaker=ref_speaker,
                ref_id=ref_id,
                ref_text=ref_text,
                sample_id=sample_id,
                word_count=len(target_text.split()),
                line_idx=line_idx,
            )
        )
    return entries


def quantile_edges(values: Sequence[int], num_bins: int) -> List[int]:
    sorted_vals = sorted(values)
    edges: List[int] = []
    for i in range(1, num_bins):
        idx = min(len(sorted_vals) - 1, max(0, round(len(sorted_vals) * i / num_bins) - 1))
        edge = sorted_vals[idx]
        if edges and edge < edges[-1]:
            edge = edges[-1]
        edges.append(edge)
    return edges


def assign_bin(word_count: int, edges: Sequence[int]) -> int:
    for idx, edge in enumerate(edges):
        if word_count <= edge:
            return idx
    return len(edges)


def allocate_targets(total: int, num_bins: int) -> List[int]:
    base = total // num_bins
    remainder = total % num_bins
    return [base + (1 if i < remainder else 0) for i in range(num_bins)]


def build_order(entries: Sequence[Entry], seed: int, global_speaker_freq: Counter) -> List[str]:
    speakers = sorted({entry.speaker for entry in entries}, key=lambda sp: (global_speaker_freq[sp], stable_int(sp, seed)))
    order: List[str] = []
    speaker_to_entries: Dict[str, List[Entry]] = {}
    for speaker in speakers:
        speaker_entries = [entry for entry in entries if entry.speaker == speaker]
        speaker_entries.sort(
            key=lambda entry: (
                entry.word_count,
                stable_int(entry.sample_id, seed),
                entry.target_id,
            )
        )
        speaker_to_entries[speaker] = speaker_entries

    progress = True
    while progress:
        progress = False
        for speaker in speakers:
            queue = speaker_to_entries[speaker]
            if queue:
                order.append(queue.pop(0).sample_id)
                progress = True
    return order


def select_entries(entries: Sequence[Entry], subset_size: int, seed: int) -> List[Entry]:
    if subset_size >= len(entries):
        return list(entries)

    num_bins = 4
    edges = quantile_edges([entry.word_count for entry in entries], num_bins)
    target_per_bin = allocate_targets(subset_size, num_bins)
    global_speaker_freq = Counter(entry.speaker for entry in entries)

    bin_to_entries: Dict[int, List[Entry]] = defaultdict(list)
    for entry in entries:
        bin_to_entries[assign_bin(entry.word_count, edges)].append(entry)

    ordered_bins: Dict[int, List[Entry]] = {}
    for bin_idx, bin_entries in bin_to_entries.items():
        ordered_sample_ids = build_order(bin_entries, seed + bin_idx, global_speaker_freq)
        by_id = {entry.sample_id: entry for entry in bin_entries}
        ordered_bins[bin_idx] = [by_id[sample_id] for sample_id in ordered_sample_ids]

    speaker_cap = 1 if subset_size <= len(global_speaker_freq) else 2
    selected: List[Entry] = []
    selected_ids = set()
    speaker_counts: Counter = Counter()

    def try_take(candidates: List[Entry], need: int, cap: int | None) -> List[Entry]:
        taken: List[Entry] = []
        for entry in candidates:
            if entry.sample_id in selected_ids:
                continue
            if cap is not None and speaker_counts[entry.speaker] >= cap:
                continue
            taken.append(entry)
            selected_ids.add(entry.sample_id)
            speaker_counts[entry.speaker] += 1
            if len(taken) >= need:
                break
        return taken

    for cap in [speaker_cap, speaker_cap + 1, None]:
        for bin_idx in range(num_bins):
            already = sum(1 for entry in selected if assign_bin(entry.word_count, edges) == bin_idx)
            need = max(0, target_per_bin[bin_idx] - already)
            if need <= 0:
                continue
            taken = try_take(ordered_bins.get(bin_idx, []), need, cap)
            selected.extend(taken)
        if len(selected) >= subset_size:
            break

    if len(selected) < subset_size:
        all_entries = []
        for bin_idx in range(num_bins):
            all_entries.extend(ordered_bins.get(bin_idx, []))
        all_entries.sort(
            key=lambda entry: (
                speaker_counts[entry.speaker],
                global_speaker_freq[entry.speaker],
                stable_int(entry.sample_id, seed + 97),
            )
        )
        selected.extend(try_take(all_entries, subset_size - len(selected), None))

    selected = selected[:subset_size]
    return selected


def write_subset(path: Path, entries: Sequence[Entry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entry.raw_line for entry in entries) + "\n", encoding="utf-8")


def summarize_subset(dataset_name: str, source_entries: Sequence[Entry], subset_entries: Sequence[Entry], subset_name: str) -> Dict[str, object]:
    source_word_counts = [entry.word_count for entry in source_entries]
    subset_word_counts = [entry.word_count for entry in subset_entries]
    speaker_counts = Counter(entry.speaker for entry in subset_entries)
    return {
        "dataset": dataset_name,
        "subset": subset_name,
        "num_samples": len(subset_entries),
        "num_source_samples": len(source_entries),
        "num_speakers": len(speaker_counts),
        "source_num_speakers": len({entry.speaker for entry in source_entries}),
        "word_count": {
            "source_min": min(source_word_counts),
            "source_median": sorted(source_word_counts)[len(source_word_counts) // 2],
            "source_max": max(source_word_counts),
            "subset_min": min(subset_word_counts),
            "subset_median": sorted(subset_word_counts)[len(subset_word_counts) // 2],
            "subset_max": max(subset_word_counts),
        },
        "top_speakers": speaker_counts.most_common(10),
        "sample_ids": [entry.sample_id for entry in subset_entries],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20, 50])
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS.keys()), default=sorted(DATASETS.keys()))
    parser.add_argument("--report-root", default=str(REPORT_ROOT))
    args = parser.parse_args()

    report_root = Path(args.report_root)
    report_root.mkdir(parents=True, exist_ok=True)

    all_summaries: List[Dict[str, object]] = []
    for dataset_key in args.datasets:
        spec = DATASETS[dataset_key]
        source_entries = parse_entries(spec.source_list)
        for size in args.sizes:
            subset_entries = select_entries(source_entries, size, seed=args.seed + size)
            output_path = spec.source_list.with_name(f"{spec.output_prefix}_mini{size}.txt")
            write_subset(output_path, subset_entries)
            all_summaries.append(
                {
                    "source_list": str(spec.source_list),
                    "output_list": str(output_path),
                    **summarize_subset(spec.name, source_entries, subset_entries, f"mini{size}"),
                }
            )

    json_path = report_root / "mini_eval_sets_summary.json"
    md_path = report_root / "mini_eval_sets_summary.md"
    json_path.write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Mini Eval Sets Summary",
        "",
        "| dataset | subset | num_samples | num_speakers | word_count(min/median/max) | output_list |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in all_summaries:
        wc = item["word_count"]
        lines.append(
            f"| {item['dataset']} | {item['subset']} | {item['num_samples']} | {item['num_speakers']} | "
            f"{wc['subset_min']}/{wc['subset_median']}/{wc['subset_max']} | {item['output_list']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
