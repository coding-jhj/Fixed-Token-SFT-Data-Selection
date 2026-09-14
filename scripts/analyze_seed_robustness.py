"""Combine the original two seeds with seed 2026 for random/diversity robustness."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


METRICS = ["ifeval_prompt_strict", "ifeval_instruction_strict", "gsm8k_accuracy", "bbh_accuracy"]
BENCHMARKS = {"ifeval": 192, "gsm8k": 256, "bbh": 216}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_id(benchmark: str, row: dict) -> str:
    return str(row["key"] if benchmark == "ifeval" else row["example_id"])


def metric_values(benchmark: str, rows: list[dict]) -> dict[str, float]:
    result = {}
    for row in rows:
        score = row["score"]
        if benchmark == "ifeval":
            value = float(score["prompt_level_strict"])
        elif benchmark == "gsm8k":
            value = float(score["exact_match"])
        else:
            value = float(score["normalized_accuracy"])
        result[row_id(benchmark, row)] = value
    return result


def instruction_values(rows: list[dict]) -> dict[str, float]:
    return {
        row_id("ifeval", row): float(np.mean(row["score"]["instruction_level_strict"]))
        for row in rows
    }


def load_run(root: Path, adapter: str, benchmark: str, metric: str) -> dict[str, float]:
    rows = load_jsonl(root / adapter / f"{benchmark}_outputs.jsonl")
    if len(rows) != BENCHMARKS[benchmark]:
        raise ValueError(f"{root / adapter}: expected {BENCHMARKS[benchmark]} {benchmark} rows, found {len(rows)}")
    return instruction_values(rows) if metric == "ifeval_instruction_strict" else metric_values(benchmark, rows)


def bootstrap(values: np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--seed2026-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_roots = {
        "random_seed13": args.primary_dir,
        "random_seed42": args.primary_dir,
        "diversity_seed13": args.primary_dir,
        "diversity_seed42": args.primary_dir,
        "random_seed2026": args.seed2026_dir,
        "diversity_seed2026": args.seed2026_dir,
    }
    seed_records = []
    pair_records = []
    strategy_map = {"random": "random", "diversity": "diversity"}
    for benchmark in ["ifeval", "gsm8k", "bbh"]:
        metric_names = ["ifeval_prompt_strict", "ifeval_instruction_strict"] if benchmark == "ifeval" else [f"{benchmark}_accuracy"]
        for metric in metric_names:
            values = {}
            for adapter, root in adapter_roots.items():
                strategy, seed_text = adapter.rsplit("_seed", 1)
                values[adapter] = load_run(root, adapter, benchmark, metric)
                seed_records.append({"adapter": adapter, "strategy": strategy_map[strategy], "seed": int(seed_text), "metric": metric, "mean": float(np.mean(list(values[adapter].values())))})
            for seed in [13, 42, 2026]:
                treatment = f"diversity_seed{seed}"
                control = f"random_seed{seed}"
                common = sorted(set(values[treatment]) & set(values[control]))
                differences = np.asarray([values[treatment][item] - values[control][item] for item in common], dtype=float)
                point, lower, upper = bootstrap(differences, args.bootstrap_repetitions, 2026)
                pair_records.append({"metric": metric, "seed": seed, "n_paired_examples": len(common), "point_estimate": point, "ci_95_lower": lower, "ci_95_upper": upper})
            all_differences = []
            for item in sorted(set.intersection(*(set(values[f"diversity_seed{seed}"]) & set(values[f"random_seed{seed}"]) for seed in [13, 42, 2026]))):
                all_differences.append(np.mean([values[f"diversity_seed{seed}"][item] - values[f"random_seed{seed}"][item] for seed in [13, 42, 2026]]))
            point, lower, upper = bootstrap(np.asarray(all_differences), args.bootstrap_repetitions, 2026)
            pair_records.append({"metric": metric, "seed": "13,42,2026", "n_paired_examples": len(all_differences), "point_estimate": point, "ci_95_lower": lower, "ci_95_upper": upper})

    with (args.output_dir / "seed_robustness.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["adapter", "strategy", "seed", "metric", "mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(seed_records)
    with (args.output_dir / "three_seed_paired_bootstrap.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["metric", "seed", "n_paired_examples", "point_estimate", "ci_95_lower", "ci_95_upper"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pair_records)
    metadata = {"primary_dir": str(args.primary_dir), "seed2026_dir": str(args.seed2026_dir), "seeds": [13, 42, 2026], "strategy_scope": ["random", "diversity"], "bootstrap_repetitions": args.bootstrap_repetitions}
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(seed_records), "paired_rows": len(pair_records)}))


if __name__ == "__main__":
    main()
