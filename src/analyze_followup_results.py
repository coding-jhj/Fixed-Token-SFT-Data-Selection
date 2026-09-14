"""Analyze sensitivity and additional-seed frozen evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


BENCHMARKS = ["ifeval", "gsm8k", "bbh"]
EXPECTED_ROWS = {"ifeval": 192, "gsm8k": 256, "bbh": 216}
METRICS = [
    "ifeval_prompt_strict",
    "ifeval_prompt_loose",
    "ifeval_instruction_strict",
    "ifeval_instruction_loose",
    "gsm8k_accuracy",
    "bbh_accuracy",
]


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append({"line": line_number, "error": "blank line"})
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_number, "error": f"json: {exc.msg}"})
                continue
            if not isinstance(value, dict):
                errors.append({"line": line_number, "error": "row is not an object"})
                continue
            rows.append(value)
    return rows, errors


def row_id(benchmark: str, row: dict[str, Any]) -> str | None:
    key = "key" if benchmark == "ifeval" else "example_id"
    value = row.get(key)
    return None if value is None else str(value)


def valid_score(benchmark: str, row: dict[str, Any]) -> bool:
    score = row.get("score")
    if not isinstance(score, dict):
        return False
    if benchmark == "ifeval":
        instruction_ids = row.get("instruction_id_list")
        return (
            isinstance(score.get("prompt_level_strict"), bool)
            and isinstance(score.get("prompt_level_loose"), bool)
            and isinstance(score.get("instruction_level_strict"), list)
            and isinstance(score.get("instruction_level_loose"), list)
            and isinstance(instruction_ids, list)
            and len(score["instruction_level_strict"]) == len(instruction_ids)
            and all(isinstance(value, bool) for value in score["instruction_level_strict"])
            and all(isinstance(value, bool) for value in score["instruction_level_loose"])
        )
    field = "exact_match" if benchmark == "gsm8k" else "normalized_accuracy"
    return isinstance(score.get(field), bool)


def scores(benchmark: str, row: dict[str, Any]) -> dict[str, float]:
    score = row["score"]
    if benchmark == "ifeval":
        strict = [float(value) for value in score["instruction_level_strict"]]
        loose = [float(value) for value in score["instruction_level_loose"]]
        return {
            "ifeval_prompt_strict": float(score["prompt_level_strict"]),
            "ifeval_prompt_loose": float(score["prompt_level_loose"]),
            "ifeval_instruction_strict": float(np.mean(strict)) if strict else math.nan,
            "ifeval_instruction_loose": float(np.mean(loose)) if loose else math.nan,
        }
    if benchmark == "gsm8k":
        return {"gsm8k_accuracy": float(score["exact_match"])}
    return {"bbh_accuracy": float(score["normalized_accuracy"])}


def expected_ids(subset_dir: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for benchmark in BENCHMARKS:
        rows, errors = load_jsonl(subset_dir / f"{benchmark}.jsonl")
        if errors:
            raise ValueError(f"invalid subset {benchmark}: {errors[:2]}")
        ids = [row_id(benchmark, row) for row in rows]
        if any(value is None for value in ids):
            raise ValueError(f"missing ID in {benchmark} subset")
        result[benchmark] = {value for value in ids if value is not None}
    return result


def response_lengths(tokenizer: Any, responses: list[str]) -> list[int]:
    if not responses:
        return []
    encoded = tokenizer(responses, add_special_tokens=False, truncation=False, padding=False)
    return [len(value) for value in encoded["input_ids"]]


def analyze_adapter(
    adapter_dir: Path,
    adapter_name: str,
    strategy: str,
    seed: int,
    expected: dict[str, set[str]],
    tokenizer: Any,
    limits: dict[str, int],
    expected_rows: dict[str, int],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    validation: dict[str, Any] = {}
    metric_maps: dict[str, dict[str, float]] = {metric: {} for metric in METRICS}
    totals = Counter()
    for benchmark in BENCHMARKS:
        path = adapter_dir / f"{benchmark}_outputs.jsonl"
        rows, parse_errors = load_jsonl(path)
        ids = [row_id(benchmark, row) for row in rows]
        counts = Counter(value for value in ids if value is not None)
        duplicate_ids = sorted(value for value, count in counts.items() if count > 1)
        actual_ids = {value for value in ids if value is not None}
        responses: list[str] = []
        valid_rows = 0
        malformed_rows = 0
        for row in rows:
            response = row.get("response")
            if isinstance(response, str) and response.strip() and valid_score(benchmark, row):
                valid_rows += 1
                responses.append(response)
                item_id = row_id(benchmark, row)
                if item_id is not None:
                    for metric, value in scores(benchmark, row).items():
                        metric_maps[metric][item_id] = value
                    if benchmark == "ifeval":
                        totals["instruction_strict_sum"] += sum(row["score"]["instruction_level_strict"])
                        totals["instruction_strict_count"] += len(row["score"]["instruction_level_strict"])
                        totals["instruction_loose_sum"] += sum(row["score"]["instruction_level_loose"])
                        totals["instruction_loose_count"] += len(row["score"]["instruction_level_loose"])
            else:
                malformed_rows += 1
        lengths = response_lengths(tokenizer, responses)
        validation[benchmark] = {
            "path": str(path),
            "rows": len(rows),
            "expected_rows": expected_rows[benchmark],
            "parse_errors": parse_errors,
            "duplicate_ids": duplicate_ids,
            "missing_ids": sorted(expected[benchmark] - actual_ids),
            "extra_ids": sorted(actual_ids - expected[benchmark]),
            "malformed_rows": malformed_rows,
            "empty_responses": sum(not isinstance(row.get("response"), str) or not row["response"].strip() for row in rows),
            "response_token_limit": limits[benchmark],
            "possible_truncation_count": sum(length >= limits[benchmark] for length in lengths),
            "response_token_min": min(lengths) if lengths else None,
            "response_token_mean": float(np.mean(lengths)) if lengths else None,
            "response_token_max": max(lengths) if lengths else None,
            "valid_scored_rows": valid_rows,
        }
    complete = all(
        value["rows"] == value["expected_rows"]
        and not value["parse_errors"]
        and not value["duplicate_ids"]
        and not value["missing_ids"]
        and not value["extra_ids"]
        and value["malformed_rows"] == 0
        for value in validation.values()
    )
    validation_record = {
        "adapter": adapter_name,
        "strategy": strategy,
        "seed": seed,
        "all_files_complete": complete,
        "possible_truncation_total": sum(value["possible_truncation_count"] for value in validation.values()),
        "files": validation,
    }
    row: dict[str, Any] = {"adapter": adapter_name, "strategy": strategy, "seed": seed}
    for metric in METRICS:
        if metric == "ifeval_instruction_strict":
            row[metric] = totals["instruction_strict_sum"] / totals["instruction_strict_count"]
        elif metric == "ifeval_instruction_loose":
            row[metric] = totals["instruction_loose_sum"] / totals["instruction_loose_count"]
        else:
            values = list(metric_maps[metric].values())
            row[metric] = float(np.mean(values)) if values else math.nan
    return validation_record, {"row": row, "maps": metric_maps}


def bootstrap(
    records: dict[str, dict[str, dict[str, float]]],
    treatment: str,
    control: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any] | None:
    treatment_records = [record for record in records.values() if record["strategy"] == treatment]
    control_records = [record for record in records.values() if record["strategy"] == control]
    if not treatment_records or not control_records:
        return None
    by_seed_t = {record["seed"]: record["maps"][metric] for record in treatment_records}
    by_seed_c = {record["seed"]: record["maps"][metric] for record in control_records}
    common_seeds = sorted(set(by_seed_t) & set(by_seed_c))
    if not common_seeds:
        return None
    ids = sorted(set.intersection(*(set(by_seed_t[run_seed]) & set(by_seed_c[run_seed]) for run_seed in common_seeds)))
    differences = np.asarray(
        [
            np.mean([by_seed_t[run_seed][item_id] - by_seed_c[run_seed][item_id] for run_seed in common_seeds])
            for item_id in ids
        ],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    samples = differences[rng.integers(0, len(differences), size=(repetitions, len(differences)))].mean(axis=1)
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "seeds": common_seeds,
        "n_paired_examples": len(ids),
        "point_estimate": float(np.mean(differences)),
        "ci_95_lower": float(np.quantile(samples, 0.025)),
        "ci_95_upper": float(np.quantile(samples, 0.975)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, default=Path("work/evaluation_subsets_final"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter", action="append", required=True, help="name=strategy=seed")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--tokenizer-revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    parser.add_argument("--ifeval-max-new-tokens", type=int, default=1024)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=256)
    parser.add_argument("--ifeval-rows", type=int, default=192)
    parser.add_argument("--gsm8k-rows", type=int, default=256)
    parser.add_argument("--bbh-rows", type=int, default=216)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    adapter_specs: list[tuple[str, str, int]] = []
    for value in args.adapter:
        name, strategy, seed_text = value.split("=", 2)
        adapter_specs.append((name, strategy, int(seed_text)))
    limits = {"ifeval": args.ifeval_max_new_tokens, "gsm8k": args.benchmark_max_new_tokens, "bbh": args.benchmark_max_new_tokens}
    expected_rows = {"ifeval": args.ifeval_rows, "gsm8k": args.gsm8k_rows, "bbh": args.bbh_rows}
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision, local_files_only=True)
    expected = expected_ids(args.subset_dir)
    validation: dict[str, Any] = {"expected_rows": expected_rows, "limits": limits, "adapters": {}}
    metric_rows: list[dict[str, Any]] = []
    records: dict[str, dict[str, dict[str, float]]] = {}
    for name, strategy, seed in adapter_specs:
        item_validation, item_data = analyze_adapter(args.evaluation_dir / name, name, strategy, seed, expected, tokenizer, limits, expected_rows)
        validation["adapters"][name] = item_validation
        metric_rows.append(item_data["row"])
        records[name] = {"strategy": strategy, "seed": seed, "maps": item_data["maps"]}
    validation["all_files_complete"] = all(item["all_files_complete"] for item in validation["adapters"].values())
    validation["possible_truncation_total"] = sum(item["possible_truncation_total"] for item in validation["adapters"].values())
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metric_fields = ["adapter", "strategy", "seed", *METRICS]
    write_csv(args.output_dir / "per_adapter_metrics.csv", metric_fields, metric_rows)
    strategies = sorted({row["strategy"] for row in metric_rows})
    aggregate_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        rows = [row for row in metric_rows if row["strategy"] == strategy]
        aggregate: dict[str, Any] = {"strategy": strategy, "n_adapters": len(rows), "seeds": ",".join(str(row["seed"]) for row in rows)}
        for metric in METRICS:
            values = np.asarray([row[metric] for row in rows], dtype=float)
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        aggregate_rows.append(aggregate)
    aggregate_fields = ["strategy", "n_adapters", "seeds"] + [field for metric in METRICS for field in (f"{metric}_mean", f"{metric}_sd")]
    write_csv(args.output_dir / "strategy_metrics.csv", aggregate_fields, aggregate_rows)

    bootstrap_rows = []
    for metric in ["ifeval_prompt_strict", "ifeval_instruction_strict", "gsm8k_accuracy", "bbh_accuracy"]:
        item = bootstrap(records, "diversity", "random", metric, args.bootstrap_repetitions, 2026)
        if item is not None:
            bootstrap_rows.append(item)
    if bootstrap_rows:
        write_csv(args.output_dir / "paired_bootstrap.csv", list(bootstrap_rows[0]), bootstrap_rows)
    metadata = {
        "evaluation_dir": str(args.evaluation_dir),
        "limits": limits,
        "adapters": [{"name": name, "strategy": strategy, "seed": seed} for name, strategy, seed in adapter_specs],
        "tokenizer": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "validation_all_files_complete": validation["all_files_complete"],
    }
    (args.output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"validation_all_files_complete": validation["all_files_complete"], "possible_truncation_total": validation["possible_truncation_total"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
