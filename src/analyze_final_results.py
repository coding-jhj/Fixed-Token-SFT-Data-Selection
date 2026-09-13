"""Validate and analyze the frozen six-adapter evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ADAPTERS = [
    ("random_seed13", "random", 13),
    ("quality_seed13", "quality", 13),
    ("diversity_seed13", "diversity", 13),
    ("random_seed42", "random", 42),
    ("quality_seed42", "quality", 42),
    ("diversity_seed42", "diversity", 42),
]
STRATEGIES = ["random", "quality", "diversity"]
SEEDS = [13, 42]
BENCHMARKS = ["ifeval", "gsm8k", "bbh"]
EXPECTED_ROWS = {"ifeval": 192, "gsm8k": 256, "bbh": 216}
MAX_NEW_TOKENS = {"ifeval": 512, "gsm8k": 128, "bbh": 128}


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


def score_fields(benchmark: str) -> list[str]:
    if benchmark == "ifeval":
        return [
            "prompt_level_strict",
            "instruction_level_strict",
            "prompt_level_loose",
            "instruction_level_loose",
        ]
    if benchmark == "gsm8k":
        return ["exact_match"]
    return ["normalized_accuracy"]


def is_valid_score(benchmark: str, row: dict[str, Any]) -> bool:
    score = row.get("score")
    if not isinstance(score, dict):
        return False
    if benchmark == "ifeval":
        required = score_fields(benchmark)
        if any(not isinstance(score.get(field), (bool, np.bool_)) for field in required if field.startswith("prompt")):
            return False
        for field in ("instruction_level_strict", "instruction_level_loose"):
            values = score.get(field)
            if not isinstance(values, list) or any(not isinstance(value, (bool, np.bool_)) for value in values):
                return False
        instruction_ids = row.get("instruction_id_list")
        return isinstance(instruction_ids, list) and len(score["instruction_level_strict"]) == len(instruction_ids)
    return isinstance(score.get(score_fields(benchmark)[0]), (bool, np.bool_))


def scalar_scores(benchmark: str, row: dict[str, Any]) -> dict[str, float]:
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
        return {"gsm8k_accuracy": float(row["score"]["exact_match"])}
    return {"bbh_accuracy": float(row["score"]["normalized_accuracy"])}


def load_expected_ids(subset_dir: Path) -> dict[str, set[str]]:
    expected: dict[str, set[str]] = {}
    for benchmark in BENCHMARKS:
        rows, errors = load_jsonl(subset_dir / f"{benchmark}.jsonl")
        if errors:
            raise ValueError(f"invalid subset JSONL: {subset_dir / f'{benchmark}.jsonl'}: {errors[:2]}")
        ids = [row_id(benchmark, row) for row in rows]
        if any(value is None for value in ids):
            raise ValueError(f"missing ID in {benchmark} subset")
        expected[benchmark] = {value for value in ids if value is not None}
    return expected


def response_token_lengths(tokenizer: Any, responses: list[str]) -> list[int]:
    if not responses:
        return []
    encoded = tokenizer(responses, add_special_tokens=False, truncation=False, padding=False)
    return [len(value) for value in encoded["input_ids"]]


def validate_outputs(
    evaluation_dir: Path,
    subset_dir: Path,
    tokenizer: Any,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
]:
    expected_ids = load_expected_ids(subset_dir)
    validation: dict[str, Any] = {"expected_rows": EXPECTED_ROWS, "files": {}}
    per_adapter_scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    bbh_tasks: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    instruction_totals: dict[str, Counter[str]] = defaultdict(Counter)

    for adapter_name, strategy, seed in ADAPTERS:
        for benchmark in BENCHMARKS:
            path = evaluation_dir / adapter_name / f"{benchmark}_outputs.jsonl"
            rows, parse_errors = load_jsonl(path)
            ids = [row_id(benchmark, row) for row in rows]
            id_counts = Counter(value for value in ids if value is not None)
            duplicate_ids = sorted(key for key, count in id_counts.items() if count > 1)
            actual_ids = {value for value in ids if value is not None}
            missing_ids = sorted(expected_ids[benchmark] - actual_ids)
            extra_ids = sorted(actual_ids - expected_ids[benchmark])
            malformed_rows = 0
            empty_responses = 0
            score_rows: list[dict[str, float]] = []
            responses: list[str] = []
            task_scores: dict[str, list[float]] = defaultdict(list)

            for row in rows:
                response = row.get("response")
                valid = isinstance(response, str) and bool(response.strip()) and is_valid_score(benchmark, row)
                if not isinstance(response, str) or not response.strip():
                    empty_responses += 1
                if not valid:
                    malformed_rows += 1
                    continue
                responses.append(response)
                values = scalar_scores(benchmark, row)
                score_rows.append(values)
                if benchmark == "ifeval":
                    instruction_totals[adapter_name]["ifeval_instruction_strict_sum"] += sum(
                        bool(value) for value in row["score"]["instruction_level_strict"]
                    )
                    instruction_totals[adapter_name]["ifeval_instruction_strict_count"] += len(
                        row["score"]["instruction_level_strict"]
                    )
                    instruction_totals[adapter_name]["ifeval_instruction_loose_sum"] += sum(
                        bool(value) for value in row["score"]["instruction_level_loose"]
                    )
                    instruction_totals[adapter_name]["ifeval_instruction_loose_count"] += len(
                        row["score"]["instruction_level_loose"]
                    )
                item_id = row_id(benchmark, row)
                if item_id is not None:
                    for metric, value in values.items():
                        per_adapter_scores[adapter_name][metric][item_id] = value
                    if benchmark == "bbh":
                        task = str(row.get("subset", "<missing>"))
                        task_scores[task].append(values["bbh_accuracy"])

            token_lengths = response_token_lengths(tokenizer, responses)
            limit = MAX_NEW_TOKENS[benchmark]
            possible_truncation = sum(length >= limit for length in token_lengths)
            file_key = f"{adapter_name}/{benchmark}"
            validation["files"][file_key] = {
                "path": str(path),
                "rows": len(rows),
                "expected_rows": EXPECTED_ROWS[benchmark],
                "parse_errors": parse_errors,
                "duplicate_ids": duplicate_ids,
                "missing_ids": missing_ids,
                "extra_ids": extra_ids,
                "malformed_rows": malformed_rows,
                "empty_responses": empty_responses,
                "response_token_limit": limit,
                "possible_truncation_count": possible_truncation,
                "response_token_min": min(token_lengths) if token_lengths else None,
                "response_token_mean": float(np.mean(token_lengths)) if token_lengths else None,
                "response_token_max": max(token_lengths) if token_lengths else None,
                "valid_scored_rows": len(score_rows),
            }
            if benchmark == "bbh":
                task_means = {task: float(np.mean(values)) for task, values in sorted(task_scores.items())}
                bbh_tasks[adapter_name]["task_scores"] = task_means
                bbh_tasks[adapter_name]["task_macro"] = float(np.mean(list(task_means.values())))
                bbh_tasks[adapter_name]["worst_task"] = min(task_means.values())

    validation["all_files_complete"] = all(
        item["rows"] == item["expected_rows"]
        and not item["parse_errors"]
        and not item["duplicate_ids"]
        and not item["missing_ids"]
        and not item["extra_ids"]
        and item["malformed_rows"] == 0
        for item in validation["files"].values()
    )
    validation["possible_truncation_total"] = sum(
        item["possible_truncation_count"] for item in validation["files"].values()
    )
    global_instruction_metrics = {
        adapter_name: {
            "ifeval_instruction_strict": totals["ifeval_instruction_strict_sum"] / totals["ifeval_instruction_strict_count"],
            "ifeval_instruction_loose": totals["ifeval_instruction_loose_sum"] / totals["ifeval_instruction_loose_count"],
        }
        for adapter_name, totals in instruction_totals.items()
    }
    return validation, per_adapter_scores, bbh_tasks, global_instruction_metrics


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_sd(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def aggregate_metrics(
    per_adapter_scores: dict[str, dict[str, dict[str, float]]],
    bbh_tasks: dict[str, dict[str, dict[str, float]]],
    global_instruction_metrics: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metric_names = [
        "ifeval_prompt_strict",
        "ifeval_prompt_loose",
        "ifeval_instruction_strict",
        "ifeval_instruction_loose",
        "gsm8k_accuracy",
        "bbh_accuracy",
    ]
    per_seed_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    adapter_metadata = {name: (strategy, seed) for name, strategy, seed in ADAPTERS}
    for adapter_name, (strategy, seed) in adapter_metadata.items():
        row: dict[str, Any] = {"adapter": adapter_name, "strategy": strategy, "seed": seed}
        for metric in metric_names:
            values = list(per_adapter_scores[adapter_name][metric].values())
            if metric in {"ifeval_instruction_strict", "ifeval_instruction_loose"}:
                row[metric] = global_instruction_metrics[adapter_name][metric]
            else:
                row[metric] = float(np.mean(values)) if values else math.nan
        row["bbh_task_macro"] = bbh_tasks[adapter_name]["task_macro"]
        row["bbh_worst_task"] = bbh_tasks[adapter_name]["worst_task"]
        per_seed_rows.append(row)

    aggregate_metrics_names = metric_names + ["bbh_task_macro", "bbh_worst_task"]
    for strategy in STRATEGIES:
        rows = [row for row in per_seed_rows if row["strategy"] == strategy]
        row = {"strategy": strategy, "n_seeds": len(rows)}
        for metric in aggregate_metrics_names:
            mean, sd = mean_sd([float(item[metric]) for item in rows])
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_seed13"] = next(item[metric] for item in rows if item["seed"] == 13)
            row[f"{metric}_seed42"] = next(item[metric] for item in rows if item["seed"] == 42)
        aggregate_rows.append(row)

    return per_seed_rows, aggregate_rows, {"metric_names": aggregate_metrics_names}


def paired_bootstrap(
    per_adapter_scores: dict[str, dict[str, dict[str, float]]],
    metric: str,
    treatment: str,
    control: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    adapter_for = {(strategy, run_seed): name for name, strategy, run_seed in ADAPTERS}
    control_maps = [per_adapter_scores[adapter_for[(control, run_seed)]][metric] for run_seed in SEEDS]
    treatment_maps = [per_adapter_scores[adapter_for[(treatment, run_seed)]][metric] for run_seed in SEEDS]
    ids = sorted(set(control_maps[0]) & set(control_maps[1]) & set(treatment_maps[0]) & set(treatment_maps[1]))
    control_values = np.asarray([[control_map[item_id] for control_map in control_maps] for item_id in ids], dtype=float).mean(axis=1)
    treatment_values = np.asarray([[treatment_map[item_id] for treatment_map in treatment_maps] for item_id in ids], dtype=float).mean(axis=1)
    differences = treatment_values - control_values
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(repetitions, len(differences)))
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "metric": metric,
        "treatment": treatment,
        "control": control,
        "n_paired_examples": len(ids),
        "point_estimate": float(np.mean(differences)),
        "ci_95_lower": float(np.quantile(bootstrap_means, 0.025)),
        "ci_95_upper": float(np.quantile(bootstrap_means, 0.975)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
        "unit": "same-example difference after averaging the two evaluation seeds",
    }


def make_plot(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager

    korean_font = Path(r"C:\Windows\Fonts\malgun.ttf")
    if korean_font.exists():
        fontManager.addfont(str(korean_font))
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

    metrics = [
        ("IFEval strict", "ifeval_prompt_strict_mean", "ifeval_prompt_strict_sd"),
        ("GSM8K", "gsm8k_accuracy_mean", "gsm8k_accuracy_sd"),
        ("BBH 정확도", "bbh_accuracy_mean", "bbh_accuracy_sd"),
        ("BBH task macro", "bbh_task_macro_mean", "bbh_task_macro_sd"),
        ("BBH 최저 task", "bbh_worst_task_mean", "bbh_worst_task_sd"),
    ]
    x = np.arange(len(STRATEGIES))
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharey=True)
    axes = axes.ravel()
    for axis, (title, mean_key, sd_key) in zip(axes, metrics):
        values = [next(row[mean_key] for row in aggregate_rows if row["strategy"] == strategy) for strategy in STRATEGIES]
        errors = [next(row[sd_key] for row in aggregate_rows if row["strategy"] == strategy) for strategy in STRATEGIES]
        axis.bar(x, values, yerr=errors, capsize=3, color=["#64748b", "#0f766e", "#d97706"])
        axis.set_title(title, fontsize=9)
        axis.set_xticks(x, ["random", "quality", "diversity"], rotation=35, ha="right", fontsize=8)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("정확도")
    axes[3].set_ylabel("정확도")
    axes[-1].axis("off")
    axes[-1].text(
        0.5,
        0.52,
        "BBH 최저 task\\n모든 정책에서 0.00%",
        ha="center",
        va="center",
        fontsize=11,
        color="#183b56",
    )
    fig.suptitle("고정 토큰 영어 SFT 데이터 선택 비교", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_paired_plot(bootstrap_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager

    korean_font = Path(r"C:\Windows\Fonts\malgun.ttf")
    if korean_font.exists():
        fontManager.addfont(str(korean_font))
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

    labels = {
        "ifeval_prompt_strict": "IFEval prompt strict",
        "ifeval_instruction_strict": "IFEval instruction strict",
        "gsm8k_accuracy": "GSM8K",
        "bbh_accuracy": "BBH",
    }
    metrics = list(labels)
    treatments = [("quality", "Quality - random", "#0f766e", -0.10), ("diversity", "Diversity - random", "#d97706", 0.10)]
    fig, axis = plt.subplots(figsize=(8.8, 4.5))
    y = np.arange(len(metrics))
    for treatment, label, color, offset in treatments:
        rows = {row["metric"]: row for row in bootstrap_rows if row["treatment"] == treatment}
        points = np.array([float(rows[metric]["point_estimate"]) * 100 for metric in metrics])
        lower = np.array([float(rows[metric]["ci_95_lower"]) * 100 for metric in metrics])
        upper = np.array([float(rows[metric]["ci_95_upper"]) * 100 for metric in metrics])
        axis.errorbar(
            points,
            y + offset,
            xerr=[points - lower, upper - points],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.8,
            capsize=3,
            label=label,
        )
    axis.axvline(0, color="#374151", linewidth=1, linestyle="--")
    axis.set_yticks(y, [labels[metric] for metric in metrics])
    axis.set_xlabel("Random 대비 차이 (percentage points)")
    axis.set_title("Paired bootstrap contrast와 95% confidence interval", fontsize=12)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-dir", type=Path, default=Path("work/evaluation_final_balanced"))
    parser.add_argument("--subset-dir", type=Path, default=Path("work/evaluation_subsets_final"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/results_final"))
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--tokenizer-revision", default="ea980cb0a6c2ae4b936e82123acc929f1cec04c1")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        local_files_only=True,
    )
    validation, per_adapter_scores, bbh_tasks, global_instruction_metrics = validate_outputs(
        args.evaluation_dir, args.subset_dir, tokenizer
    )
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    per_seed_rows, aggregate_rows, metadata = aggregate_metrics(
        per_adapter_scores, bbh_tasks, global_instruction_metrics
    )
    metric_fields = [
        "adapter", "strategy", "seed", "ifeval_prompt_strict", "ifeval_prompt_loose",
        "ifeval_instruction_strict", "ifeval_instruction_loose", "gsm8k_accuracy",
        "bbh_accuracy", "bbh_task_macro", "bbh_worst_task",
    ]
    write_csv(args.output_dir / "per_seed_metrics.csv", metric_fields, per_seed_rows)
    aggregate_fields = ["strategy", "n_seeds"]
    for metric in metadata["metric_names"]:
        aggregate_fields.extend([f"{metric}_mean", f"{metric}_sd", f"{metric}_seed13", f"{metric}_seed42"])
    write_csv(args.output_dir / "strategy_metrics.csv", aggregate_fields, aggregate_rows)

    bootstrap_rows = []
    for metric in ["ifeval_prompt_strict", "ifeval_instruction_strict", "gsm8k_accuracy", "bbh_accuracy"]:
        for treatment in ["quality", "diversity"]:
            bootstrap_rows.append(paired_bootstrap(per_adapter_scores, metric, treatment, "random", args.bootstrap_repetitions, 2026))
    write_csv(
        args.output_dir / "paired_bootstrap.csv",
        ["metric", "treatment", "control", "n_paired_examples", "point_estimate", "ci_95_lower", "ci_95_upper", "bootstrap_repetitions", "bootstrap_seed", "unit"],
        bootstrap_rows,
    )

    task_rows: list[dict[str, Any]] = []
    for adapter_name, strategy, seed in ADAPTERS:
        for task, value in sorted(bbh_tasks[adapter_name]["task_scores"].items()):
            task_rows.append({"adapter": adapter_name, "strategy": strategy, "seed": seed, "task": task, "accuracy": value})
    write_csv(args.output_dir / "bbh_task_metrics.csv", ["adapter", "strategy", "seed", "task", "accuracy"], task_rows)
    (args.output_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "tokenizer": args.tokenizer,
                "tokenizer_revision": args.tokenizer_revision,
                "max_new_tokens": MAX_NEW_TOKENS,
                "selection_manifest": str(args.subset_dir / "manifest.json"),
                "bootstrap": "paired same-example bootstrap after averaging seed-specific scores",
                "validation_all_files_complete": validation["all_files_complete"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        make_plot(aggregate_rows, args.output_dir / "strategy_metrics.png")
        make_plot(aggregate_rows, args.output_dir / "strategy_metrics.pdf")
        make_paired_plot(bootstrap_rows, args.output_dir / "paired_contrast.png")
    except ImportError:
        (args.output_dir / "plot_unavailable.txt").write_text("matplotlib is not installed; regenerate the plot in an environment with matplotlib.\n", encoding="utf-8")

    print(json.dumps({"validation_all_files_complete": validation["all_files_complete"], "possible_truncation_total": validation["possible_truncation_total"], "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
