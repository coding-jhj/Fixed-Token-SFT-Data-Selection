"""Validate follow-up evaluation, subset, and fixed-token artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


EXPECTED_FINAL = {"ifeval": 192, "gsm8k": 256, "bbh": 216}
EXPECTED_EXPANDED = {"ifeval": 384, "gsm8k": 512, "bbh": 432}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            rows.append(value)
    return rows


def id_value(benchmark: str, row: dict) -> str | None:
    key = "key" if benchmark == "ifeval" else "example_id"
    value = row.get(key)
    return None if value is None else str(value)


def validate_subset(subset_dir: Path, expected: dict[str, int]) -> dict:
    report: dict[str, dict] = {}
    for benchmark, expected_rows in expected.items():
        path = subset_dir / f"{benchmark}.jsonl"
        rows = load_jsonl(path)
        ids = [id_value(benchmark, row) for row in rows]
        counts = Counter(value for value in ids if value is not None)
        report[benchmark] = {
            "path": str(path),
            "rows": len(rows),
            "expected_rows": expected_rows,
            "missing_id_count": sum(value is None for value in ids),
            "duplicate_id_count": sum(count > 1 for count in counts.values()),
            "complete": len(rows) == expected_rows and all(value is not None for value in ids) and all(count == 1 for count in counts.values()),
        }
    return report


def validate_outputs(output_dir: Path, subset_dir: Path, expected: dict[str, int]) -> dict:
    expected_ids = {
        benchmark: {id_value(benchmark, row) for row in load_jsonl(subset_dir / f"{benchmark}.jsonl")}
        for benchmark in expected
    }
    report: dict[str, dict] = {}
    for benchmark, expected_rows in expected.items():
        path = output_dir / f"{benchmark}_outputs.jsonl"
        rows = load_jsonl(path)
        ids = [id_value(benchmark, row) for row in rows]
        actual_ids = {value for value in ids if value is not None}
        counts = Counter(value for value in ids if value is not None)
        malformed = sum(
            not isinstance(row.get("response"), str)
            or not row["response"].strip()
            or not isinstance(row.get("score"), dict)
            for row in rows
        )
        report[benchmark] = {
            "path": str(path),
            "rows": len(rows),
            "expected_rows": expected_rows,
            "duplicate_id_count": sum(count > 1 for count in counts.values()),
            "missing_id_count": len(expected_ids[benchmark] - actual_ids),
            "extra_id_count": len(actual_ids - expected_ids[benchmark]),
            "malformed_row_count": malformed,
            "complete": (
                len(rows) == expected_rows
                and all(value is not None for value in ids)
                and all(count == 1 for count in counts.values())
                and actual_ids == expected_ids[benchmark]
                and malformed == 0
            ),
        }
    return report


def validate_fixed_token_summaries(paths: list[Path]) -> list[dict]:
    reports = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        reports.append(
            {
                "path": str(path),
                "strategy": value.get("strategy"),
                "seed": value.get("seed"),
                "actual_tokens": value.get("actual_tokens"),
                "target_tokens": value.get("target_tokens"),
                "exact_budget": value.get("exact_budget"),
                "valid": value.get("actual_tokens") == 1_000_000 and value.get("target_tokens") == 1_000_000 and value.get("exact_budget") is True,
            }
        )
    return reports


def validate_quality_audit(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        return {"path": str(path), "valid": False, "reason": "missing"}
    value = json.loads(path.read_text(encoding="utf-8"))
    valid = (
        value.get("audit_type", "").startswith("automatic structural")
        and value.get("human_ratings_performed") is False
        and value.get("manifest_count") == 8
        and value.get("total_manifest_rows") == 8144
        and value.get("stored_vs_recomputed_quality_mismatches", {}).get("count") == 0
        and value.get("human_audit_sample", {}).get("n") == 200
        and value.get("human_audit_sample", {}).get("found_in_manifests") == 200
    )
    return {"path": str(path), "valid": valid}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-output", type=Path, required=True)
    parser.add_argument("--base-output", type=Path)
    parser.add_argument("--expanded-output", type=Path)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--expanded-subset-dir", type=Path)
    parser.add_argument("--summary", action="append", type=Path, default=[])
    parser.add_argument("--quality-audit-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report: dict = {
        "final_subset": validate_subset(args.subset_dir, EXPECTED_FINAL),
        "seed2026": {},
        "base_model": None,
        "expanded_subset": None,
        "expanded_outputs": {},
        "fixed_token_summaries": validate_fixed_token_summaries(args.summary),
        "automatic_quality_audit": validate_quality_audit(args.quality_audit_summary),
    }
    for adapter in ["random_seed2026", "diversity_seed2026"]:
        report["seed2026"][adapter] = validate_outputs(args.seed_output / adapter, args.subset_dir, EXPECTED_FINAL)
    if args.base_output is not None and args.base_output.exists():
        report["base_model"] = validate_outputs(args.base_output / "base_model", args.subset_dir, EXPECTED_FINAL)
    if args.expanded_subset_dir is not None and args.expanded_subset_dir.exists():
        report["expanded_subset"] = validate_subset(args.expanded_subset_dir, EXPECTED_EXPANDED)
    if args.expanded_output is not None and args.expanded_output.exists() and args.expanded_subset_dir is not None and args.expanded_subset_dir.exists():
        for adapter in ["random_seed2026", "diversity_seed2026"]:
            report["expanded_outputs"][adapter] = validate_outputs(args.expanded_output / adapter, args.expanded_subset_dir, EXPECTED_EXPANDED)

    checks = []
    checks.extend(item["complete"] for item in report["final_subset"].values())
    for adapter in report["seed2026"].values():
        checks.extend(item["complete"] for item in adapter.values())
    if report["base_model"] is not None:
        checks.extend(item["complete"] for item in report["base_model"].values())
    if report["expanded_subset"] is not None:
        checks.extend(item["complete"] for item in report["expanded_subset"].values())
    for adapter in report["expanded_outputs"].values():
        checks.extend(item["complete"] for item in adapter.values())
    checks.extend(item["valid"] for item in report["fixed_token_summaries"])
    if report["automatic_quality_audit"] is not None:
        checks.append(report["automatic_quality_audit"]["valid"])
    report["all_checks_pass"] = all(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_pass": report["all_checks_pass"], "output": str(args.output)}, ensure_ascii=False))
    if not report["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
