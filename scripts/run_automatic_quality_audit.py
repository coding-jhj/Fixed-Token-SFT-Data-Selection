"""Audit selected training manifests with reproducible structural heuristics.

This is an automatic data-quality audit, not a human rating.  It re-computes
the selection quality heuristic, checks message structure, and reports
selection overlap and suspicious surface patterns without copying raw text to
the report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any


ENGLISH_WORDS = {
    "a", "an", "and", "are", "for", "from", "how", "in", "is", "of",
    "please", "the", "to", "what", "why", "with",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def text_quality(messages: list[dict[str, str]], token_count: int) -> float:
    text = " ".join(m["content"] for m in messages)
    words = re.findall(r"[A-Za-z]+", text.lower())
    if not words:
        return 0.0
    english_signal = min(1.0, sum(w in ENGLISH_WORDS for w in words) / 4.0)
    letters = re.findall(r"[A-Za-z]", text)
    ascii_ratio = len(letters) / max(1, len(re.findall(r"\S", text)))
    english_score = 0.5 * english_signal + 0.5 * min(1.0, ascii_ratio)
    assistant_text = " ".join(m["content"] for m in messages if m["role"] == "assistant")
    response_score = min(1.0, len(re.findall(r"\w+", assistant_text)) / 80.0)
    length_score = 1.0 if 32 <= token_count <= 1536 else 0.7
    repeated = sum(count - 1 for count in Counter(words).values() if count > 3)
    repetition_score = max(0.0, 1.0 - repeated / max(1, len(words)))
    artifact_score = 0.0 if ("\x00" in text or re.search(r"(.)\1{9,}", text)) else 1.0
    return round(
        0.25 * english_score
        + 0.25 * response_score
        + 0.20 * length_score
        + 0.20 * repetition_score
        + 0.10 * artifact_score,
        6,
    )


def structural_flags(row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    flags = {
        "messages_is_list": isinstance(messages, list),
        "valid_message_items": False,
        "has_user": False,
        "has_assistant": False,
        "assistant_last": False,
        "nonempty_assistant": False,
        "null_byte": False,
        "long_repeated_character_run": False,
        "short_assistant_word_count": False,
        "recomputed_quality_matches": False,
    }
    if not isinstance(messages, list):
        return flags
    flags["valid_message_items"] = all(
        isinstance(message, dict)
        and str(message.get("role", "")).lower() in {"system", "user", "assistant"}
        and isinstance(message.get("content"), str)
        and bool(message.get("content", "").strip())
        for message in messages
    )
    roles = [str(message.get("role", "")).lower() for message in messages if isinstance(message, dict)]
    flags["has_user"] = "user" in roles
    flags["has_assistant"] = "assistant" in roles
    flags["assistant_last"] = bool(roles) and roles[-1] == "assistant"
    assistant_text = " ".join(
        str(message.get("content", "")) for message in messages if isinstance(message, dict) and message.get("role") == "assistant"
    )
    all_text = " ".join(str(message.get("content", "")) for message in messages if isinstance(message, dict))
    flags["nonempty_assistant"] = bool(assistant_text.strip())
    flags["null_byte"] = "\x00" in all_text
    flags["long_repeated_character_run"] = bool(re.search(r"(.)\1{9,}", all_text))
    flags["short_assistant_word_count"] = len(re.findall(r"\w+", assistant_text)) < 8
    return flags


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values) if values else None,
        "median": median(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--audit-key", type=Path, default=None)
    parser.add_argument("--audit-sampling-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    rows_by_id: dict[str, list[str]] = defaultdict(list)
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_flags: Counter[str] = Counter()
    quality_by_manifest: dict[str, list[float]] = {}
    quality_by_source: dict[str, list[float]] = defaultdict(list)
    score_mismatches: list[dict[str, Any]] = []

    for path in args.manifest:
        name = path.stem
        rows = load_jsonl(path)
        manifest_rows[name] = rows
        qualities: list[float] = []
        for row in rows:
            example_id = str(row.get("example_id", ""))
            rows_by_id[example_id].append(name)
            source = str(row.get("source", "unknown"))
            source_rows[source].append(row)
            flags = structural_flags(row)
            all_flags.update(key for key, value in flags.items() if value)
            qualities.append(float(row.get("quality_score", 0.0)))
            quality_by_source[source].append(float(row.get("quality_score", 0.0)))
            token_count = int(row.get("token_count", 0))
            messages = row.get("messages")
            if isinstance(messages, list):
                recomputed = text_quality(messages, token_count)
                if abs(recomputed - float(row.get("quality_score", 0.0))) > 1e-9:
                    score_mismatches.append({"manifest": name, "example_id": example_id, "stored": row.get("quality_score"), "recomputed": recomputed})
            flags["recomputed_quality_matches"] = True
        quality_by_manifest[name] = qualities

    duplicate_ids = {example_id: names for example_id, names in rows_by_id.items() if len(names) > 1}
    selection_overlap: dict[str, dict[str, int]] = {}
    manifest_sets = {name: {str(row.get("example_id")) for row in rows} for name, rows in manifest_rows.items()}
    for name, ids in manifest_sets.items():
        selection_overlap[name] = {other: len(ids & other_ids) for other, other_ids in manifest_sets.items()}

    audit_summary: dict[str, Any] = {}
    if args.audit_key is not None:
        audit_rows = load_jsonl(args.audit_key)
        sampling_rows = {str(row.get("audit_id")): row for row in audit_rows}
        if args.audit_sampling_manifest is not None:
            with args.audit_sampling_manifest.open(encoding="utf-8", newline="") as handle:
                sampling_rows = {str(row["audit_id"]): row for row in csv.DictReader(handle)}
        audit_summary = {
            "n": len(sampling_rows),
            "unique_example_ids": len({str(row.get("example_id")) for row in sampling_rows.values()}),
            "found_in_manifests": sum(str(row.get("example_id")) in rows_by_id for row in sampling_rows.values()),
            "source_counts": dict(Counter(str(row.get("source")) for row in sampling_rows.values())),
            "quality_bin_counts": dict(Counter(str(row.get("quality_bin")) for row in sampling_rows.values())),
        }

    summary = {
        "audit_type": "automatic structural and heuristic audit; not human rating",
        "human_ratings_performed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_count": len(manifest_rows),
        "manifests": {name: {"rows": len(rows), "quality_score": stats(quality_by_manifest[name])} for name, rows in sorted(manifest_rows.items())},
        "total_manifest_rows": sum(len(rows) for rows in manifest_rows.values()),
        "unique_example_ids_across_manifests": len({example_id for example_id in rows_by_id if example_id}),
        "duplicate_example_ids_across_manifests": len(duplicate_ids),
        "duplicate_example_id_examples": dict(list(sorted(duplicate_ids.items()))[:20]),
        "source_summary": {
            source: {"rows": len(rows), "quality_score": stats(quality_by_source[source])}
            for source, rows in sorted(source_rows.items())
        },
        "selection_overlap_counts": selection_overlap,
        "true_flag_counts": dict(sorted(all_flags.items())),
        "stored_vs_recomputed_quality_mismatches": {
            "count": len(score_mismatches),
            "first_examples": score_mismatches[:20],
        },
        "human_audit_sample": audit_summary,
        "limitations": [
            "Surface and heuristic checks cannot determine factual correctness or usefulness for every item.",
            "A human rater did not participate; this report cannot be called a human audit.",
            "Repeated content, short responses, or a non-final assistant role are review flags rather than automatic failures.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "automatic_quality_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Automatic quality audit (not human rating)",
        "",
        "이 보고서는 선택 manifest 전체에 대해 구조·표면 패턴·기존 quality heuristic 재현성을 점검한 자동 audit입니다. 실제 사람의 rating이 아니므로 human audit 또는 factual-quality validation으로 해석하지 않습니다.",
        "",
        "## Coverage",
        "",
        f"- Manifests: {summary['manifest_count']}",
        f"- Manifest rows: {summary['total_manifest_rows']}",
        f"- Unique example IDs: {summary['unique_example_ids_across_manifests']}",
        f"- IDs appearing in more than one manifest: {summary['duplicate_example_ids_across_manifests']}",
        "",
        "## Per-manifest quality-score distribution",
        "",
        "| Manifest | n | Mean | Median | Min | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in sorted(summary["manifests"].items()):
        score = values["quality_score"]
        lines.append(f"| {name} | {score['n']} | {score['mean']:.6f} | {score['median']:.6f} | {score['min']:.6f} | {score['max']:.6f} |")
    lines.extend(["", "## Structural and surface flags", "", "| Flag | Count |", "|---|---:|"])
    for flag, count in sorted(summary["true_flag_counts"].items()):
        lines.append(f"| {flag} | {count} |")
    lines.extend(
        [
            "",
            "## Heuristic reproducibility",
            "",
            f"- Stored quality scores that differ from a local re-computation: {summary['stored_vs_recomputed_quality_mismatches']['count']}",
            "- Selection overlap is reported as counts only; overlap is expected across policies and seeds and is not a quality failure.",
            "",
            "## Blind sample coverage",
            "",
        ]
    )
    if audit_summary:
        lines.extend(
            [
                f"- Audit-key rows: {audit_summary['n']}",
                f"- Unique IDs: {audit_summary['unique_example_ids']}",
                f"- Found in manifests: {audit_summary['found_in_manifests']}",
                f"- Source counts: {json.dumps(audit_summary['source_counts'], ensure_ascii=False, sort_keys=True)}",
                f"- Quality-bin counts: {json.dumps(audit_summary['quality_bin_counts'], ensure_ascii=False, sort_keys=True)}",
            ]
        )
    else:
        lines.append("- No audit-key was supplied.")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.append("")
    (args.output_dir / "automatic_quality_audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "report": str(args.output_dir / 'automatic_quality_audit_report.md'), "manifest_rows": summary["total_manifest_rows"], "unique_ids": summary["unique_example_ids_across_manifests"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
