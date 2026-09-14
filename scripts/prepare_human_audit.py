"""Prepare a blinded 200-example quality-proxy audit sheet.

This script prepares materials only. It never creates human ratings.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    unique: dict[str, dict] = {}
    for path in args.manifest:
        for row in load_jsonl(path):
            unique.setdefault(str(row["example_id"]), row)
    rows = list(unique.values())
    if len(rows) < args.n:
        raise ValueError(f"Only {len(rows)} unique rows available for {args.n} audit items")
    ordered_scores = sorted(float(row["quality_score"]) for row in rows)
    cut1 = ordered_scores[len(ordered_scores) // 3]
    cut2 = ordered_scores[(2 * len(ordered_scores)) // 3]

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        score = float(row["quality_score"])
        score_bin = "low" if score <= cut1 else "mid" if score <= cut2 else "high"
        groups[(str(row["source"]), score_bin)].append(row)
    names = sorted(groups)
    base = args.n // len(names)
    remainder = args.n - base * len(names)
    rng = random.Random(args.seed)
    counts = {name: min(base + (1 if index < remainder else 0), len(groups[name])) for index, name in enumerate(names)}
    remaining = args.n - sum(counts.values())
    while remaining:
        eligible = [name for name in names if counts[name] < len(groups[name])]
        if not eligible:
            raise ValueError("Not enough rows to fill the requested audit sample")
        rng.shuffle(eligible)
        for name in eligible:
            if not remaining:
                break
            counts[name] += 1
            remaining -= 1
    selected: list[dict] = []
    for name in names:
        selected.extend(rng.sample(groups[name], counts[name]))
    rng.shuffle(selected)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blind_path = args.output_dir / "blind_items.jsonl"
    key_path = args.output_dir / "answer_key.jsonl"
    with blind_path.open("w", encoding="utf-8", newline="\n") as blind, key_path.open("w", encoding="utf-8", newline="\n") as key:
        for index, row in enumerate(selected, start=1):
            audit_id = f"audit-{index:03d}"
            blind.write(json.dumps({"audit_id": audit_id, "text": row["text"]}, ensure_ascii=False) + "\n")
            key.write(json.dumps({"audit_id": audit_id, "example_id": row["example_id"], "source": row["source"], "length_bin": row["length_bin"], "quality_score": row["quality_score"]}, ensure_ascii=False) + "\n")

    with (args.output_dir / "sampling_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audit_id", "example_id", "source", "length_bin", "quality_bin", "quality_score"])
        writer.writeheader()
        for index, row in enumerate(selected, start=1):
            score = float(row["quality_score"])
            quality_bin = "low" if score <= cut1 else "mid" if score <= cut2 else "high"
            writer.writerow({"audit_id": f"audit-{index:03d}", "example_id": row["example_id"], "source": row["source"], "length_bin": row["length_bin"], "quality_bin": quality_bin, "quality_score": row["quality_score"]})

    rubric = (
        "# Blind quality audit rubric\n\n"
        "각 항목을 1~5점으로 평가합니다. 평가자는 `blind_items.jsonl`만 보고 평가하며 `answer_key.jsonl`과 `sampling_manifest.csv`는 보지 않습니다.\n\n"
        "- Correctness: 내용이 사실에 맞고 질문에 적절히 답하는가.\n"
        "- Instruction following: 명시된 형식·제약·요구를 지키는가.\n"
        "- Clarity: 읽기 쉽고 모호하지 않은가.\n"
        "- Self-containedness: 외부 정보 없이 답변 자체가 충분한가.\n"
        "- Overall quality: 위 항목을 종합한 전체 품질.\n\n"
        "1점은 심각한 결함, 3점은 부분적으로 유용하지만 개선 필요, 5점은 명확하고 완전한 답변입니다. 판단이 불가능하면 별도 `uncertain` 플래그를 기록합니다. 자동 점수는 사람이 평가한 뒤에만 비교합니다.\n"
    )
    (args.output_dir / "rubric.md").write_text(rubric, encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(selected), "unique_source_score_strata": len(names)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
