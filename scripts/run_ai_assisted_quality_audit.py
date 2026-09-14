"""Run a transparent, model-assisted quality audit over the blinded sample.

This is deliberately not a human-rating script.  It uses the locally cached
base model as an exploratory judge, stores only per-item scores and aggregate
diagnostics, and records the limitations needed to keep the result separate
from human agreement evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CRITERIA = (
    "correctness",
    "instruction_following",
    "clarity",
    "self_containedness",
    "overall_quality",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_prompt(user_text: str, assistant_text: str, variant: str) -> str:
    variant_note = (
        "Re-evaluate independently and pay particular attention to observable evidence; "
        "do not infer quality from length alone."
        if variant == "b"
        else "Use the rubric literally and base every score on observable evidence."
    )
    return f"""You are an evaluator for an academic data-quality audit.

{variant_note}

The item contains one user request and one assistant answer. Judge only the
assistant answer relative to the user request. Do not judge the dataset source,
the hidden heuristic score, or the writing style of the prompt itself.

Rubric: use integer scores from 1 to 5.
- correctness: factually sound and responsive; 1 = seriously wrong or irrelevant,
  3 = partly useful or has a material omission, 5 = accurate and complete.
- instruction_following: satisfies explicit constraints and requested format;
  1 = largely violates them, 3 = mixed compliance, 5 = fully compliant.
- clarity: understandable, organized, and unambiguous; 1 = confusing, 3 = usable
  with notable ambiguity, 5 = clear and well organized.
- self_containedness: sufficient explanation in the answer itself; 1 = unusable
  without missing external information, 3 = partly self-contained, 5 = self-contained.
- overall_quality: holistic score using the four criteria above.

Return exactly one JSON object and no markdown. Include short evidence phrases
that quote or point to visible features, a score for every criterion, and a
confidence between 0 and 1. Use this schema:
{{
  "scores": {{
    "correctness": 1,
    "instruction_following": 1,
    "clarity": 1,
    "self_containedness": 1,
    "overall_quality": 1
  }},
  "evidence": ["short observable evidence"],
  "confidence": 0.0,
  "uncertain": false
}}

<USER>
{user_text}
</USER>
<ASSISTANT>
{assistant_text}
</ASSISTANT>
"""


def split_messages(row: dict[str, Any]) -> tuple[str, str]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"missing messages for {row.get('example_id')}")
    user_parts = [str(item["content"]) for item in messages if item.get("role") == "user"]
    assistant_parts = [str(item["content"]) for item in messages if item.get("role") == "assistant"]
    if not user_parts or not assistant_parts:
        raise ValueError(f"missing user/assistant message for {row.get('example_id')}")
    return "\n\n".join(user_parts), "\n\n".join(assistant_parts)


def parse_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def normalize_judgment(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"parse_ok": False, "scores": {}, "confidence": None, "uncertain": True}
    raw_scores = value.get("scores")
    scores: dict[str, int] = {}
    if isinstance(raw_scores, dict):
        for criterion in CRITERIA:
            score = raw_scores.get(criterion)
            if isinstance(score, bool):
                continue
            try:
                numeric = int(score)
            except (TypeError, ValueError):
                continue
            if 1 <= numeric <= 5:
                scores[criterion] = numeric
    confidence = value.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and not 0 <= confidence <= 1:
        confidence = None
    return {
        "parse_ok": len(scores) == len(CRITERIA) and confidence is not None,
        "scores": scores,
        "confidence": confidence,
        "uncertain": bool(value.get("uncertain", False)),
    }


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            ranks[position] = average
        index = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) < 2 or len(values_a) != len(values_b):
        return None
    ranks_a = rank(values_a)
    ranks_b = rank(values_b)
    mean_a = mean(ranks_a)
    mean_b = mean(ranks_b)
    assert mean_a is not None and mean_b is not None
    numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    denominator_a = math.sqrt(sum((a - mean_a) ** 2 for a in ranks_a))
    denominator_b = math.sqrt(sum((b - mean_b) ** 2 for b in ranks_b))
    if denominator_a == 0 or denominator_b == 0:
        return None
    return numerator / (denominator_a * denominator_b)


def process_judge(
    model_path: Path,
    tokenizer_path: Path,
    items: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    output_path: Path,
    max_input_tokens: int,
    max_new_tokens: int,
    use_4bit: bool,
    resume: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError("transformers and torch are required for the model-assisted audit") from exc

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)
    model.eval()
    device = next(model.parameters()).device
    existing: dict[str, dict[str, Any]] = {}
    if resume and output_path.exists():
        for result in load_jsonl(output_path):
            if isinstance(result, dict) and result.get("audit_id"):
                existing[str(result["audit_id"])] = result
    results: list[dict[str, Any]] = list(existing.values())
    parse_counts = Counter()
    output_mode = "a" if resume and existing else "w"

    with output_path.open(output_mode, encoding="utf-8", newline="\n") as output_handle:
        for index, item in enumerate(items, start=1):
            audit_id = str(item["audit_id"])
            if audit_id in existing and existing[audit_id].get("pass_a", {}).get("parse_ok") and existing[audit_id].get("pass_b", {}).get("parse_ok"):
                continue
            example_id = str(item["example_id"])
            user_text, assistant_text = split_messages(records[example_id])
            pass_results: dict[str, Any] = {}
            for variant in ("a", "b"):
                prompt = make_prompt(user_text, assistant_text, variant)
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                continuation = generated[0, encoded["input_ids"].shape[1] :]
                decoded = tokenizer.decode(continuation, skip_special_tokens=True)
                judgment = normalize_judgment(parse_json_object(decoded))
                parse_counts[f"{variant}_{judgment['parse_ok']}"] += 1
                pass_results[f"pass_{variant}"] = judgment
            result = {"audit_id": audit_id, "example_id": example_id, **pass_results}
            results.append(result)
            output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            output_handle.flush()
            if index % 10 == 0 or index == len(items):
                print(json.dumps({"processed": index, "total": len(items), "new_results": len(results) - len(existing), "parse_counts": dict(parse_counts)}), flush=True)

    metadata = {
        "judge_model": str(model_path),
        "judge_model_type": "locally cached Qwen3-1.7B-Base; not instruction-tuned and not independent of the evaluated base model",
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "use_4bit": use_4bit,
        "resume": resume,
        "existing_results": len(existing),
        "prompt_hash_a": compact_hash(make_prompt("<user>", "<assistant>", "a")),
        "prompt_hash_b": compact_hash(make_prompt("<user>", "<assistant>", "b")),
        "parse_counts": dict(parse_counts),
    }
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results, metadata


def summarize(
    items: list[dict[str, Any]],
    records: dict[str, dict[str, Any]],
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    item_by_id = {str(item["audit_id"]): item for item in items}
    complete = [
        result
        for result in results
        if result["pass_a"]["parse_ok"] and result["pass_b"]["parse_ok"]
    ]
    criterion_summary: dict[str, Any] = {}
    for criterion in CRITERIA:
        paired = [
            (result["pass_a"]["scores"][criterion], result["pass_b"]["scores"][criterion])
            for result in complete
        ]
        differences = [abs(a - b) for a, b in paired]
        criterion_summary[criterion] = {
            "exact_agreement": sum(a == b for a, b in paired) / len(paired) if paired else None,
            "mean_absolute_difference": mean([float(value) for value in differences]),
            "mean_pass_a": mean([float(a) for a, _ in paired]),
            "mean_pass_b": mean([float(b) for _, b in paired]),
            "unique_scores": sorted({score for pair in paired for score in pair}),
            "max_score_rate": sum(score == 5 for pair in paired for score in pair) / (2 * len(paired)) if paired else None,
        }

    heuristic_by_audit: dict[str, float] = {}
    for item in items:
        record = records[str(item["example_id"])]
        heuristic_by_audit[str(item["audit_id"])] = float(record["quality_score"])
    overall_pairs = [
        (heuristic_by_audit[result["audit_id"]], (result["pass_a"]["scores"]["overall_quality"] + result["pass_b"]["scores"]["overall_quality"]) / 2.0)
        for result in complete
    ]

    grouped: dict[str, list[float]] = defaultdict(list)
    for result in complete:
        item = item_by_id[result["audit_id"]]
        source = str(item["source"])
        score = (result["pass_a"]["scores"]["overall_quality"] + result["pass_b"]["scores"]["overall_quality"]) / 2.0
        grouped[source].append(float(score))
    source_summary = {
        source: {"n": len(scores), "mean_overall_1_to_5": mean(scores)}
        for source, scores in sorted(grouped.items())
    }

    structural = {
        "items": len(items),
        "unique_example_ids": len({str(item["example_id"]) for item in items}),
        "records_found": sum(str(item["example_id"]) in records for item in items),
        "valid_user_and_assistant_messages": 0,
        "nonempty_assistant_messages": 0,
    }
    for item in items:
        record = records.get(str(item["example_id"]))
        if record is None:
            continue
        messages = record.get("messages", [])
        roles = {str(message.get("role")) for message in messages if isinstance(message, dict)}
        if "user" in roles and "assistant" in roles:
            structural["valid_user_and_assistant_messages"] += 1
        if any(str(message.get("content", "")).strip() for message in messages if message.get("role") == "assistant"):
            structural["nonempty_assistant_messages"] += 1

    return {
        "audit_type": "AI-assisted exploratory quality audit; not human rating",
        "human_ratings_performed": False,
        "n_items": len(items),
        "n_both_passes_parsed": len(complete),
        "parse_rate_both_passes": len(complete) / len(items) if items else None,
        "criterion_consistency_between_two_prompt_variants": criterion_summary,
        "source_summary": source_summary,
        "structural_checks": structural,
        "heuristic_vs_judge_overall_spearman": spearman(
            [value for value, _ in overall_pairs], [value for _, value in overall_pairs]
        ),
        "judge_overall_mean_1_to_5": mean([value for _, value in overall_pairs]),
        "parsed_overall_score_distribution": dict(Counter(value for _, value in overall_pairs)),
        "parsed_scores_degenerate_at_maximum": bool(overall_pairs) and all(value == 5 for _, value in overall_pairs),
        "limitations": [
            "No human rater participated; this cannot be reported as a human audit.",
            "The judge is the locally cached Qwen3-1.7B-Base model, which is not instruction-tuned and is not independent of the evaluated base model.",
            "The judge scores are exploratory quality-proxy evidence and do not validate the heuristic against human judgments.",
            "The two prompt variants are a consistency probe, not independent raters.",
            "Objective benchmark correctness remains determined by the frozen benchmark scorers, not by this judge.",
            "In this run only 55/200 items parsed in both passes, and 49/55 parsed overall scores were 5/5; the judge therefore showed weak score discrimination.",
        ],
        "judge_metadata": metadata,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AI-assisted quality audit (not human rating)",
        "",
        "이 문서는 실제 사람의 rating이 아닌, 로컬에 캐시된 `Qwen3-1.7B-Base`를 사용한 탐색적 quality-proxy audit입니다. 따라서 논문에서 human agreement 또는 human audit 결과로 보고하면 안 됩니다.",
        "",
        "## Scope",
        "",
        f"- Items: {summary['n_items']}",
        f"- Both prompt variants parsed: {summary['n_both_passes_parsed']} ({summary['parse_rate_both_passes']:.1%})",
        f"- Heuristic quality score와 judge overall score의 Spearman correlation: {summary['heuristic_vs_judge_overall_spearman'] if summary['heuristic_vs_judge_overall_spearman'] is not None else 'N/A'}",
        f"- Judge overall mean: {summary['judge_overall_mean_1_to_5']:.3f} / 5" if summary["judge_overall_mean_1_to_5"] is not None else "- Judge overall mean: N/A",
        "",
        "## Two-prompt consistency",
        "",
        "| Criterion | Exact agreement | Mean absolute difference | Unique parsed scores | Max-score rate |",
        "|---|---:|---:|---|---:|",
    ]
    for criterion, values in summary["criterion_consistency_between_two_prompt_variants"].items():
        lines.append(
            f"| {criterion} | {values['exact_agreement']:.1%} | {values['mean_absolute_difference']:.3f} | {values['unique_scores']} | {values['max_score_rate']:.1%} |"
            if values["exact_agreement"] is not None
            else f"| {criterion} | N/A | N/A | N/A | N/A |"
        )
    lines.extend(["", "## Parsed-judge source summary", "", "| Source | Parsed n | Mean overall (1–5) |", "|---|---:|---:|"])
    for source, values in summary["source_summary"].items():
        lines.append(f"| {source} | {values['n']} | {values['mean_overall_1_to_5']:.3f} |")
    lines.extend(
        [
            "",
            "## Structural checks",
            "",
            f"- Unique audit example IDs: {summary['structural_checks']['unique_example_ids']} / {summary['structural_checks']['items']}",
            f"- Records found in source manifests: {summary['structural_checks']['records_found']} / {summary['structural_checks']['items']}",
            f"- Rows with both user and assistant messages: {summary['structural_checks']['valid_user_and_assistant_messages']} / {summary['structural_checks']['items']}",
            f"- Rows with a non-empty assistant message: {summary['structural_checks']['nonempty_assistant_messages']} / {summary['structural_checks']['items']}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in summary["limitations"])
    lines.extend(
        [
            "",
            "Per-item judge outputs are kept in the local workspace only. The curated reproduction ZIP should contain this aggregate report and the script, not the blind text, answer key, or raw per-item evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blind-items", type=Path, required=True)
    parser.add_argument("--answer-key", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true", help="Rebuild the report from an existing per-item score file without loading the model")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit; omit for the full audit")
    args = parser.parse_args()

    items = load_jsonl(args.answer_key)
    blind_items = load_jsonl(args.blind_items)
    if len(items) != len(blind_items):
        raise ValueError(f"answer key and blind item count differ: {len(items)} != {len(blind_items)}")
    if [str(item["audit_id"]) for item in items] != [str(item["audit_id"]) for item in blind_items]:
        raise ValueError("answer key and blind item audit IDs are not aligned")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        items = items[: args.limit]

    records: dict[str, dict[str, Any]] = {}
    for manifest_path in args.manifest:
        for record in load_jsonl(manifest_path):
            records.setdefault(str(record["example_id"]), record)
    missing = [str(item["example_id"]) for item in items if str(item["example_id"]) not in records]
    if missing:
        raise ValueError(f"{len(missing)} audit examples are absent from manifests; first={missing[0]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "ai_assisted_scores.jsonl"
    if args.summarize_existing:
        if not score_path.exists():
            raise ValueError(f"existing score file is missing: {score_path}")
        results = load_jsonl(score_path)
        previous_summary_path = args.output_dir / "ai_assisted_audit_summary.json"
        previous_summary = json.loads(previous_summary_path.read_text(encoding="utf-8")) if previous_summary_path.exists() else {}
        metadata = previous_summary.get("judge_metadata", {})
        summary = summarize(items, records, results, metadata)
        summary["command"] = " ".join(sys.argv)
        summary_path = args.output_dir / "ai_assisted_audit_summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_report(args.output_dir / "ai_assisted_audit_report.md", summary)
        print(json.dumps({"summary": str(summary_path), "report": str(args.output_dir / 'ai_assisted_audit_report.md'), "scores": str(score_path), "n_items": len(items), "summarize_existing": True}, ensure_ascii=False))
        return
    results, metadata = process_judge(
        args.model_path,
        args.tokenizer_path,
        items,
        records,
        score_path,
        args.max_input_tokens,
        args.max_new_tokens,
        args.use_4bit,
        args.resume,
    )
    summary = summarize(items, records, results, metadata)
    summary["command"] = " ".join(sys.argv)
    summary_path = args.output_dir / "ai_assisted_audit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(args.output_dir / "ai_assisted_audit_report.md", summary)
    print(json.dumps({"summary": str(summary_path), "report": str(args.output_dir / 'ai_assisted_audit_report.md'), "scores": str(score_path), "n_items": len(items)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
