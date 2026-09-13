"""Check frozen GSM8K and BBH prompts against selected training manifests."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import difflib
import hashlib
import json
import re
from pathlib import Path


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, width: int = 5) -> set[str]:
    if len(text) <= width:
        return {text}
    return {text[index : index + width] for index in range(len(text) - width + 1)}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def manifest_prompt(row: dict) -> str:
    return "\n".join(message["content"] for message in row["messages"] if message.get("role") == "user")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--gsm8k", type=Path, required=True)
    parser.add_argument("--bbh", type=Path, required=True)
    parser.add_argument("--near-threshold", type=float, default=0.92)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    eval_rows = load_jsonl(args.gsm8k) + load_jsonl(args.bbh)
    eval_prompts = [normalize(row["prompt"]) for row in eval_rows]
    eval_hashes = {hashlib.sha256(prompt.encode("utf-8")).hexdigest(): index for index, prompt in enumerate(eval_prompts)}
    eval_shingles = [shingles(prompt) for prompt in eval_prompts]
    inverted_index: dict[str, set[int]] = defaultdict(set)
    for index, prompt_shingles in enumerate(eval_shingles):
        for shingle in prompt_shingles:
            inverted_index[shingle].add(index)

    report = {
        "evaluation_sets": {
            "gsm8k": {"path": str(args.gsm8k), "rows": len(load_jsonl(args.gsm8k))},
            "bbh": {"path": str(args.bbh), "rows": len(load_jsonl(args.bbh))},
        },
        "near_threshold": args.near_threshold,
        "manifests": {},
    }
    for manifest_path in args.manifests:
        rows = load_jsonl(manifest_path)
        exact: list[dict] = []
        near: list[dict] = []
        for row in rows:
            prompt = normalize(manifest_prompt(row))
            prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            if prompt_hash in eval_hashes:
                eval_index = eval_hashes[prompt_hash]
                exact.append(
                    {
                        "example_id": row["example_id"],
                        "benchmark": eval_rows[eval_index]["benchmark"],
                        "eval_example_id": eval_rows[eval_index]["example_id"],
                    }
                )
                continue
            prompt_shingles = shingles(prompt)
            candidate_votes = Counter(
                candidate_index
                for shingle in prompt_shingles
                for candidate_index in inverted_index.get(shingle, ())
            )
            best_ratio = 0.0
            best_index = None
            for index, _votes in candidate_votes.most_common(100):
                eval_prompt = eval_prompts[index]
                length_ratio = max(len(prompt), len(eval_prompt)) / max(1, min(len(prompt), len(eval_prompt)))
                if length_ratio > 1.5:
                    continue
                jaccard = len(prompt_shingles & eval_shingles[index]) / max(1, len(prompt_shingles | eval_shingles[index]))
                if jaccard < 0.35:
                    continue
                ratio = difflib.SequenceMatcher(None, prompt, eval_prompt, autojunk=False).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_index = index
            if best_ratio >= args.near_threshold:
                near.append(
                    {
                        "example_id": row["example_id"],
                        "benchmark": eval_rows[best_index]["benchmark"],
                        "eval_example_id": eval_rows[best_index]["example_id"],
                        "ratio": round(best_ratio, 6),
                    }
                )
        by_benchmark = {}
        for benchmark in ("gsm8k", "bbh"):
            by_benchmark[benchmark] = {
                "exact_overlap_count": sum(item["benchmark"] == benchmark for item in exact),
                "near_overlap_count": sum(item["benchmark"] == benchmark for item in near),
            }
        report["manifests"][str(manifest_path)] = {
            "training_count": len(rows),
            "exact_overlap_count": len(exact),
            "near_overlap_count": len(near),
            "by_benchmark": by_benchmark,
            "exact_matches": exact,
            "near_matches": near,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
