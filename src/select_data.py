"""Build reproducible random, quality, and quality-plus-diversity manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import MiniBatchKMeans
from transformers import AutoTokenizer


DATASET_ID = "HuggingFaceTB/smoltalk"
DATASET_REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
TOKENIZER_ID = "Qwen/Qwen3-1.7B-Base"
TOKENIZER_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
CONFIGS = (
    "smol-magpie-ultra",
    "smol-constraints",
    "smol-rewrite",
    "smol-summarize",
)
LENGTH_BINS = ((0, 256), (256, 512), (512, 1024), (1024, 1536), (1536, 2049))
ENGLISH_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "please",
    "the",
    "to",
    "what",
    "why",
    "with",
}


@dataclass(frozen=True)
class Record:
    example_id: str
    source: str
    source_row: int
    length_bin: str
    token_count: int
    quality_score: float
    text: str
    messages: list[dict[str, str]]


def stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def canonical_messages(value: Any) -> list[dict[str, str]] | None:
    if not isinstance(value, list):
        return None
    messages: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return None
        role = str(item.get("role", "")).strip().lower()
        content = item.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        content = re.sub(r"\s+", " ", content).strip()
        if not content:
            return None
        messages.append({"role": role, "content": content})
    if not messages or "user" not in {m["role"] for m in messages} or "assistant" not in {m["role"] for m in messages}:
        return None
    return messages


def formatted_token_count(tokenizer, messages: list[dict[str, str]]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    token_ids = rendered["input_ids"] if hasattr(rendered, "__getitem__") and not isinstance(rendered, list) else rendered
    if token_ids and isinstance(token_ids[0], list):
        return len(token_ids[0])
    return len(token_ids)


def formatted_token_counts(tokenizer, batch: list[list[dict[str, str]]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        batch,
        tokenize=True,
        add_generation_prompt=False,
    )
    return [len(token_ids) for token_ids in rendered["input_ids"]]


def length_bin(token_count: int) -> str:
    for lower, upper in LENGTH_BINS:
        if lower <= token_count < upper:
            return f"{lower}-{upper - 1}"
    raise ValueError(f"Unsupported token count: {token_count}")


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


def hard_valid(messages: list[dict[str, str]], token_count: int, max_length: int) -> bool:
    if token_count <= 0 or token_count > max_length:
        return False
    text = " ".join(m["content"] for m in messages)
    words = re.findall(r"[A-Za-z]+", text.lower())
    letters = re.findall(r"[A-Za-z]", text)
    if len(words) < 4 or len(letters) < 20:
        return False
    if len(letters) / max(1, len(re.findall(r"\S", text))) < 0.25:
        return False
    if "\x00" in text or re.search(r"(.)\1{12,}", text):
        return False
    return True


def load_pool(args: argparse.Namespace, tokenizer) -> tuple[dict[str, list[Record]], dict[str, int], dict[str, int], dict[str, int]]:
    heaps: dict[str, list[tuple[int, str, Record]]] = defaultdict(list)
    stratum_counts: dict[str, int] = Counter()
    stratum_tokens: dict[str, int] = Counter()
    source_rows: dict[str, int] = Counter()
    seen: set[str] = set()

    def process_batch(source: str, batch: list[tuple[int, list[dict[str, str]]]]) -> None:
        if not batch:
            return
        token_counts = formatted_token_counts(tokenizer, [messages for _, messages in batch])
        for (row_index, messages), token_count in zip(batch, token_counts):
            if not hard_valid(messages, token_count, args.max_length):
                continue
            example_id = hashlib.sha256(
                json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if example_id in seen:
                continue
            seen.add(example_id)
            source_rows[source] += 1
            stratum = f"{source}|{length_bin(token_count)}"
            stratum_counts[stratum] += 1
            stratum_tokens[stratum] += token_count
            record = Record(
                example_id=example_id,
                source=source,
                source_row=row_index,
                length_bin=length_bin(token_count),
                token_count=token_count,
                quality_score=text_quality(messages, token_count),
                text="\n".join(m["content"] for m in messages),
                messages=messages,
            )
            priority = stable_int(f"{args.pool_seed}:{example_id}")
            heap = heaps[stratum]
            item = (-priority, example_id, record)
            if len(heap) < args.candidate_cap_per_stratum:
                heapq.heappush(heap, item)
            elif priority < -heap[0][0]:
                heapq.heapreplace(heap, item)

    for source in CONFIGS:
        stream = load_dataset(
            DATASET_ID,
            source,
            split="train",
            streaming=True,
            revision=DATASET_REVISION,
        )
        batch: list[tuple[int, list[dict[str, str]]]] = []
        for row_index, row in enumerate(stream):
            if args.max_rows_per_config and row_index >= args.max_rows_per_config:
                break
            messages = canonical_messages(row.get("messages"))
            if messages is None:
                continue
            batch.append((row_index, messages))
            if len(batch) >= args.tokenize_batch_size:
                process_batch(source, batch)
                batch = []
        process_batch(source, batch)

    candidates = {
        stratum: [item[2] for item in sorted(heap, key=lambda value: (value[2].source_row, value[1]))]
        for stratum, heap in heaps.items()
    }
    return candidates, dict(stratum_counts), dict(stratum_tokens), dict(source_rows)


def token_quotas(stratum_tokens: dict[str, int], target_tokens: int) -> dict[str, int]:
    total = sum(stratum_tokens.values())
    if not total:
        return {}
    raw = {key: target_tokens * value / total for key, value in stratum_tokens.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remainder = target_tokens - sum(quotas.values())
    for key in sorted(raw, key=lambda item: (-(raw[item] - quotas[item]), item))[:remainder]:
        quotas[key] += 1
    return quotas


def random_order(records: list[Record], seed: int) -> list[Record]:
    ordered = sorted(records, key=lambda record: record.example_id)
    random.Random(seed).shuffle(ordered)
    return ordered


def quality_order(records: list[Record], quality_floor: float) -> list[Record]:
    return sorted(
        (record for record in records if record.quality_score >= quality_floor),
        key=lambda record: (-record.quality_score, stable_int(record.example_id), record.example_id),
    )


def diversity_order(
    records: list[Record],
    quality_floor: float,
    seed: int,
    embedder: SentenceTransformer,
) -> list[Record]:
    filtered = [record for record in records if record.quality_score >= quality_floor]
    if len(filtered) <= 2:
        return filtered
    embeddings = embedder.encode(
        [record.text for record in filtered],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    cluster_count = min(len(filtered), max(2, round(math.sqrt(len(filtered)))))
    labels = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=seed,
        n_init=3,
        batch_size=min(256, len(filtered)),
    ).fit_predict(np.asarray(embeddings))
    clusters: dict[int, list[Record]] = defaultdict(list)
    for record, label in zip(filtered, labels):
        clusters[int(label)].append(record)
    for cluster in clusters.values():
        cluster.sort(key=lambda record: (-record.quality_score, stable_int(f"{seed}:{record.example_id}")))
    ordered: list[Record] = []
    for cluster_id in sorted(clusters):
        clusters[cluster_id].append(None)  # sentinel to simplify round-robin termination
    while True:
        added = False
        for cluster_id in sorted(clusters):
            cluster = clusters[cluster_id]
            if cluster and cluster[0] is not None:
                ordered.append(cluster.pop(0))
                added = True
        if not added:
            return ordered


def take_under_budget(ordered: Iterable[Record], budget: int) -> tuple[list[Record], int]:
    ordered = list(ordered)
    chosen: list[Record] = []
    chosen_ids: set[str] = set()
    total = 0
    for record in ordered:
        if total + record.token_count <= budget:
            chosen.append(record)
            chosen_ids.add(record.example_id)
            total += record.token_count
        if total == budget:
            break
    if total < budget:
        remaining = sorted(
            (record for record in ordered if record.example_id not in chosen_ids),
            key=lambda record: (record.token_count, stable_int(record.example_id), record.example_id),
        )
        for record in remaining:
            if total + record.token_count <= budget:
                chosen.append(record)
                total += record.token_count
            if total == budget:
                break
    return chosen, total


def find_subset_sum(records: list[Record], target: int) -> list[Record] | None:
    """Find a deterministic small subset whose token counts sum to target."""
    if target == 0:
        return []
    candidates = sorted(
        (record for record in records if 0 < record.token_count <= target),
        key=lambda record: (record.token_count, record.example_id),
    )
    states: dict[int, tuple[int, ...]] = {0: ()}
    for index, record in enumerate(candidates):
        additions: dict[int, tuple[int, ...]] = {}
        for subtotal, chosen_indices in list(states.items()):
            new_total = subtotal + record.token_count
            if new_total > target or new_total in states or new_total in additions:
                continue
            additions[new_total] = chosen_indices + (index,)
            if new_total == target:
                return [candidates[i] for i in additions[new_total]]
        states.update(additions)
        if len(states) > 100_000:
            break
    return None


def fit_exact_budget(selected: list[Record], candidates: list[Record], target: int) -> list[Record]:
    current = sum(record.token_count for record in selected)
    if current == target:
        return selected
    if current > target:
        return selected
    selected_ids = {record.example_id for record in selected}
    remaining = [record for record in candidates if record.example_id not in selected_ids]
    residual = target - current
    direct = find_subset_sum(remaining, residual)
    if direct is not None:
        return selected + direct

    smallest_selected = sorted(selected, key=lambda record: (record.token_count, record.example_id))[:24]
    for removed in smallest_selected:
        replacement = find_subset_sum(remaining, residual + removed.token_count)
        if replacement is not None:
            removed_id = removed.example_id
            return [record for record in selected if record.example_id != removed_id] + replacement

    pair_candidates = smallest_selected[:12]
    for left_index, left in enumerate(pair_candidates):
        for right in pair_candidates[left_index + 1 :]:
            replacement = find_subset_sum(remaining, residual + left.token_count + right.token_count)
            if replacement is not None:
                removed_ids = {left.example_id, right.example_id}
                return [record for record in selected if record.example_id not in removed_ids] + replacement
    return selected


def write_manifest(out_dir: Path, strategy: str, seed: int, records: list[Record], metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"manifest_{strategy}_seed{seed}"
    with (out_dir / f"{stem}.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with (out_dir / f"{stem}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["example_id", "source", "source_row", "length_bin", "token_count", "quality_score"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow({key: getattr(record, key) for key in writer.fieldnames})
    (out_dir / f"{stem}.summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=["random", "quality", "diversity"], choices=["random", "quality", "diversity"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13])
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--target-tokens", type=int, default=100_000)
    parser.add_argument("--max-rows-per-config", type=int, default=500)
    parser.add_argument("--candidate-cap-per-stratum", type=int, default=250)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--quality-floor", type=float, default=0.55)
    parser.add_argument("--tokenize-batch-size", type=int, default=128)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out-dir", type=Path, default=Path("work/selection_smoke"))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
    args.seed = args.pool_seed
    candidates, stratum_counts, stratum_tokens, source_rows = load_pool(args, tokenizer)
    quotas = token_quotas(stratum_tokens, args.target_tokens)
    all_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        embedder = None
        for strategy in args.strategies:
            selected: list[Record] = []
            by_stratum: dict[str, list[Record]] = {}
            orders_by_stratum: dict[str, list[Record]] = {}
            for stratum, records in sorted(candidates.items()):
                if strategy == "random":
                    ordered = random_order(records, seed)
                elif strategy == "quality":
                    ordered = quality_order(records, args.quality_floor)
                else:
                    if embedder is None:
                        embedder = SentenceTransformer(args.embedding_model, device="cpu")
                    ordered = diversity_order(records, args.quality_floor, seed, embedder)
                orders_by_stratum[stratum] = ordered
                chosen, _ = take_under_budget(ordered, quotas.get(stratum, 0))
                by_stratum[stratum] = chosen
                selected.extend(chosen)
            actual_tokens = sum(record.token_count for record in selected)
            if actual_tokens < args.target_tokens:
                selected_ids = {record.example_id for record in selected}
                remaining = [
                    record
                    for stratum in sorted(orders_by_stratum)
                    for record in orders_by_stratum[stratum]
                    if record.example_id not in selected_ids
                ]
                extra, _ = take_under_budget(remaining, args.target_tokens - actual_tokens)
                selected.extend(extra)
            selected = fit_exact_budget(
                selected,
                [record for records in candidates.values() for record in records],
                args.target_tokens,
            )
            selected.sort(key=lambda record: (record.source, record.source_row, record.example_id))
            actual_tokens = sum(record.token_count for record in selected)
            metadata = {
                "strategy": strategy,
                "seed": seed,
                "pool_seed": args.pool_seed,
                "dataset": DATASET_ID,
                "dataset_revision": DATASET_REVISION,
                "tokenizer": TOKENIZER_ID,
                "tokenizer_revision": TOKENIZER_REVISION,
                "max_length": args.max_length,
                "quality_floor": args.quality_floor,
                "target_tokens": args.target_tokens,
                "actual_tokens": actual_tokens,
                "budget_error": args.target_tokens - actual_tokens,
                "exact_budget": actual_tokens == args.target_tokens,
                "candidate_cap_per_stratum": args.candidate_cap_per_stratum,
                "max_rows_per_config": args.max_rows_per_config,
                "candidate_pool_rows_by_source": source_rows,
                "candidate_pool_rows_by_stratum": stratum_counts,
                "candidate_pool_tokens_by_stratum": stratum_tokens,
                "selected_rows_by_source": dict(Counter(record.source for record in selected)),
                "selected_rows_by_stratum": dict(Counter(f"{record.source}|{record.length_bin}" for record in selected)),
                "selected_tokens_by_stratum": {
                    key: sum(record.token_count for record in records)
                    for key, records in by_stratum.items()
                },
            }
            write_manifest(args.out_dir, strategy, seed, selected, metadata)
            all_summaries.append(metadata)
    (args.out_dir / "run_summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
