"""Create deterministic, balanced evaluation subsets for the T10 study."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_IFEVAL = Path("work/evaluation_sets/ifeval_train.jsonl")
DEFAULT_GSM8K = Path("work/evaluation_sets/gsm8k_test.jsonl")
DEFAULT_BBH = Path("work/evaluation_sets/bbh_test.jsonl")
DEFAULT_OUTPUT = Path("work/evaluation_subsets")
IFEVAL_CACHE_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
BBH_REVISION = "982bb89fd79532a8ac676a61fc42eb1aeec63f99"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def allocate_counts(group_sizes: dict[str, int], total: int) -> dict[str, int]:
    if total > sum(group_sizes.values()):
        raise ValueError("Requested subset is larger than the available rows")
    names = sorted(group_sizes)
    base = {name: 0 for name in names}
    remaining = total
    while remaining:
        eligible = [name for name in names if base[name] < group_sizes[name]]
        if not eligible:
            raise ValueError("Could not allocate the requested subset")
        for name in eligible:
            if not remaining:
                break
            base[name] += 1
            remaining -= 1
    return base


def sample_groups(
    rows: list[dict], groups: dict[str, list[int]], total: int, seed: int
) -> list[int]:
    counts = allocate_counts({name: len(indices) for name, indices in groups.items()}, total)
    selected: list[int] = []
    for group_number, name in enumerate(sorted(groups)):
        rng = random.Random(seed + group_number * 1009)
        selected.extend(rng.sample(groups[name], counts[name]))
    return sorted(selected)


def select_ifeval(rows: list[dict], total: int, seed: int) -> list[int]:
    if total < len({item for row in rows for item in row["instruction_id_list"]}):
        raise ValueError("IFEval subset must cover every instruction family")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        for instruction_id in row["instruction_id_list"]:
            groups[instruction_id].append(index)

    rng = random.Random(seed)
    selected: set[int] = set()
    covered: defaultdict[str, int] = defaultdict(int)

    # Cover every instruction family once before filling the remaining budget.
    for instruction_id in sorted(groups):
        candidates = [index for index in groups[instruction_id] if index not in selected]
        index = max(
            candidates,
            key=lambda value: (
                sum(other not in covered for other in rows[value]["instruction_id_list"]),
                rng.random(),
            ),
        )
        selected.add(index)
        for other in rows[index]["instruction_id_list"]:
            covered[other] += 1

    while len(selected) < total:
        candidates = [index for index in range(len(rows)) if index not in selected]
        index = max(
            candidates,
            key=lambda value: (
                sum(1.0 / (1 + covered[other]) for other in rows[value]["instruction_id_list"]),
                rng.random(),
            ),
        )
        selected.add(index)
        for other in rows[index]["instruction_id_list"]:
            covered[other] += 1
    return sorted(selected)


def length_groups(rows: list[dict], number_of_groups: int) -> dict[str, list[int]]:
    ordered = sorted(range(len(rows)), key=lambda index: (len(rows[index]["prompt"]), index))
    groups: dict[str, list[int]] = defaultdict(list)
    for rank, index in enumerate(ordered):
        group = min(number_of_groups - 1, rank * number_of_groups // len(rows))
        groups[f"length_q{group + 1}"] .append(index)
    return dict(groups)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifeval-jsonl", type=Path, default=DEFAULT_IFEVAL)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--bbh-jsonl", type=Path, default=DEFAULT_BBH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--ifeval-n", type=int, default=192)
    parser.add_argument("--gsm8k-n", type=int, default=256)
    parser.add_argument("--bbh-per-task", type=int, default=8)
    args = parser.parse_args()

    ifeval = load_jsonl(args.ifeval_jsonl)
    gsm8k = load_jsonl(args.gsm8k_jsonl)
    bbh = load_jsonl(args.bbh_jsonl)

    ifeval_indices = select_ifeval(ifeval, args.ifeval_n, args.seed)
    gsm8k_indices = sample_groups(gsm8k, length_groups(gsm8k, 4), args.gsm8k_n, args.seed + 1)
    bbh_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(bbh):
        bbh_groups[row["subset"]].append(index)
    bbh_indices = [
        index
        for group_number, name in enumerate(sorted(bbh_groups))
        for index in sorted(random.Random(args.seed + 100 + group_number).sample(bbh_groups[name], args.bbh_per_task))
    ]

    selected = {
        "ifeval": [ifeval[index] for index in ifeval_indices],
        "gsm8k": [gsm8k[index] for index in gsm8k_indices],
        "bbh": [bbh[index] for index in bbh_indices],
    }
    paths = {
        "ifeval": args.output_dir / "ifeval.jsonl",
        "gsm8k": args.output_dir / "gsm8k.jsonl",
        "bbh": args.output_dir / "bbh.jsonl",
    }
    for name, rows in selected.items():
        write_jsonl(paths[name], rows)

    manifest = {
        "selection_seed": args.seed,
        "method": {
            "ifeval": "coverage-first greedy stratification over instruction_id_list",
            "gsm8k": "deterministic four-quantile prompt-length stratification",
            "bbh": "deterministic equal allocation per task",
        },
        "sources": {
            "ifeval": {"repo_id": "google/IFEval", "revision": IFEVAL_CACHE_REVISION, "rows": len(ifeval)},
            "gsm8k": {"repo_id": "openai/gsm8k", "revision": GSM8K_REVISION, "config": "main", "split": "test", "rows": len(gsm8k)},
            "bbh": {"repo_id": "lukaemon/bbh", "revision": BBH_REVISION, "split": "test", "rows": len(bbh)},
        },
        "selected_rows": {name: len(rows) for name, rows in selected.items()},
        "selected_ids": {
            "ifeval": [row["key"] for row in selected["ifeval"]],
            "gsm8k": [row["example_id"] for row in selected["gsm8k"]],
            "bbh": [row["example_id"] for row in selected["bbh"]],
        },
        "files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "selected_rows": manifest["selected_rows"]}, indent=2))


if __name__ == "__main__":
    main()
