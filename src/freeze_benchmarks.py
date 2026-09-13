"""Freeze the GSM8K and BBH evaluation prompts with source revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from huggingface_hub import HfApi


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_sha(repo_id: str, revision: str) -> str | None:
    try:
        return HfApi().dataset_info(repo_id, revision=revision).sha
    except Exception as exc:  # pragma: no cover - metadata fallback for offline runs
        print(f"warning: could not resolve {repo_id}@{revision}: {exc}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("work/evaluation_sets"))
    parser.add_argument("--gsm8k-revision", default="main")
    parser.add_argument("--bbh-revision", default="main")
    args = parser.parse_args()

    gsm8k = load_dataset("openai/gsm8k", "main", split="test", revision=args.gsm8k_revision)
    gsm_rows = [
        {
            "benchmark": "gsm8k",
            "subset": "main",
            "example_id": f"gsm8k-main-test-{index:04d}",
            "prompt": row["question"],
            "target": row["answer"],
        }
        for index, row in enumerate(gsm8k)
    ]

    bbh_configs = sorted(get_dataset_config_names("lukaemon/bbh", revision=args.bbh_revision))
    bbh_rows: list[dict] = []
    for config in bbh_configs:
        dataset = load_dataset("lukaemon/bbh", config, split="test", revision=args.bbh_revision)
        for index, row in enumerate(dataset):
            bbh_rows.append(
                {
                    "benchmark": "bbh",
                    "subset": config,
                    "example_id": f"bbh-{config}-test-{index:04d}",
                    "prompt": row["input"],
                    "target": row["target"],
                }
            )

    write_jsonl(args.out_dir / "gsm8k_test.jsonl", gsm_rows)
    write_jsonl(args.out_dir / "bbh_test.jsonl", bbh_rows)
    metadata = {
        "gsm8k": {
            "repo_id": "openai/gsm8k",
            "requested_revision": args.gsm8k_revision,
            "resolved_revision": dataset_sha("openai/gsm8k", args.gsm8k_revision),
            "config": "main",
            "split": "test",
            "rows": len(gsm_rows),
        },
        "bbh": {
            "repo_id": "lukaemon/bbh",
            "requested_revision": args.bbh_revision,
            "resolved_revision": dataset_sha("lukaemon/bbh", args.bbh_revision),
            "configs": bbh_configs,
            "split": "test",
            "rows": len(bbh_rows),
        },
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
