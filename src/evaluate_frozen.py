"""Run deterministic evaluation on the frozen IFEval, GSM8K, and BBH sets."""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
import types
import unicodedata
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def install_langdetect_fallback() -> None:
    """Keep IFEval scoring usable when the optional langdetect package is absent."""
    try:
        import langdetect  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class LangDetectException(Exception):
        pass

    markers = {
        "de": {"der", "die", "das", "und", "ist", "nicht", "ein"},
        "it": {"il", "lo", "la", "gli", "che", "una", "non"},
        "pt": {"que", "não", "uma", "para", "com", "dos", "das"},
        "fi": {"että", "ja", "on", "sekä", "tämä", "ovat"},
        "vi": {"và", "của", "là", "một", "những", "trong"},
        "sw": {"na", "ya", "kwa", "wa", "ni", "hii"},
    }

    def detect(text: str) -> str:
        letters = [char for char in text if char.isalpha()]
        if not letters:
            raise LangDetectException("No letters in text")
        if any("가" <= char <= "힣" for char in letters):
            return "ko"
        if any("ก" <= char <= "๛" for char in letters):
            return "th"
        if any("Ѐ" <= char <= "я" for char in letters):
            return "bg" if any(char in text for char in "ъьщю") else "ru"
        if any("अ" <= char <= "ॿ" for char in letters):
            return "hi"
        if any("ಕ" <= char <= "ೞ" for char in letters):
            return "kn"
        if any("म" <= char <= "ॿ" for char in letters):
            return "mr"
        if any("ਅ" <= char <= "ਲ਼" for char in letters):
            return "pa"
        if any("અ" <= char <= "ૹ" for char in letters):
            return "gu"
        if any("త" <= char <= "ఌ" for char in letters):
            return "te"
        if any("அ" <= char <= "௺" for char in letters):
            return "ta"
        if any("অ" <= char <= "৾" for char in letters):
            return "bn"
        if any("अ" <= char <= "ॿ" for char in letters):
            return "ne"
        if any("ء" <= char <= "ۿ" for char in letters):
            return "fa" if any(char in text for char in "پچژگ") else "ur"
        tokens = set(re.findall(r"[a-zÀ-ÿ]+", text.casefold()))
        marker_scores = {language: len(tokens & words) for language, words in markers.items()}
        best_language, best_score = max(marker_scores.items(), key=lambda item: item[1])
        return best_language if best_score else "en"

    sys.modules["langdetect"] = types.SimpleNamespace(
        detect=detect,
        LangDetectException=LangDetectException,
    )


def install_ifeval_dependency_fallbacks() -> None:
    """Provide offline fallbacks for optional IFEval scoring dependencies."""
    try:
        import immutabledict  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["immutabledict"] = types.SimpleNamespace(
            immutabledict=lambda value: dict(value)
        )

    try:
        import nltk
    except ModuleNotFoundError:
        return

    # IFEval only needs tokenization and sentence splitting; avoid a network
    # download when the punkt_tab resource is absent in the offline environment.
    nltk.download = lambda *args, **kwargs: False
    nltk.word_tokenize = lambda value: re.findall(r"\w+|[^\w\s]", value, re.UNICODE)
    original_load = nltk.data.load

    def load_resource(resource, *args, **kwargs):
        if resource == "nltk:tokenizers/punkt/english.pickle":
            return nltk.tokenize.PunktSentenceTokenizer()
        return original_load(resource, *args, **kwargs)

    nltk.data.load = load_resource


install_langdetect_fallback()
install_ifeval_dependency_fallbacks()
from lm_eval.tasks.ifeval import utils as ifeval_utils


DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DEFAULT_TOKENIZER_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DEFAULT_IFEVAL = Path("work/evaluation_sets/ifeval_train.jsonl")
DEFAULT_GSM8K = Path("work/evaluation_sets/gsm8k_test.jsonl")
DEFAULT_BBH = Path("work/evaluation_sets/bbh_test.jsonl")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_completed_outputs(path: Path) -> list[dict]:
    """Load valid completed rows and truncate a possibly interrupted final line."""
    if not path.exists():
        return []
    rows: list[dict] = []
    valid_end = 0
    with path.open("rb") as handle:
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                valid_end = handle.tell()
                break
            if not raw_line.strip():
                valid_end = handle.tell()
                continue
            try:
                rows.append(json.loads(raw_line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                valid_end = line_start
                break
            valid_end = handle.tell()
    if valid_end < path.stat().st_size:
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
    return rows


def build_model(model_id: str, model_revision: str, tokenizer_revision: str, adapter: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=model_revision,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tokenizer


def chat_inputs(tokenizer, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    encoded_rows = []
    for prompt in prompts:
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        encoded_rows.append(input_ids)
    max_length = max(len(row) for row in encoded_rows)
    pad_id = tokenizer.pad_token_id
    input_ids = torch.tensor(
        [[pad_id] * (max_length - len(row)) + row for row in encoded_rows], dtype=torch.long
    )
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    input_ids, attention_mask = chat_inputs(tokenizer, prompts)
    input_ids = input_ids.to(model.device)
    attention_mask = attention_mask.to(model.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[:, input_ids.shape[1] :]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True)


def ifeval_score(row: dict, response: str) -> dict:
    score = ifeval_utils.process_results(row, [response])
    return {
        "prompt_level_strict": bool(score["prompt_level_strict_acc"]),
        "instruction_level_strict": [bool(value) for value in score["inst_level_strict_acc"]],
        "prompt_level_loose": bool(score["prompt_level_loose_acc"]),
        "instruction_level_loose": [bool(value) for value in score["inst_level_loose_acc"]],
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def extract_gsm8k_answer(text: str) -> str | None:
    matches = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
    if matches:
        return matches[-1].replace(",", "").strip()
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return matches[-1].replace(",", "").strip() if matches else None


def gsm8k_score(target: str, response: str) -> bool:
    target_answer = extract_gsm8k_answer(target)
    response_answer = extract_gsm8k_answer(response)
    return target_answer is not None and target_answer == response_answer


def extract_bbh_answer(target: str, response: str) -> str | None:
    target = target.strip()
    if re.fullmatch(r"\([A-Z]\)", target):
        choices = re.findall(r"\([A-Z]\)", response)
        if choices:
            return choices[-1]
        labeled = re.findall(r"(?:answer|choice|option|final)\s*(?:is|:)?\s*\(?([A-Z])\)?", response, re.I)
        return f"({labeled[-1].upper()})" if labeled else None
    if target in {"True", "False"}:
        choices = re.findall(r"\b(True|False)\b", response, re.I)
        return choices[-1].capitalize() if choices else None
    target_norm = normalize_text(target).strip(" .,:;!?\n")
    if target_norm and target_norm in normalize_text(response):
        return target
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1].strip(" .") if lines else None


def bbh_score(target: str, response: str) -> bool:
    extracted = extract_bbh_answer(target, response)
    return extracted is not None and normalize_text(extracted).strip(" .,:;!?\n") == normalize_text(target).strip(" .,:;!?\n")


def evaluation_prompt(benchmark: str, prompt: str) -> str:
    if benchmark == "gsm8k":
        return prompt + "\n\nSolve the problem. End your final numeric answer with #### followed by the number."
    if benchmark == "bbh":
        return prompt + "\n\nReturn only the final answer. Do not add an explanation."
    return prompt


def evaluate_benchmark(
    model,
    tokenizer,
    benchmark: str,
    rows: list[dict],
    max_new_tokens: int,
    batch_size: int,
    output_path: Path,
    resume: bool = False,
) -> dict:
    started = time.time()
    completed_rows = load_completed_outputs(output_path) if resume else []
    if len(completed_rows) > len(rows):
        raise ValueError(f"Existing output has more rows than the input: {output_path}")

    def row_id(row: dict, index: int) -> str:
        return str(row.get("example_id", row.get("key", index)))

    for index, stored in enumerate(completed_rows):
        expected = row_id(rows[index], index)
        actual = row_id(stored, index)
        if expected != actual:
            raise ValueError(
                f"Resume row mismatch at index {index}: expected {expected}, found {actual}"
            )

    scores: list[dict] = []
    for index, stored in enumerate(completed_rows):
        score = stored["score"]
        if benchmark == "ifeval":
            correct = score["prompt_level_strict"]
        elif benchmark == "gsm8k":
            correct = score["exact_match"]
        else:
            correct = score["normalized_accuracy"]
        scores.append({"example_id": row_id(rows[index], index), "correct": bool(correct), **score})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed_rows else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for start in range(len(completed_rows), len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = [evaluation_prompt(benchmark, row["prompt"]) for row in batch_rows]
            responses = generate_batch(model, tokenizer, prompts, max_new_tokens)
            for offset, (row, response) in enumerate(zip(batch_rows, responses), start=1):
                index = start + offset
                if benchmark == "ifeval":
                    score = ifeval_score(row, response)
                    correct = score["prompt_level_strict"]
                elif benchmark == "gsm8k":
                    score = {"exact_match": gsm8k_score(row["target"], response)}
                    correct = score["exact_match"]
                else:
                    score = {"normalized_accuracy": bbh_score(row["target"], response)}
                    correct = score["normalized_accuracy"]
                handle.write(json.dumps({**row, "response": response, "score": score}, ensure_ascii=False) + "\n")
                scores.append({"example_id": row_id(row, index), "correct": bool(correct), **score})
                handle.flush()
            completed = min(start + batch_size, len(rows))
            if completed == batch_size or completed % 25 == 0 or completed == len(rows):
                print(json.dumps({"benchmark": benchmark, "completed": completed, "total": len(rows)}), flush=True)

    summary: dict = {
        "benchmark": benchmark,
        "rows": len(rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "outputs": str(output_path),
        "resumed_rows": len(completed_rows),
    }
    if benchmark == "ifeval":
        summary["prompt_level_strict_accuracy"] = sum(item["prompt_level_strict"] for item in scores) / len(scores)
        summary["prompt_level_loose_accuracy"] = sum(item["prompt_level_loose"] for item in scores) / len(scores)
        strict_items = [value for item in scores for value in item["instruction_level_strict"]]
        loose_items = [value for item in scores for value in item["instruction_level_loose"]]
        summary["instruction_level_strict_accuracy"] = sum(strict_items) / len(strict_items)
        summary["instruction_level_loose_accuracy"] = sum(loose_items) / len(loose_items)
    else:
        summary["accuracy"] = sum(item["correct"] for item in scores) / len(scores)
        if benchmark == "bbh":
            by_subset: dict[str, list[bool]] = {}
            for row, item in zip(rows, scores):
                by_subset.setdefault(row["subset"], []).append(item["correct"])
            summary["by_subset"] = {key: sum(values) / len(values) for key, values in sorted(by_subset.items())}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", action="append", required=True, help="name=path; repeat for each adapter")
    parser.add_argument("--benchmarks", nargs="+", choices=["ifeval", "gsm8k", "bbh"], default=["ifeval", "gsm8k", "bbh"])
    parser.add_argument("--ifeval-jsonl", type=Path, default=DEFAULT_IFEVAL)
    parser.add_argument("--gsm8k-jsonl", type=Path, default=DEFAULT_GSM8K)
    parser.add_argument("--bbh-jsonl", type=Path, default=DEFAULT_BBH)
    parser.add_argument("--output-dir", type=Path, default=Path("work/evaluation_runs"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--ifeval-max-new-tokens", type=int, default=512)
    parser.add_argument("--benchmark-max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true", help="resume valid rows already present in output files")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen evaluation")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    datasets = {
        "ifeval": load_jsonl(args.ifeval_jsonl),
        "gsm8k": load_jsonl(args.gsm8k_jsonl),
        "bbh": load_jsonl(args.bbh_jsonl),
    }
    if args.limit is not None:
        datasets = {key: rows[: args.limit] for key, rows in datasets.items()}
    adapters = {}
    for value in args.adapter:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Adapter must use name=path syntax: {value}")
        adapters[name] = Path(path)

    all_summaries = {}
    for name, adapter in adapters.items():
        print(json.dumps({"adapter": name, "status": "loading"}), flush=True)
        model, tokenizer = build_model(args.model, args.model_revision, args.tokenizer_revision, adapter)
        adapter_dir = args.output_dir / name
        adapter_dir.mkdir(parents=True, exist_ok=True)
        summaries = {}
        for benchmark in args.benchmarks:
            max_new_tokens = args.ifeval_max_new_tokens if benchmark == "ifeval" else args.benchmark_max_new_tokens
            output_path = adapter_dir / f"{benchmark}_outputs.jsonl"
            summaries[benchmark] = evaluate_benchmark(
                model,
                tokenizer,
                benchmark,
                datasets[benchmark],
                max_new_tokens,
                args.batch_size,
                output_path,
                resume=args.resume,
            )
            (adapter_dir / f"{benchmark}_summary.json").write_text(
                json.dumps(summaries[benchmark], indent=2, ensure_ascii=False), encoding="utf-8"
            )
        all_summaries[name] = summaries
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    result = {
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "benchmarks": args.benchmarks,
        "ifeval_rows": len(datasets["ifeval"]),
        "gsm8k_rows": len(datasets["gsm8k"]),
        "bbh_rows": len(datasets["bbh"]),
        "summaries": all_summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
