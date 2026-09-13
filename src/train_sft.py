"""Run a reproducible QLoRA SFT smoke or main run from a frozen JSONL manifest."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DEFAULT_TOKENIZER_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DEFAULT_MAX_LENGTH = 2048


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ManifestDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, max_rows: int | None) -> None:
        self.rows: list[dict] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                token_count = int(row["token_count"])
                if token_count > max_length:
                    raise ValueError(f"Manifest row exceeds max_length: {row['example_id']} {token_count}")
                encoded = tokenizer.apply_chat_template(
                    row["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                )
                input_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
                if hasattr(input_ids, "tolist"):
                    input_ids = input_ids.tolist()
                if input_ids and isinstance(input_ids[0], list):
                    input_ids = input_ids[0]
                if len(input_ids) != token_count:
                    raise ValueError(
                        f"Manifest token mismatch: {row['example_id']} "
                        f"manifest={token_count} encoded={len(input_ids)}"
                    )
                self.rows.append({"example_id": row["example_id"], "input_ids": input_ids})
                if max_rows is not None and len(self.rows) >= max_rows:
                    break
        if not self.rows:
            raise ValueError(f"Manifest contains no rows: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


def make_collate_fn(pad_token_id: int):
    def collate(rows: list[dict]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in rows)
        input_ids = []
        attention_mask = []
        labels = []
        for row in rows:
            ids = row["input_ids"]
            pad_count = max_length - len(ids)
            input_ids.append(ids + [pad_token_id] * pad_count)
            attention_mask.append([1] * len(ids) + [0] * pad_count)
            labels.append(ids + [-100] * pad_count)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def build_model(model_id: str, model_revision: str, tokenizer_revision: str):
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
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for QLoRA training")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch-size and gradient-accumulation-steps must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive when provided")
    if args.max_steps is None and args.epochs < 1:
        raise ValueError("epochs must be positive when max-steps is not provided")

    set_seed(args.seed)
    started = time.time()
    model, tokenizer = build_model(args.model, args.model_revision, args.tokenizer_revision)
    dataset = ManifestDataset(args.manifest, tokenizer, args.max_length, args.max_rows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collate_fn(tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    device = model.device
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    losses: list[float] = []
    tokens_seen = 0
    optimizer_steps = 0
    batches = 0
    iterator = iter(loader)
    target_batches = (
        args.max_steps * args.gradient_accumulation_steps
        if args.max_steps is not None
        else len(loader) * args.epochs
    )
    micro_batches = 0
    accumulation_batches = 0
    while micro_batches < target_batches:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {key: value.to(device) for key, value in batch.items()}
        tokens_seen += int(batch["attention_mask"].sum().item())
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = model(**batch).loss
        losses.append(float(loss.detach().cpu()))
        loss.backward()
        batches += 1
        micro_batches += 1
        accumulation_batches += 1
        is_last_batch = micro_batches == target_batches
        if accumulation_batches == args.gradient_accumulation_steps or is_last_batch:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(accumulation_batches)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            accumulation_batches = 0
            del batch, loss
            gc.collect()
            torch.cuda.empty_cache()
            if optimizer_steps % args.log_every == 0 or micro_batches == target_batches:
                print(
                    json.dumps(
                        {
                            "optimizer_step": optimizer_steps,
                            "micro_batches": micro_batches,
                            "tokens_seen": tokens_seen,
                        }
                    ),
                    flush=True,
                )

    torch.cuda.synchronize()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {
        "manifest": str(args.manifest),
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "seed": args.seed,
        "max_length": args.max_length,
        "manifest_rows_loaded": len(dataset),
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs if args.max_steps is None else None,
        "target_batches": target_batches,
        "optimizer_steps": optimizer_steps,
        "tokens_seen": tokens_seen,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "loss_curve": losses,
        "device": torch.cuda.get_device_name(0),
        "peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
