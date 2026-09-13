"""Generate a deterministic sample from the base model or a saved LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_MODEL = "Qwen/Qwen3-1.7B-Base"
DEFAULT_MODEL_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
DEFAULT_TOKENIZER_REVISION = "ea980cb0a6c2ae4b936e82123acc929f1cec04c1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--tokenizer-revision", default=DEFAULT_TOKENIZER_REVISION)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this generation check")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.tokenizer_revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.model_revision,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.float16,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    messages = [{"role": "user", "content": args.prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = generated[:, input_ids.shape[1] :]
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    result = {
        "model": args.model,
        "model_revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "prompt": args.prompt,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": int(new_tokens.shape[1]),
        "output": text,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
