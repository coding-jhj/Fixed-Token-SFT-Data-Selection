# Data and Environment Audit

## Status

The primary training pool and evaluation sets are frozen. A short smoke adapter and all six main comparison adapters (`random`, `quality`, `diversity` at seeds `13` and `42`) have been trained, reloaded, and generation-checked. A 32-row-per-benchmark frozen evaluation pilot and the final balanced evaluation have completed for all six adapters. The final balanced run contains 3,984/3,984 complete rows; processed validation and statistical results are under `work/results_final/`.

## Primary dataset candidate: `HuggingFaceTB/smoltalk`

### Verified from the official dataset card

- Purpose: synthetic supervised fine-tuning data for language models.
- Dataset card description: approximately 1M samples were used in the SmolLM2 development description.
- Current Hub viewer reports 2,197,730 rows and approximately 4.15 GB; the exact usable count depends on the selected configuration/subset.
- New subsets named `Smol-Magpie-Ultra`, `Smol-contraints`, `Smol-rewrite`, and `Smol-summarize` are described as Apache 2.0.
- Existing public subsets retain their original dataset licenses. The collection must not be treated as uniformly Apache 2.0 without a source-level audit.
- `Smol-contraints` is described as decontaminated against IFEval. This does not establish that every other subset is contamination-free.

### Newly created Apache-2.0 subset counts from the official card

- `smol-magpie-ultra`: 400,000 samples;
- `smol-constraints`: 36,000 samples;
- `smol-rewrite`: 50,000 samples;
- `smol-summarize`: 100,000 samples.

These four subsets are the default primary pool. The full local selector measured the final Qwen chat-template token counts, excluded examples over 2,048 tokens, and produced exact 1,000,000-token manifests.

### Frozen selection results

- Fixed dataset revision: `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`.
- Fixed tokenizer revision: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Accepted candidate pool: 555,479 rows after hard filters and `quality_floor=0.55`.
- Candidate cap: deterministic reservoir cap of 1,000 rows per source x length stratum, with shared `pool_seed=0`.
- Selection conditions: stratified random, quality-only, and quality-plus-diversity; selection seeds 13 and 42.
- Each of the six manifests has exactly 1,000,000 formatted tokens, unique `example_id` values, and no selected row above 2,048 tokens.
- IFEval contamination report covers the official 541-example train split. Exact and near overlap are both zero for all six manifests at threshold 0.92.
- The contamination checker supports direct reading of the cached IFEval Arrow file because the local `datasets.load_dataset` path can stall after its cache lookup under the current Windows environment.

### Frozen evaluation sets

- GSM8K: `openai/gsm8k`, config `main`, split `test`, revision `740312add88f781978c0658806c59bc2815b9866`, 1,319 rows.
- BBH: `lukaemon/bbh`, all 27 configs, split `test`, revision `982bb89fd79532a8ac676a61fc42eb1aeec63f99`, 6,511 rows.
- IFEval: `google/IFEval`, split `train`, 541 prompts from the cached official dataset revision recorded in `work/selection_manifests/contamination_report_all.json`.
- All six training manifests have zero exact and near overlap with GSM8K and BBH at normalized `SequenceMatcher` threshold 0.92. The frozen JSONL files and metadata are under `work/evaluation_sets/`.
- `src/evaluate_frozen.py` uses the pinned base-model revision and PEFT adapters, official IFEval scoring utilities where available, deterministic GSM8K/BBH answer extraction, and batch generation. Offline fallbacks for optional evaluator dependencies are recorded in `work/logs/bugs.md`.
- Pilot output: `work/evaluation_pilot_32/evaluation_summary.json`; six adapters x IFEval/GSM8K/BBH x 32 rows, batch size 4. Pilot metrics are diagnostic only and must not be reported as final evidence.
- The GSM8K card reports an MIT license. The BBH conversion card does not expose a clear redistribution license in the retrieved metadata; do not redistribute raw BBH rows in the final ZIP until that license is verified. Keep the revision, metadata, and download script for reproduction.

### Required local audit before training

1. List every configuration and source field.
2. Record row counts and token counts after the final Qwen chat template.
3. Record the license URL for every included source.
4. Exclude sources with unclear redistribution terms.
5. Detect exact duplicates and near duplicates.
6. Check overlap with IFEval, GSM8K, and the selected BBH subset.
7. Freeze the included source list, dataset revision, and preprocessing hash.

### Initial streaming sample check

Using the Qwen3 tokenizer and 100 streaming examples per subset:

```text
smol-magpie-ultra: n=100, min=533, mean=1489.8, max=4288 tokens
smol-constraints:  n=100, min=75,  mean=211.3,  max=693 tokens
smol-rewrite:      n=100, min=198, mean=339.6,  max=694 tokens
smol-summarize:    n=100, min=150, mean=415.5, max=1814 tokens
```

These are sample statistics, not full-dataset statistics. The same measurements were reproduced in the dedicated research environment with `work/run_data_template_smoke.py`. The primary protocol now uses a 2,048-token context target. Examples above that limit are excluded rather than truncated. The sample row schema contains a `messages` field; subset configuration names serve as the source label.

## Secondary dataset candidate: `allenai/tulu-3-sft-mixture`

- The official card reports 939,343 examples.
- The collection is presented under ODC-BY-1.0, but the card explicitly states that different licenses apply to subsets and that some portions are non-commercial.
- It is therefore a research reference and fallback candidate, not the default training pool until subset-level license review is complete.

## Local environment check

Initial global-environment check on 2026-09-12:

```text
Python: 3.13.9
torch: 2.11.0
transformers: 4.57.3
datasets: 4.8.5
bitsandbytes: 0.49.2
sentence-transformers: 5.3.0
trl: missing
peft: missing
lm_eval: missing
GPU: NVIDIA GeForce RTX 5060, 8151 MiB
```

The base Anaconda environment was not used for experiments because importing `torch` failed with a duplicate `libiomp5md.dll` OpenMP runtime error. A dedicated conda environment was then created by cloning the CUDA-capable `qwen3-tts` environment and installing the research dependencies.

Dedicated research environment, verified on 2026-09-12:

```text
Environment: C:\Users\ghksw\anaconda3\envs\posttrain-research
Python: 3.12.13
torch: 2.11.0+cu128
CUDA: available
GPU: NVIDIA GeForce RTX 5060
transformers: 5.17.0
datasets: 5.0.1
bitsandbytes: 0.49.2
sentence-transformers: 6.0.1
trl: 1.13.0
peft: 0.20.0
lm_eval: 0.4.13
```

Integrated imports passed. The original `qwen3-tts` environment was restored separately and is not used for this research.

## QLoRA smoke test

Using `work/run_qlora_smoke.py` in the dedicated research environment:

```text
Model: Qwen/Qwen3-1.7B-Base
Quantization: 4-bit NF4 with double quantization
LoRA: r=16, alpha=32, dropout=0.05
Sequence length: 2,048
Forward/backward: passed for one batch-1 step
Loss: 6.6917
Peak allocated: 6,584.9 MiB
Peak reserved: 7,516.0 MiB
GPU memory reported: 8,151 MiB
```

The smoke test fits with approximately 635 MiB between reported peak reserved memory and total GPU memory. This validates feasibility for gradient accumulation, but not the full multi-seed training run.

## Current decision

Use the four newly created Apache-2.0 subsets as the primary pool. Before training, inspect the Hub configurations and source metadata, measure their post-template token counts, and run the evaluation-overlap check. Do not include the public-source subsets by default.

## Sources

- SmolTalk dataset card: https://huggingface.co/datasets/HuggingFaceTB/smoltalk
- Tülu 3 SFT mixture dataset card: https://huggingface.co/datasets/allenai/tulu-3-sft-mixture
