# 연구 프로토콜

## 상태

실험 전 수립한 영어 post-training 연구 프로토콜이며, 현재는 본 학습·최종 balanced evaluation·통계 분석까지 완료되었습니다. 원고와 서술 문서는 한국어로 작성하고, 실험 코드·명령어·dataset/model ID·metric 이름은 재현성을 위해 영어 technical string을 유지합니다. 실험 시작 이후 변경 사항은 사유와 함께 `work/THREAD.md`에 기록합니다.

## 연구 제목

**Quality–Diversity Data Selection under a Fixed Token Budget for English Instruction Post-Training**

## 문서 언어

The manuscript, tables, figures, captions, README, and reproduction package will be written in Korean. Experiment code, commands, dataset/model IDs, metric names, and logs retain their exact English technical strings when needed for reproducibility. Internal discussion may remain in Korean.

## 1. Research Question and Significance

### Research question

Under a fixed training-token budget and identical post-training settings, does selecting English instruction-tuning data with both quality and semantic diversity improve instruction-following generalization more reliably than random selection or quality-only selection for a compact language model?

### Significance

Post-training research requires more than running supervised fine-tuning. It requires specifying the desired behavior, curating data, controlling the training budget, evaluating unseen behavior, and reporting failures. This study isolates the data-selection decision while keeping the model and training budget fixed.

## 2. Hypotheses

### Primary hypothesis (H1)

Under a fixed token budget, the quality-plus-diversity selection policy will achieve higher held-out English instruction-following performance than both random selection and quality-only selection.

### Robustness hypothesis (H2)

The quality-plus-diversity policy will improve the performance of the weakest predefined evaluation group, not only the overall average.

### Null hypothesis (H0)

After controlling for training tokens, source proportions, length distribution, model, hyperparameters, and random seed, quality-plus-diversity selection will not produce a reproducible improvement over the random baseline.

### Falsification rule

H1 is not supported if the quality-plus-diversity condition does not exceed the random baseline on the primary metric with a 95% confidence interval excluding zero, or if the apparent gain disappears across seeds. If quality-only wins, the conclusion will be revised to state that the measured diversity procedure did not add value under this budget.

## 3. Experimental Scope

### Base model

- Model: `Qwen/Qwen3-1.7B-Base`
- Training method: 4-bit QLoRA supervised fine-tuning
- Fallback model: `Qwen/Qwen2.5-1.5B` only if the Qwen3 training stack cannot be made reproducible
- Hardware: one `NVIDIA GeForce RTX 5060` with approximately 8 GB VRAM

### Data pool

Primary candidate: the four newly created Apache-2.0 subsets inside `HuggingFaceTB/smoltalk`:

- `smol-magpie-ultra`;
- `smol-constraints`;
- `smol-rewrite`;
- `smol-summarize`.

The public-source subsets bundled in the full SmolTalk mixture will be excluded from the primary experiment unless a separate source-level license audit explicitly admits them. `allenai/tulu-3-sft-mixture` is a secondary candidate only if its subset-level license audit permits the intended use.

The data audit will record:

- dataset and subset name;
- source license and URL;
- language and task metadata;
- number of examples and token count;
- duplicate and near-duplicate counts;
- overlap with evaluation prompts;
- preprocessing decisions and hashes.

### Fixed training budget

The primary comparison will use exactly 1,000,000 formatted, non-padding training tokens per condition. The token count will be computed after applying the final chat template. The same source proportions and length-bin proportions will be used across conditions wherever the available pool allows it.

The main sequence length is 2,048 tokens because an initial 100-example sample from `smol-magpie-ultra` averaged approximately 1,490 formatted tokens. Examples exceeding 2,048 formatted tokens are excluded before selection; no silent truncation is used. The QLoRA smoke test passed at this context length on the available GPU.

The frozen selector uses one shared candidate pool (`pool_seed=0`) for all conditions. It applies a deterministic reservoir cap of 1,000 rows per source x length stratum, then allocates the 1,000,000-token budget by stratum and performs a deterministic whole-example subset-sum correction to reach the exact budget. The cap and exclusion rule are limitations to report, not hidden implementation details.

## 4. Data-Selection Conditions

All conditions begin from the same license-cleared and deduplicated pool.

### Condition A: Stratified random selection

Sample examples randomly while matching source and token-length strata. This is the primary baseline.

### Condition B: Quality-only selection

Rank examples using a deterministic quality proxy based on:

- valid English-language content;
- non-empty and complete assistant response;
- absence of malformed markup and excessive repetition;
- suitable instruction and response length;
- absence of obvious template or metadata artifacts.

The quality proxy is not treated as ground-truth quality. A 200-example human audit will estimate whether the proxy tracks relevance, correctness, completeness, and clarity.

### Condition C: Quality-plus-diversity selection

First apply the same hard quality filters and quality floor used by Condition B. Then represent each example with a fixed English sentence embedding, cluster the pool, and select examples across semantic clusters using a round-robin policy while preserving the source and length constraints. The embedding model, clustering seed, number of clusters, and selection seed will be recorded in the configuration.

## 5. Training Controls

The following settings will be identical across conditions:

- model checkpoint and tokenizer revision;
- chat template;
- formatted token budget;
- maximum sequence length;
- LoRA rank, alpha, dropout, and target modules;
- optimizer, learning rate, scheduler, warm-up, and weight decay;
- effective batch size and number of training tokens;
- evaluation and checkpoint rules;
- random seeds.

Initial configuration for the smoke test:

- LoRA rank: 16;
- LoRA alpha: 32;
- LoRA dropout: 0.05;
- target modules: all linear modules when supported;
- effective batch size: 8 sequences at the 2,048-token setting;
- gradient checkpointing: enabled if required by the 8 GB VRAM limit;
- one pass over the fixed token budget;
- seeds: 13 and 42 for the first comparison, with 2026 added to the strongest comparison.

The QLoRA smoke test passed with this context length and initial configuration. It may not be silently tuned between comparison conditions.

## 6. Evaluation Plan

### Primary metric

`IFEval` prompt-level strict instruction-following accuracy, using one fixed evaluation command and deterministic generation settings.

### Secondary metrics

- `IFEval` instruction-level accuracy;
- `GSM8K` exact-match accuracy on a fixed evaluation subset;
- `BBH` normalized accuracy on a fixed evaluation subset;
- held-out instruction tasks not used for selection;
- format compliance rate for structured-output prompts;
- blind human ratings on helpfulness, correctness, instruction adherence, and clarity.

### Human evaluation

Use 100 held-out English prompts, randomized and blind to model identity. Each output will receive ratings from at least two independent raters when available. Agreement and disagreements will be reported. A single-rater result will be labeled as a limitation rather than as a reliable human-preference claim.

### Regression checks

Report whether post-training reduces performance on a fixed general-capability subset or increases malformed, repetitive, or refusal-like outputs. A gain on IFEval alone will not be described as a general improvement.

The frozen evaluation files use GSM8K `main/test` (1,319 examples, revision `740312add88f781978c0658806c59bc2815b9866`) and all 27 BBH test tasks (6,511 examples, revision `982bb89fd79532a8ac676a61fc42eb1aeec63f99`). The training manifests have zero exact and near overlap with both sets under the normalized near-duplicate check with threshold 0.92.

### Resource-constrained evaluation amendment (2026-09-12)

The full frozen sets were retained for provenance, but the final comparison uses a deterministic, predeclared subset because the available 8 GB GPU made exhaustive six-adapter evaluation impractical. The subset is generated by `src/make_eval_subset.py` with selection seed `2026` and recorded in `work/evaluation_subsets_final/manifest.json`:

- IFEval: 192 prompts selected with coverage-first stratification over `instruction_id_list`;
- GSM8K: 256 prompts selected across four prompt-length quantiles;
- BBH: 8 prompts from each of the 27 task subsets, for 216 prompts total.

All six adapters use exactly the same rows. IFEval uses `max_new_tokens=512`; GSM8K and BBH use `max_new_tokens=128`. Batch size 8 passed a one-adapter smoke test. Results are interpreted as a fixed-subset comparison and are not claimed to estimate performance on the complete benchmark distributions.

## 7. Statistical Analysis

- Report mean and standard deviation across seeds.
- Use paired bootstrap confidence intervals over evaluation examples for the primary metric.
- Compare Condition C against Condition A as the primary contrast.
- Treat other pairwise comparisons and secondary benchmarks as exploratory unless explicitly pre-registered.
- Report per-group results, not only a single aggregate score.
- Preserve all raw model outputs used for scoring.

## 8. Reproducibility Package

The final ZIP will include:

```text
paper.pdf
README.md
data_manifest.csv
configs/
src/select_data.py
src/train.py
src/evaluate.py
scripts/
results/raw/
results/processed/
figures/
environment.lock
CHANGELOG.md
```

The README will state the exact model revision, dataset revisions, licenses, preprocessing hash, seed, command, hardware, software versions, and known limitations.

## 9. Expected Limitations

- A 1.7B model may not represent larger-model post-training behavior.
- The quality proxy may not measure factual correctness reliably.
- A single data pool and a single token budget limit generalization.
- Human evaluation may have limited sample size and annotator agreement.
- Results may depend on the embedding model and clustering procedure.
- QLoRA results may differ from full-parameter fine-tuning.

## 10. Execution Gates

### Gate 1: Protocol and data audit

Do not train until the dataset source, license, token count, split, and contamination report exist.

### Gate 2: Baseline and smoke test

Do not start the main comparison until the base-model baseline, VRAM usage, loss curve, generation sample, and effective token count are saved.

### Gate 3: Main comparison

Run all conditions with identical configurations and seeds. Do not inspect the final test results to change the selection policy.

### Gate 4: Analysis and writing

Write the Results section from saved outputs first. Write the Abstract and Conclusion last. State whether H1 and H2 were supported, modified, or rejected.

## References for the protocol

- Ouyang et al. (2022), *Training language models to follow instructions with human feedback*: https://arxiv.org/abs/2203.02155
- Rafailov et al. (2023), *Direct Preference Optimization*: https://arxiv.org/abs/2305.18290
- Lambert et al. (2024), *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*: https://arxiv.org/abs/2411.15124
- Bukharin et al. (2023), *Data Diversity Matters for Robust Instruction Tuning*: https://arxiv.org/abs/2311.14736
- Liu et al. (2023), *What Makes Good Data for Alignment?*: https://arxiv.org/abs/2312.15685
