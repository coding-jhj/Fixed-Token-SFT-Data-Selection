# 고정 토큰 예산에서의 데이터 선택 효과

**저자:** 정환주  
**부제:** 품질·다양성 기반 선택이 영어 지시 수행에 미치는 영향  
**작성일:** 2026-09-13

## 초록

Instruction-tuning 데이터 선택은 품질과 양의 문제로 자주 논의되지만, 선택 정책마다 학습 토큰 수가 다르거나 평가 예시가 달라지면 정책의 효과를 분리하기 어렵습니다. 본 연구는 포맷팅 후 학습 토큰 수를 정확히 고정한 자원 제약 조건에서 층화 random, quality-only, quality-plus-semantic-diversity 선택을 비교합니다. Qwen/Qwen3-1.7B-Base에 4-bit NF4 QLoRA supervised fine-tuning을 적용하고, 각 정책을 seed 13과 42에서 실행했습니다. 각 adapter는 정확히 1,000,000개의 formatted training token을 사용했습니다. 사전에 primary metric으로 지정한 deterministic IFEval subset의 prompt-level strict accuracy에서 diversity의 평균은 11.46%(SD 0.74)로 random의 11.72%(SD 1.84)보다 낮았으며, paired bootstrap 차이는 -0.26 percentage points, 95% confidence interval은 [-3.39, 2.60]이었습니다. 따라서 quality-plus-diversity가 영어 instruction following을 안정적으로 개선한다는 주가설은 지지되지 않았습니다. 다만 secondary BBH subset에서는 diversity가 33.56%로 random의 24.07%보다 높았고, paired 95% confidence interval은 [4.63, 14.58] percentage points였습니다. GSM8K의 우위는 불확실했습니다. 최종 평가의 3,984개 출력 행은 모두 구조적으로 유효했지만, 3,937개 행은 generation cap에 도달했을 가능성이 있는 것으로 진단되었습니다. 본 연구는 이 결과를 일반적 우위가 아닌 task-dependent evidence로 해석하고, 더 긴 생성 한도·추가 seed·human evaluation을 후속 과제로 남깁니다.

**핵심어:** instruction tuning, data selection, data diversity, QLoRA, fixed token budget, evaluation reproducibility

## 1. 연구 개요

### 1.1 문제와 연구 질문

Post-training 성능은 어떤 행동을 학습하는지, 어떤 예시를 포함하는지, 학습에 몇 개의 토큰을 사용하는지, held-out 행동을 어떻게 측정하는지에 따라 달라집니다. 같은 데이터셋을 사용하더라도 선택 정책이 더 긴 예시를 많이 포함하면 실제 학습 신호의 양과 구성은 달라집니다. 따라서 데이터 선택 정책을 비교하려면 최소한 모델 revision, 데이터 revision, 포맷팅 규칙, 학습 토큰 예산, 평가 예시, 채점 절차를 함께 통제해야 합니다.

본 연구의 연구 질문은 다음과 같습니다. 고정된 학습 토큰 예산과 동일한 학습 설정에서 quality와 semantic diversity를 함께 사용하는 선택 정책이 compact language model의 영어 instruction-following generalization을 stratified random 선택이나 quality-only 선택보다 개선하는가? 또한 평균 성능의 변화가 아니라 task별 최저 성능과 seed 간 변동성에서도 일관된 이점이 나타나는가?

본 비교는 약 8 GB GPU 한 대에서 실행 가능한 규모로 설계했습니다. 목표는 가장 큰 모델에서의 최고 점수가 아니라, 선택 정책의 차이를 재현 가능한 방식으로 분리하고 결과가 음성일 때도 원인을 추적할 수 있는 실험 기록을 남기는 것이었습니다.

### 1.2 가설과 판정 규칙

주가설 H1은 quality-plus-diversity 선택이 stratified random 선택보다 primary metric인 IFEval prompt-level strict accuracy에서 높은 성능을 보일 것이라는 예측입니다. 강건성 가설 H2는 quality-plus-diversity 선택이 사전에 지정한 최저 그룹 지표인 BBH minimum task accuracy를 개선할 것이라는 예측입니다.

최종 평가 전에 다음 판정 규칙을 고정했습니다. Diversity minus random의 paired 95% confidence interval이 0을 포함하면 H1을 지지하지 않습니다. 또한 두 seed에서 효과의 방향이 유지되지 않으면 안정적 개선으로 주장하지 않습니다. H2는 minimum BBH task score가 random보다 높을 때만 지지합니다. 이 규칙에 따라 secondary benchmark에서 유리한 결과가 있더라도 primary hypothesis의 증거로 소급하여 사용하지 않았습니다.

### 1.3 연구 기여와 범위

- 고정된 1,000,000 formatted-token 예산 아래에서 세 가지 선택 정책과 두 개의 seed를 동일 조건으로 비교했습니다.
- 데이터 source revision, license 범위, contamination 검사, 평가 subset manifest, 출력 무결성 검사를 함께 보존했습니다.
- 수행하지 않은 full benchmark, 세 번째 seed, human audit, regression suite를 결과로 포장하지 않고 실행 범위와 한계에 명시했습니다.

본 연구는 특정 데이터 선택기가 모든 모델과 모든 task에서 최적이라는 주장을 하지 않습니다. 하나의 English data pool, 하나의 base model, 하나의 embedding model, 하나의 토큰 예산에서 관찰되는 효과를 검증하는 resource-constrained study입니다.

## 2. 관련 연구

### 2.1 Instruction post-training

InstructGPT는 supervised demonstration과 preference feedback을 사용해 언어 모델이 사용자의 지시를 따르도록 학습하는 대표적인 post-training 패턴을 제시했습니다 [1]. Direct Preference Optimization은 별도의 reward-model pipeline 없이 preference data를 직접 최적화하는 목적함수를 제안했습니다 [2]. Tülu 3는 공개 post-training recipe에서 data mixture, decontamination, staged training, held-out evaluation을 함께 관리해야 한다는 점을 보여줍니다 [3].

이 연구들의 공통점은 학습 objective만으로 결과를 설명하지 않고 데이터 구성과 평가 설계를 결과 해석의 일부로 취급한다는 점입니다. 본 연구는 그중 데이터 선택 단계에 초점을 두고, 다른 조건을 가능한 한 고정한 뒤 선택 정책의 추가 효과를 확인합니다.

### 2.2 품질과 다양성 기반 데이터 선택

Alignment data의 유용한 특성을 분석하는 연구는 단순한 데이터 양보다 응답의 품질, 난이도, 유효성, 학습 대상 행동과의 관련성이 중요할 수 있음을 지적합니다 [4]. Robust instruction tuning 연구는 데이터 다양성이 특정 형식이나 문구에 과적합되는 위험을 낮추고 새로운 task로의 전이를 돕는지 조사했습니다 [5].

그러나 quality와 diversity를 결합한 점수가 항상 일반화 성능의 개선으로 이어지는 것은 아닙니다. Quality proxy가 실제 유용성을 충분히 반영하지 못하거나, diversity가 primary task에 필요한 행동을 희석할 수 있습니다. 본 연구는 새로운 universal quality function을 제안하지 않고, heuristic quality score와 embedding clustering을 고정하여 추가적인 측정 가능 이점이 있는지 검증합니다.

### 2.3 본 연구의 차별점

기존 논의와 달리 본 연구는 선택 정책의 비교 단위를 formatted training token으로 고정했습니다. 후보 행 수가 아니라 실제 모델 입력에 들어간 토큰 수를 budget으로 정의하고, 전체 평가 원천과 최종 deterministic subset을 분리했습니다. 그 결과 평균 점수뿐 아니라 paired contrast, seed 표준편차, 출력 completeness, generation-cap 진단을 함께 보고할 수 있습니다.

## 3. 연구 설계

### 3.1 데이터 풀과 provenance

학습 후보 풀은 HuggingFaceTB/smoltalk의 다음 네 subset에서 구성했습니다. Dataset revision은 `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`로 고정했습니다. 본 실험에서는 source-level license 검토 결과가 명확한 새 Apache-2.0 설명 subset만 사용했으며, public-source subset과 license 범위를 audit에서 명확히 확인하지 못한 Tulu 계열 source는 후보 풀에서 제외했습니다.

| Source | 공식 설명 규모 | 필터 후 후보 행 |
|---|---:|---:|
| smol-magpie-ultra | 400,000 | 372,669 |
| smol-constraints | 36,000 | 34,294 |
| smol-rewrite | 50,000 | 53,342 |
| smol-summarize | 100,000 | 95,174 |
| 합계 | 586,000 | 555,479 |

후보 행에는 언어 신호, 구조, 반복, 길이, quality-floor 검사를 적용했습니다. 이후 source-by-length stratum별로 최대 1,000행의 공유 후보 cap을 `pool_seed=0`으로 적용했습니다. 이 cap은 세 선택 정책이 서로 다른 후보 모집단을 보지 않도록 공통으로 사용했습니다. 필터 후 후보 수가 공식 설명 규모보다 많아지는 source가 있는 이유는 원천 card의 설명 규모와 로컬 snapshot에서 source label을 기준으로 집계한 필터 후 행 수가 서로 다른 표본 정의를 사용하기 때문입니다. 분석에서 사용하는 실제 모집단은 revision이 고정된 로컬 snapshot의 필터 후 555,479행입니다.

### 3.2 전처리와 고정 토큰 예산

각 예시는 Qwen3 chat template으로 포맷팅한 뒤 tokenizer로 길이를 계산했습니다. 최대 길이는 2,048 token이며, 이를 넘는 예시는 조용히 자르지 않고 완전한 example 단위로 제외했습니다. 선택 결과는 고유한 `example_id`를 가지며, 각 manifest의 formatted non-padding token 합이 정확히 1,000,000이 되도록 whole-example subset-sum correction을 적용했습니다.

따라서 정책별 selected row 수는 같을 필요가 없지만 실제 학습 토큰 예산은 같습니다. 이 설계는 긴 예시를 포함하는 정책이 단순히 더 많은 token update를 얻는 혼동을 줄입니다. 학습 manifest와 token summary는 `work/selection_manifests/`에 보존했습니다.

### 3.3 선택 정책

모든 정책에 동일한 hard filter를 적용했습니다. Formatted token 수가 0보다 크고 2,048 이하이며, alphabetic word가 최소 4개이고 alphabetic letter가 최소 20개여야 합니다. ASCII ratio는 0.25 이상이어야 하며, NUL character와 12회 이상 반복되는 동일 문자열 패턴을 제거했습니다. 그 뒤 quality floor 0.55를 적용했습니다.

Quality score는 다음 다섯 요소의 가중합입니다. English signal과 ASCII ratio 평균으로 계산한 `english_score`에 0.25, assistant response word count에 기반한 `response_score`에 0.25, 32~1,536 token 구간이면 1.0이고 그 밖의 허용 길이면 0.7인 `length_score`에 0.20, 반복 억제 점수에 0.20, artifact 억제 점수에 0.10을 부여했습니다. 이 점수는 human quality label이 아니라 선택을 위한 heuristic proxy입니다.

세 정책은 다음과 같이 구현했습니다.

1. Random 선택은 source와 token-length stratum을 유지하면서 seed 13 또는 42로 층화 무작위 선택을 수행했습니다.
2. Quality 선택은 동일한 quality floor 이후 quality score와 stable example ID를 기준으로 deterministic ranking을 수행했습니다.
3. Diversity 선택은 동일한 quality floor을 통과한 행에 `sentence-transformers/all-MiniLM-L6-v2`의 normalized embedding을 계산하고, `MiniBatchKMeans`로 clustering했습니다. 클러스터 수는 filtered row 수를 n이라고 할 때 `min(n, max(2, round(sqrt(n))))`로 정하고 random state는 선택 seed로 고정했습니다. 각 cluster 안에서는 quality score와 stable ID 순으로 정렬한 뒤 cluster 간 round-robin으로 뽑고, source-by-length quota를 적용했습니다.

세 정책 모두 마지막에 exact-token correction을 적용하고, 선택 seed 13과 42를 사용했습니다. 이 순서로 인해 diversity가 품질이 지나치게 낮은 예시를 보충하는 방식으로 작동하지 않도록 quality floor와 hard filter를 먼저 공유했습니다.

### 3.4 모델과 학습 조건

Base model과 tokenizer는 `Qwen/Qwen3-1.7B-Base`이며 revision은 `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`로 고정했습니다. 4-bit NF4 quantization, double quantization, fp16 compute를 사용했습니다. LoRA는 rank 16, alpha 32, dropout 0.05이며 target module은 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`입니다.

Maximum sequence length는 2,048, device batch size는 1, gradient accumulation은 8입니다. 학습은 exact token budget을 한 번 통과하는 1 epoch equivalent procedure로 수행했습니다. Optimizer는 AdamW, learning rate는 2e-4, weight decay는 0.0입니다. Custom loop에는 scheduler와 warm-up을 두지 않았고 gradient clipping은 1.0, gradient checkpointing은 활성화했습니다. 정책과 seed를 제외한 설정은 모두 동일합니다.

실험 GPU는 NVIDIA GeForce RTX 5060이며 시스템에 보고된 GPU memory는 8,151 MiB입니다. 연구 전용 환경의 주요 버전은 다음과 같습니다.

| 환경 구성 요소 | 버전 또는 값 |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Transformers | 5.17.0 |
| Datasets | 5.0.1 |
| bitsandbytes | 0.49.2 |
| sentence-transformers | 6.0.1 |
| TRL | 1.13.0 |
| PEFT | 0.20.0 |
| lm-evaluation-harness | 0.4.13 |
| GPU | NVIDIA GeForce RTX 5060, 8151 MiB |

### 3.5 오염 검사와 평가 데이터

평가 원천은 provenance를 위해 전체 revision을 고정했습니다. IFEval의 cached official train split에는 541 prompts가 있었고, GSM8K `main/test`는 1,319행, BBH는 27개 task의 6,511행이었습니다. GSM8K revision은 `740312add88f781978c0658806c59bc2815b9866`, BBH revision은 `982bb89fd79532a8ac676a61fc42eb1aeec63f99`입니다.

여섯 개 학습 manifest와 IFEval train, GSM8K test, BBH test 사이의 exact overlap과 normalized `SequenceMatcher` threshold 0.92 이상의 near overlap을 검사했습니다. 모든 manifest에서 두 종류의 overlap은 0이었습니다. 이 검사는 학습 데이터가 평가 문항을 직접 포함하지 않는지 확인하는 절차이며, 의미적으로 유사한 모든 문장을 제거했다는 주장은 아닙니다.

### 3.6 고정 평가 subset과 채점

최종 비교는 모든 adapter에 동일한 deterministic subset을 사용했습니다. Subset selection seed는 2026이며 subset JSONL의 SHA-256 hash와 선택 스크립트를 보존했습니다. IFEval은 instruction ID family별 coverage-first 방식으로 192개 prompt를 뽑았습니다. GSM8K는 prompt length를 네 quantile로 나누어 256개를 뽑았습니다. BBH는 27개 task에서 각각 8개씩 뽑아 216개로 구성했습니다.

| Benchmark | 원천 규모 | 고정 subset | 선택 정책 | 생성 한도 |
|---|---:|---:|---|---:|
| IFEval | 541 train prompts 확인 | 192 prompts | instruction family coverage-first | 512 new tokens |
| GSM8K | 1,319 test rows | 256 prompts | prompt length 4-quantile | 128 new tokens |
| BBH | 6,511 rows, 27 tasks | 216 prompts | task별 8개 | 128 new tokens |

Primary metric은 IFEval prompt-level strict accuracy입니다. IFEval instruction-level strict accuracy, GSM8K exact-match accuracy, BBH normalized accuracy, BBH task macro-average, BBH minimum task accuracy를 secondary 또는 robustness metric으로 사용했습니다. 모든 평가에서 greedy decoding(`do_sample=False`)과 batch size 8을 사용했습니다. IFEval은 instruction-level 판정도 함께 산출했으며, GSM8K와 BBH는 평가기 구현의 exact-match와 normalized answer 판정을 사용했습니다.

### 3.7 통계 분석과 무결성 검사

각 strategy의 평균과 표준편차는 seed 13과 42의 adapter 점수에서 계산했습니다. Paired bootstrap은 동일 evaluation ID의 두 strategy 출력을 짝지은 뒤, 먼저 두 seed 점수를 ID별로 평균하고 10,000회 재표집했습니다. Bootstrap seed는 2026입니다. 보고한 paired contrast는 quality minus random과 diversity minus random입니다.

최종 결과는 6개 adapter와 3개 benchmark의 18개 output file을 대상으로 검사했습니다. 파일별 예상 행 수와 실제 행 수, JSON parse error, duplicate ID, missing ID, extra ID, malformed score object, empty response를 확인했습니다. Decoded text를 다시 encoding한 token count가 generation cap 이상이면 possible truncation으로 표시했습니다. 이 flag는 원래 generated token ID가 저장되지 않은 상태에서 수행한 보수적 진단이므로, 모든 사례가 실제 truncation이었다고 단정하지 않습니다.

### 3.8 계획 대비 실제 실행 범위

| 변경 항목 | 계획 | 실제 실행 | 해석상 조치 |
|---|---|---|---|
| IFEval 전체 평가 | 541 prompts 전체 | 183행에서 비용 문제로 중단 | partial run은 diagnostic으로만 보존 |
| 최종 평가 | full benchmark 우선 | 고정 balanced subset 사용 | 최종 표에 subset임을 명시 |
| 추가 seed | seed 2026 추가 | 수행하지 않음 | 2-seed 한계로 명시 |
| human audit | 200-example quality audit | 수행하지 않음 | quality score를 heuristic으로 기술 |
| 회귀 평가 | general-capability suite | 완료하지 않음 | broad capability 개선 주장 금지 |
| base baseline | 전체 최종 benchmark baseline | generation smoke만 수행 | full baseline 부재를 한계로 명시 |

처음 실행한 32-example benchmark pilot도 pipeline 검증 산출물로 보존했지만, 최종 통계 비교에는 포함하지 않았습니다. 이 표는 계획과 실제의 차이를 숨기지 않기 위한 amendment 기록입니다.

## 4. 실행 결과

### 4.1 선택 manifest와 학습 실행

여섯 adapter 모두 formatted training token 합계가 1,000,000으로 확인되었습니다. Selected row 수는 예시 길이 분포와 whole-example token correction에 따라 달라졌습니다. Optimizer step 수와 elapsed time은 실제 run summary에 기록된 값입니다.

| 전략 | Seed | 선택 행 수 | Formatted token | Optimizer step | 실행 시간(분) |
|---|---:|---:|---:|---:|---:|
| Random | 13 | 1,006 | 1,000,000 | 126 | 27.9 |
| Random | 42 | 1,000 | 1,000,000 | 125 | 30.0 |
| Quality | 13 | 1,033 | 1,000,000 | 130 | 28.1 |
| Quality | 42 | 1,033 | 1,000,000 | 130 | 27.6 |
| Diversity | 13 | 1,029 | 1,000,000 | 129 | 25.7 |
| Diversity | 42 | 1,021 | 1,000,000 | 128 | 26.9 |

학습 loss의 첫 값과 마지막 값은 각각 random seed 13에서 2.5431과 1.0485, random seed 42에서 2.2237과 2.4013, quality seed 13에서 1.6032와 1.4287, quality seed 42에서 1.8829와 1.1492, diversity seed 13에서 2.5650과 0.9765, diversity seed 42에서 1.1651과 0.9089였습니다. 이 값은 training fit의 기록이며 evaluation performance의 대체 지표로 해석하지 않았습니다.

### 4.2 평가 출력의 완전성

모든 adapter에 대해 동일한 subset과 생성 설정으로 평가를 완료했습니다. 전체 3,984개 row가 예상된 example ID를 한 번씩 포함했고, 모든 validation 항목에서 오류가 0이었습니다.

| 검사 대상 | Adapter당 행 수 | 전체 행 수 | 유효 행 | Possible truncation |
|---|---:|---:|---:|---:|
| IFEval | 192 | 1,152 | 1,152 | 1,114 |
| GSM8K | 256 | 1,536 | 1,536 | 1,530 |
| BBH | 216 | 1,296 | 1,296 | 1,293 |
| 합계 | 664 | 3,984 | 3,984 | 3,937 |

Parse error, duplicate ID, missing ID, extra ID, malformed score object, empty response는 모두 0이었습니다. Possible truncation은 decoded response의 재인코딩 길이가 cap 이상인 행의 수입니다. 특히 GSM8K와 BBH에서는 거의 모든 응답이 이 진단에 걸렸으므로, 해당 benchmark 점수를 최종적인 완전 응답 성능으로 과대해석하지 않았습니다.

### 4.3 주요 평가 결과

| 선택 정책 | IFEval strict | IFEval instruction strict | GSM8K | BBH | BBH task macro | BBH worst task |
|---|---:|---:|---:|---:|---:|---:|
| Random | 11.72% ± 1.84 | 37.89% ± 1.51 | 20.12% ± 1.38 | 24.07% ± 1.96 | 24.07% ± 1.96 | 0.00% ± 0.00 |
| Quality | 10.94% ± 1.47 | 36.46% ± 1.18 | 21.09% ± 5.52 | 26.85% ± 1.31 | 26.85% ± 1.31 | 0.00% ± 0.00 |
| Diversity | 11.46% ± 0.74 | 36.22% ± 0.17 | 22.85% ± 3.59 | 33.56% ± 0.33 | 33.56% ± 0.33 | 0.00% ± 0.00 |

값은 두 seed의 평균 ± 표준편차입니다. IFEval과 GSM8K/BBH percentage는 전체 원천 distribution이 아니라 고정 subset에서 계산했습니다. BBH task macro는 27개 task에 동일한 가중치를 주며, 각 task에서 동일한 8개 예시를 사용했습니다. 모든 조건에서 적어도 한 task의 정답 수가 0이었기 때문에 BBH minimum task accuracy는 세 정책 모두 0이었습니다.

[그림 1. 전략별 평균과 seed 표준편차.]

### 4.4 Paired contrast

| Metric | Quality - random | Diversity - random |
|---|---:|---:|
| IFEval prompt strict | -0.78 pp [-4.43, 2.86] | -0.26 pp [-3.39, 2.60] |
| IFEval instruction strict | -2.13 pp [-5.64, 1.30] | -1.35 pp [-4.30, 1.61] |
| GSM8K accuracy | +0.98 pp [-2.93, 4.88] | +2.73 pp [-0.98, 6.45] |
| BBH accuracy | +2.78 pp [-1.85, 7.64] | +9.49 pp [4.63, 14.58] |

구간은 두 seed별 score를 example ID별로 평균한 뒤 수행한 paired 95% bootstrap confidence interval입니다. 네 contrast 중 0을 포함하지 않는 것은 secondary BBH에서 diversity가 random보다 높은 경우뿐입니다. Primary IFEval contrast에는 0이 포함되므로 주가설 H1을 지지할 근거로 사용할 수 없습니다.

### 4.5 보조 지표와 안정성

Diversity는 BBH에서 random보다 9.49 percentage points 높았고, 두 seed 표준편차도 0.33으로 낮았습니다. 그러나 동일한 diversity 정책이 IFEval에서는 random보다 낮았고, IFEval instruction-level strict에서도 -1.35 percentage points였습니다. 따라서 diversity의 효과는 benchmark가 측정하는 행동에 의존할 가능성이 있습니다.

Quality의 GSM8K 표준편차는 5.52로 random의 1.38보다 컸습니다. 이 차이는 quality policy가 수학 문항에 안정적인 이점을 제공한다고 보기 어렵게 합니다. 다만 두 seed만으로 분산의 신뢰할 만한 추정치를 얻기는 어렵기 때문에, 이 관찰은 후속 seed 추가를 요구하는 신호로만 보고합니다.

## 5. 논의

### 5.1 연구 질문에 대한 답

본 조건에서 quality-plus-diversity 선택은 고정된 1,000,000-token QLoRA budget 아래 primary IFEval prompt-level strict accuracy를 개선하지 못했습니다. Diversity의 평균은 random보다 0.26 percentage points 낮았고, paired confidence interval은 [-3.39, 2.60]으로 0을 포함했습니다. Quality-only 역시 primary metric에서 random보다 낮았습니다. 그러므로 H1은 지지되지 않습니다.

H2도 지지되지 않습니다. BBH minimum task accuracy는 모든 정책과 두 seed에서 0이었습니다. 이 지표는 최저 성능 task를 확인하는 stress signal로는 의미가 있지만, 현재의 task별 표본과 모델 규모에서는 정책 차이를 구분하지 못했습니다.

### 5.2 왜 BBH와 IFEval의 방향이 달랐는가

Diversity가 BBH에서만 유리하게 나타난 것은 본 실험의 관찰입니다. 가능한 설명으로는 embedding cluster round-robin이 heterogeneous reasoning pattern의 범위를 넓혀 BBH task coverage에 도움을 주었을 가능성이 있습니다. 그러나 이는 결과에 대한 inference이며, 본 연구는 cluster별 causal ablation이나 task별 coverage mediation analysis를 수행하지 않았습니다.

반대로 IFEval strict는 지시 형식과 constraint를 정확히 만족하는지를 측정합니다. Semantic diversity가 늘어도 선택된 예시가 IFEval의 특정 formatting behavior를 더 잘 가르친다는 보장은 없습니다. 현재 결과만으로 diversity가 instruction following에 해롭다고 결론 내릴 수도 없습니다. Primary subset의 generation cap 진단과 2-seed 설계가 함께 존재하기 때문입니다.

### 5.3 연구·엔지니어링 관점의 의미

이번 비교에서 가장 강한 결론은 특정 selector의 우승이 아니라 비교 방법에 관한 것입니다. Fixed-token accounting을 적용하면 row count가 다른 manifest도 공정하게 비교할 수 있습니다. Source-level license review와 revision pinning은 나중에 결과를 재생성할 수 있는 범위를 명확히 합니다. Deterministic subset, paired ID, raw output, validation report를 보존하면 음성 결과도 실험 오류와 구분할 수 있습니다.

실무적으로는 하나의 quality score에 의존하여 데이터를 줄이기보다, 목표 task의 행동과 subset 설계를 함께 검토해야 합니다. Diversity가 secondary benchmark에서 높은 점수를 얻었다는 이유만으로 데이터 선택 정책을 운영에 바로 적용하기보다, 긴 generation cap과 추가 seed로 효과를 먼저 확인해야 합니다.

## 6. 한계와 후속 실험

첫째, 모델은 1.7B parameter에 불과하므로 더 큰 모델에 결과가 전이된다고 보장할 수 없습니다. 둘째, quality proxy는 heuristic이며 계획했던 200-example human audit로 검증하지 않았습니다. 셋째, 최종 비교는 전체 benchmark distribution이 아닌 resource-constrained deterministic subset에 기반합니다. 넷째, 3,984개 출력 중 3,937개가 generation cap에 도달했을 가능성이 있어, 특히 GSM8K와 BBH 결과에 불완전한 응답의 영향이 있을 수 있습니다.

다섯째, 세 번째 seed 2026을 실행하지 않았고 blind human evaluation도 완료하지 않았습니다. 여섯째, general-capability regression suite와 full base-model benchmark baseline을 수행하지 않았으므로 helpfulness, fluency, safety, broad capability가 개선되었다고 주장할 수 없습니다. 일곱째, 하나의 English data pool, 하나의 embedding model, 하나의 token budget만 비교했습니다. 마지막으로 BBH raw row는 conversion card의 redistribution license가 audit에서 명확하지 않아 reproduction ZIP에 포함하지 않았습니다.

후속 실험은 다음 순서가 적절합니다. 먼저 동일한 model revision과 manifest를 유지하면서 IFEval 512, GSM8K/BBH 128보다 긴 generation limit으로 평가를 반복해야 합니다. 다음으로 seed를 하나 이상 추가하고, quality score의 상위·중간·하위 구간에 대한 human audit를 수행해야 합니다. 그 뒤 full benchmark와 general-capability regression suite를 추가하고, task별 표본 수를 늘린 robustness statistic을 사전에 고정해야 합니다. 마지막으로 더 큰 model과 다른 data pool에서 같은 selector를 재현하여 결과의 범위를 확인해야 합니다.

## 7. 결론

고정된 1,000,000 formatted-token QLoRA budget과 Qwen3-1.7B-Base 조건에서 quality-plus-diversity 선택은 사전에 정한 IFEval prompt-level strict accuracy를 안정적으로 개선하지 못했습니다. Secondary BBH subset에서는 개선이 관찰되었지만, IFEval에서의 음성 결과와 generation-cap 진단을 고려하면 이를 일반적 우위로 해석할 수 없습니다. 따라서 본 연구의 결론은 “diversity가 항상 좋다”가 아니라, 데이터 선택 정책의 효과가 task와 평가 설계에 의존하며 고정 예산·고정 subset·무결성 검사를 갖춘 비교가 필요하다는 것입니다.

본 연구는 여섯 개 adapter의 frozen manifest, exact-token 학습 기록, 18개 평가 출력, validation report, paired bootstrap 분석, 계획 대비 실제 실행 범위를 함께 제공했습니다. 실행하지 않은 third seed, human audit, full baseline, regression suite는 명시적으로 후속 과제로 남겼습니다.

## 데이터와 코드 공개 및 재현 절차

Workspace의 `work/selection_manifests/`에는 여섯 개 frozen selection manifest와 token summary가 있습니다. `work/evaluation_subsets_final/`에는 최종 IFEval, GSM8K, BBH subset manifest와 hash가 있습니다. `work/main_*`에는 adapter run summary와 training log가 있고, `work/evaluation_final_balanced/`에는 18개 benchmark output이 있습니다. `work/results_final/`에는 processed metric, validation report, paired bootstrap 결과와 figure가 있습니다.

분석 코드는 `src/analyze_final_results.py`, frozen evaluation 코드는 `src/evaluate_frozen.py`, subset 생성 코드는 `src/make_eval_subset.py`, 전체 재현 명령은 `scripts/reproduce_analysis.ps1`에 기록했습니다. Reproduction ZIP에는 manuscript, PDF, 실행 스크립트, 환경 고정 파일, processed result와 validation 자료를 포함했습니다. Raw BBH row는 redistribution license가 명확하지 않아 의도적으로 제외했으며, 대신 BBH revision, subset hash, task allocation, download/regeneration 절차를 포함했습니다.

재현 시 먼저 고정된 Python 환경과 model/dataset revision을 확인하고, selection manifest의 token 합계와 example ID uniqueness를 검사해야 합니다. 이후 adapter별 output completeness를 검증한 다음 분석 스크립트를 실행합니다. Partial IFEval run과 32-row pilot은 pipeline 진단 기록이므로 final table의 통계에 합산하지 않습니다.

## 참고문헌

1. Ouyang, L., et al. (2022). *Training language models to follow instructions with human feedback*. arXiv:2203.02155. https://arxiv.org/abs/2203.02155
2. Rafailov, R., et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. arXiv:2305.18290. https://arxiv.org/abs/2305.18290
3. Lambert, N., et al. (2024). *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*. arXiv:2411.15124. https://arxiv.org/abs/2411.15124
4. Liu, W., et al. (2023). *What Makes Good Data for Alignment?*. arXiv:2312.15685. https://arxiv.org/abs/2312.15685
5. Bukharin, A., et al. (2023). *Data Diversity Matters for Robust Instruction Tuning*. arXiv:2311.14736. https://arxiv.org/abs/2311.14736
6. Zhou, J., et al. (2023). *Instruction-Following Evaluation for Large Language Models*. arXiv:2311.07911. https://arxiv.org/abs/2311.07911
7. Cobbe, K., et al. (2021). *Training Verifiers to Solve Math Word Problems*. arXiv:2110.14168. https://arxiv.org/abs/2110.14168
8. Suzgun, M., et al. (2022). *Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them*. arXiv:2210.09261. https://arxiv.org/abs/2210.09261
9. Hugging Face. *HuggingFaceTB/smoltalk dataset card*. https://huggingface.co/datasets/HuggingFaceTB/smoltalk
10. Qwen Team. *Qwen/Qwen3-1.7B-Base model card*. https://huggingface.co/Qwen/Qwen3-1.7B-Base
