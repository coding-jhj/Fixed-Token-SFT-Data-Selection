# 고정 토큰 예산에서의 데이터 선택 효과

**저자:** 정환주  
**부제:** 품질·다양성 기반 선택이 영어 지시 수행에 미치는 영향  
**작성일:** 2026-09-14

## 초록

Instruction-tuning 데이터 선택은 품질과 양의 문제로 자주 논의되지만, 선택 정책마다 학습 토큰 수가 다르거나 평가 예시가 달라지면 정책의 효과를 분리하기 어렵습니다. 본 연구는 포맷팅 후 학습 토큰 수를 정확히 고정한 자원 제약 조건에서 층화 random, quality-only, quality-plus-semantic-diversity 선택을 비교합니다. Qwen/Qwen3-1.7B-Base에 4-bit NF4 QLoRA supervised fine-tuning을 적용하고, 세 정책의 primary 비교는 seed 13과 42에서 실행했습니다. 각 adapter는 정확히 1,000,000개의 formatted training token을 사용했습니다. 사전에 지정한 IFEval prompt-level strict accuracy에서 diversity의 2-seed 평균은 11.46%(SD 0.74)로 random의 11.72%(SD 1.84)보다 낮았고, paired bootstrap 차이는 -0.26 percentage points, 95% confidence interval은 [-3.39, 2.60]이었습니다. 따라서 quality-plus-diversity가 영어 instruction following을 안정적으로 개선한다는 주가설은 지지되지 않았습니다. 추가로 random과 diversity만 seed 2026에서 재평가했으며, 세 seed를 함께 본 diversity-minus-random의 IFEval 차이는 -0.69 percentage points, 95% confidence interval은 [-3.30, 1.91]이었습니다. Secondary BBH에서는 세 seed 차이가 +7.72 percentage points, 95% confidence interval [3.40, 12.19]로 나타났지만 탐색적 결과로 유지합니다. 동일 subset의 frozen base-model baseline은 IFEval 13.02%, GSM8K 43.75%, BBH 28.70%였습니다. 모든 후속 출력은 구조적으로 유효했지만 generation-cap 도달 가능성 진단은 seed 2026 adapter에서 1,318/1,328행, 기존 6개 adapter에서 3,928/3,984행으로 높았습니다. 따라서 본 연구는 결과를 일반적 우위가 아닌 task-dependent evidence로 해석합니다.

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
- 세 번째 seed, frozen base-model baseline, 고정 subset·manifest·출력 회귀검사를 추가하고, 실제 사람이 평가하지 않은 human audit은 준비 자료와 미실행 상태를 분리해 기록했습니다.

본 연구는 특정 데이터 선택기가 모든 모델과 모든 task에서 최적이라는 주장을 하지 않습니다. 하나의 English data pool, 하나의 base model, 하나의 embedding model, 하나의 토큰 예산에서 관찰되는 효과를 검증하는 resource-constrained study입니다.

## 2. 관련 연구

### 2.1 Instruction post-training

InstructGPT는 supervised demonstration과 preference feedback을 사용해 언어 모델이 사용자의 지시를 따르도록 학습하는 대표적인 post-training 패턴을 제시했습니다 [1]. Direct Preference Optimization은 별도의 reward-model pipeline 없이 preference data를 직접 최적화하는 목적함수를 제안했습니다 [2]. Tülu 3는 공개 post-training recipe에서 data mixture, decontamination, staged training, held-out evaluation을 함께 관리해야 한다는 점을 보여줍니다 [3].

이 연구들의 공통점은 학습 objective만으로 결과를 설명하지 않고 데이터 구성과 평가 설계를 결과 해석의 일부로 취급한다는 점입니다. 본 연구는 그중 데이터 선택 단계에 초점을 두고, 다른 조건을 가능한 한 고정한 뒤 선택 정책의 추가 효과를 확인합니다.

### 2.2 품질과 다양성 기반 데이터 선택

Alignment data의 유용한 특성을 분석하는 연구는 단순한 데이터 양보다 응답의 품질, 난이도, 유효성, 학습 대상 행동과의 관련성이 중요할 수 있음을 지적합니다 [4]. Robust instruction tuning 연구는 데이터 다양성이 특정 형식이나 문구에 과적합되는 위험을 낮추고 새로운 task로의 전이를 돕는지 조사했습니다 [5].

그러나 quality와 diversity를 결합한 점수가 항상 일반화 성능의 개선으로 이어지는 것은 아닙니다. Quality proxy가 실제 유용성을 충분히 반영하지 못하거나, diversity가 primary task에 필요한 행동을 희석할 수 있습니다. 본 연구는 새로운 universal quality function을 제안하지 않고, heuristic quality score와 embedding clustering을 고정하여 추가적인 측정 가능 이점이 있는지 검증합니다.

이 선행 연구에서 남는 실험적 공백은 선택 정책의 효과와 학습량의 효과를 분리하는 일입니다. 행 수가 같아도 예시 길이가 다르면 모델이 본 실제 token 수가 달라지고, benchmark의 평가 subset이 조건마다 다르면 데이터 선택 효과와 평가 난이도 차이를 구분하기 어렵습니다. 따라서 본 연구는 더 복잡한 selector를 제안하기보다 비교의 단위를 formatted training token으로 고정하고, 같은 평가 ID를 모든 adapter에 반복 적용하는 설계를 택했습니다.

이 설계는 두 가지 질문을 분리합니다. 첫째, quality-plus-diversity가 평균 성능을 높이는가를 primary IFEval 지표로 확인합니다. 둘째, 평균값이 비슷하더라도 특정 task에 대한 취약성을 줄이는지 BBH task 단위와 paired contrast로 확인합니다. 결과가 두 질문에 모두 답하지 못하더라도, 어느 단계에서 주장이 약해지는지를 재현 가능한 기록으로 남기는 것이 본 연구의 범위입니다.

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

선택 정책의 차이는 후보를 정렬하는 순서에만 두었습니다. 각 source와 length stratum의 quota는 전체 후보 token 비율에서 계산했고, 그 quota 아래에서 선택 순서를 적용한 뒤 마지막에 whole-example correction을 수행했습니다. 그러므로 quality와 diversity의 선택 행 수가 달라지는 것은 오류가 아니라 예시 길이 분포와 정확한 token 합계를 동시에 만족한 결과입니다. 이 절차는 선택 정책이 더 많은 optimizer update를 얻어서 유리해지는 경로를 줄이지만, 서로 다른 example composition 자체를 제거하지는 않습니다.

구현에 사용한 quality proxy를 식으로 쓰면 다음과 같습니다.

`Q(x) = 0.25E(x) + 0.25R(x) + 0.20L(x) + 0.20P(x) + 0.10A(x)`

여기서 `E(x)`는 영어 신호와 ASCII 비율의 평균, `R(x)`는 assistant response의 단어 수를 80단어 기준으로 정규화한 값, `L(x)`는 32~1,536 token 구간에서 1.0이고 그 밖의 허용 길이에서 0.7인 값입니다. `P(x)`는 반복 억제 점수이고 `A(x)`는 NUL 또는 장문자 반복 artifact가 없을 때 1.0인 값입니다. 점수는 소수점 여섯째 자리에서 반올림했으며 0.55 미만인 행은 세 정책에서 모두 제외했습니다. 이 식은 사람의 정답성·사실성 판단을 대체하지 않는다는 점을 명시해야 합니다.

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

여덟 개 학습 manifest와 IFEval train, GSM8K test, BBH test 사이의 exact overlap과 normalized `SequenceMatcher` threshold 0.92 이상의 near overlap을 검사했습니다. 기존 subset과 확장 subset 모두에서 모든 manifest의 exact·near overlap은 0이었습니다. 이 검사는 학습 데이터가 평가 문항을 직접 포함하지 않는지 확인하는 절차이며, 의미적으로 유사한 모든 문장을 제거했다는 주장은 아닙니다.

### 3.6 고정 평가 subset과 채점

최종 비교는 모든 adapter에 동일한 deterministic subset을 사용했습니다. Subset selection seed는 2026이며 subset JSONL의 SHA-256 hash와 선택 스크립트를 보존했습니다. IFEval은 instruction ID family별 coverage-first 방식으로 192개 prompt를 뽑았습니다. GSM8K는 prompt length를 네 quantile로 나누어 256개를 뽑았습니다. BBH는 27개 task에서 각각 8개씩 뽑아 216개로 구성했습니다.

| Benchmark | 원천 규모 | 고정 subset | 선택 정책 | 생성 한도 |
|---|---:|---:|---|---:|
| IFEval | 541 train prompts 확인 | 192 prompts | instruction family coverage-first | 512 new tokens |
| GSM8K | 1,319 test rows | 256 prompts | prompt length 4-quantile | 128 new tokens |
| BBH | 6,511 rows, 27 tasks | 216 prompts | task별 8개 | 128 new tokens |

추가 reliability protocol은 동일한 최종 subset에서 IFEval 1,024, GSM8K·BBH 256 new tokens를 사용했습니다. 이 protocol에는 기존 6개 adapter의 long-generation sensitivity, seed 2026의 random/diversity adapter, frozen base model을 포함했습니다. 이 결과는 원래 primary protocol을 대체하지 않습니다. 별도로 확장 subset은 IFEval 384, GSM8K 512, BBH 432(task별 16개)로 생성하고 contamination을 재검사했으며, seed 2026의 random/diversity adapter 성능도 같은 별도 protocol에서 분석했습니다.

Primary metric은 IFEval prompt-level strict accuracy입니다. IFEval instruction-level strict accuracy, GSM8K exact-match accuracy, BBH normalized accuracy, BBH task macro-average, BBH minimum task accuracy를 secondary 또는 robustness metric으로 사용했습니다. 모든 평가에서 greedy decoding(`do_sample=False`)과 batch size 8을 사용했습니다. IFEval은 instruction-level 판정도 함께 산출했으며, GSM8K와 BBH는 평가기 구현의 exact-match와 normalized answer 판정을 사용했습니다.

### 3.7 통계 분석과 무결성 검사

각 strategy의 평균과 표준편차는 seed 13과 42의 adapter 점수에서 계산했습니다. Paired bootstrap은 동일 evaluation ID의 두 strategy 출력을 짝지은 뒤, 먼저 두 seed 점수를 ID별로 평균하고 10,000회 재표집했습니다. Bootstrap seed는 2026입니다. 보고한 primary paired contrast는 quality minus random과 diversity minus random입니다. 추가 seed 분석은 quality seed 2026이 없으므로 random 대 diversity에만 적용하고, seed별 차이와 세 seed를 ID별로 평균한 paired bootstrap을 별도 표에 기록했습니다. Base model은 strategy 평균이나 paired strategy contrast에 포함하지 않았습니다.

최종 결과는 6개 adapter와 3개 benchmark의 18개 output file을 대상으로 검사했습니다. 파일별 예상 행 수와 실제 행 수, JSON parse error, duplicate ID, missing ID, extra ID, malformed score object, empty response를 확인했습니다. Decoded text를 다시 encoding한 token count가 generation cap 이상이면 possible truncation으로 표시했습니다. 이 flag는 원래 generated token ID가 저장되지 않은 상태에서 수행한 보수적 진단이므로, 모든 사례가 실제 truncation이었다고 단정하지 않습니다.

### 3.8 분석 단위와 재현성 판정

분석의 기본 단위는 adapter, benchmark, evaluation ID의 세 겹으로 고정했습니다. 평균과 표준편차는 두 seed의 adapter 수준 점수에서 계산하고, paired bootstrap은 동일한 evaluation ID를 먼저 두 seed에 걸쳐 평균한 뒤 재표집했습니다. 이 순서는 seed 간 변동을 별도의 독립 표본으로 과대 계산하지 않으면서도, 같은 문항에 대한 정책 간 차이를 직접 비교하기 위한 선택입니다. 따라서 보고된 confidence interval은 모델·데이터·benchmark 전체로 일반화되는 불확실성 구간이 아니라, 고정 subset에서의 paired contrast 구간입니다.

무결성 검사는 성능 점수와 분리했습니다. 각 파일에 대해 예상 행 수, ID 중복·누락·추가, JSON parse error, score object 형식, 빈 응답을 검사한 뒤에만 점수를 집계했습니다. 반면 possible truncation은 구조적 오류가 아니라 길이 진단으로 취급했습니다. 이 분리는 유효한 JSON을 생성했다는 사실과 충분히 긴 답변을 생성했다는 주장을 혼동하지 않기 위해 필요합니다.

### 3.9 계획 대비 실제 실행 범위

| 변경 항목 | 계획 | 실제 실행 | 해석상 조치 |
|---|---|---|---|
| IFEval 전체 평가 | 541 prompts 전체 | 183행에서 비용 문제로 중단 | partial run은 diagnostic으로만 보존 |
| 최종 평가 | full benchmark 우선 | 고정 balanced subset 사용 | 최종 표에 subset임을 명시 |
| 추가 seed | seed 2026 추가 | random·diversity만 완료 | 3-policy 3-seed replication으로 과장하지 않음 |
| human audit | 200-example quality audit | 수행하지 않음 | quality score를 heuristic으로 기술 |
| 회귀 평가 | artifact·schema·token·문서 검사 | 완료 | general-capability 개선과 구분 |
| base baseline | 동일 subset frozen baseline | 완료 | full official benchmark와 구분 |
| 확장 subset | 2배 fixed subset | subset·오염 검사·random/diversity 성능 평가 완료 | primary 및 3-seed 결과와 별도 protocol로 보고 |

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

추가 seed 2026의 manifest와 학습은 random 995행, diversity 1,027행으로 각각 정확히 1,000,000 formatted token을 사용했습니다. 두 adapter 모두 동일한 4-bit NF4 QLoRA 조건에서 1 epoch equivalent procedure로 학습했습니다. Quality seed 2026은 실행하지 않았으므로 이 추가 seed는 random 대 diversity robustness 비교로만 사용합니다.

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

추가 long-generation protocol도 같은 ID·score schema 검사를 통과했습니다. Seed 2026 random/diversity는 각각 664행으로 총 1,328행이며, random은 656행, diversity는 662행이 possible truncation으로 표시되었습니다. Frozen base model은 664행 중 297행이 표시되었습니다. 이 수치는 구조적 validity와 별도의 길이 진단이며, 원래 2-seed primary output의 completeness를 변경하지 않습니다.

확장 subset의 2개 adapter와 3개 benchmark, 총 2,656행도 모두 구조적으로 완료되었습니다. Random은 1,318행, diversity는 1,322행이 possible truncation으로 표시되었습니다. IFEval·GSM8K·BBH의 cap 도달 가능성은 각각 random에서 374·512·432행, diversity에서 381·511·430행이었습니다. 따라서 확장 subset에서도 출력 무결성과 충분한 생성 길이는 별개의 축으로 관리했습니다.

### 4.3 주요 평가 결과

| 선택 정책 | IFEval strict | IFEval instruction strict | GSM8K | BBH | BBH task macro | BBH worst task |
|---|---:|---:|---:|---:|---:|---:|
| Random | 11.72% ± 1.84 | 37.89% ± 1.51 | 20.12% ± 1.38 | 24.07% ± 1.96 | 24.07% ± 1.96 | 0.00% ± 0.00 |
| Quality | 10.94% ± 1.47 | 36.46% ± 1.18 | 21.09% ± 5.52 | 26.85% ± 1.31 | 26.85% ± 1.31 | 0.00% ± 0.00 |
| Diversity | 11.46% ± 0.74 | 36.22% ± 0.17 | 22.85% ± 3.59 | 33.56% ± 0.33 | 33.56% ± 0.33 | 0.00% ± 0.00 |

값은 두 seed의 평균 ± 표준편차입니다. IFEval과 GSM8K/BBH percentage는 전체 원천 distribution이 아니라 고정 subset에서 계산했습니다. BBH task macro는 27개 task에 동일한 가중치를 주며, 각 task에서 동일한 8개 예시를 사용했습니다. 모든 조건에서 적어도 한 task의 정답 수가 0이었기 때문에 BBH minimum task accuracy는 세 정책 모두 0이었습니다.

### 4.4 Seed별 결과와 변동성

평균값만으로는 두 seed에서 효과의 방향이 같았는지 확인하기 어렵습니다. 따라서 각 adapter의 핵심 benchmark 점수를 별도로 제시합니다. 이 표의 IFEval, GSM8K, BBH는 각각 동일한 고정 subset에서 계산되었으며, 별도의 full benchmark 성능을 뜻하지 않습니다.

| 정책 | Seed | IFEval prompt strict | GSM8K | BBH | BBH task macro |
|---|---:|---:|---:|---:|---:|
| Random | 13 | 13.02% | 21.09% | 25.46% | 25.46% |
| Quality | 13 | 11.98% | 17.19% | 27.78% | 27.78% |
| Diversity | 13 | 11.98% | 25.39% | 33.33% | 33.33% |
| Random | 42 | 10.42% | 19.14% | 22.69% | 22.69% |
| Quality | 42 | 9.90% | 25.00% | 25.93% | 25.93% |
| Diversity | 42 | 10.94% | 20.31% | 33.80% | 33.80% |

Diversity와 random의 seed별 차이는 IFEval prompt strict에서 seed 13은 -1.04 percentage points, seed 42는 +0.52 percentage points였습니다. 반면 BBH에서는 각각 +7.87과 +11.11 percentage points였고, GSM8K에서는 +4.30과 +1.17 percentage points였습니다. 이 비교는 BBH와 GSM8K의 방향이 두 seed에서 일관되었다는 점을 보여주지만, IFEval primary metric에서 일관된 개선이 없었다는 결론도 함께 뒷받침합니다.

| 정책 | Seed | IFEval prompt strict | GSM8K | BBH |
|---|---:|---:|---:|---:|
| Random | 2026 | 9.90% | 35.94% | 34.26% |
| Diversity | 2026 | 8.33% | 43.36% | 38.43% |

Seed 2026에서도 IFEval prompt strict의 diversity-minus-random은 -1.56 percentage points였고, GSM8K는 +7.42, BBH는 +4.17 percentage points였습니다. 세 seed를 random과 diversity에 한정해 ID별로 평균한 paired bootstrap은 IFEval -0.69 percentage points, 95% CI [-3.30, 1.91], GSM8K +4.30 points, 95% CI [1.04, 7.68], BBH +7.72 points, 95% CI [3.40, 12.19]였습니다. 이는 primary IFEval H1을 지지하지 않지만, BBH와 GSM8K의 탐색적 방향이 추가 seed에서 유지되었음을 보여줍니다.

### 4.4.1 확장 subset robustness

확장 subset에서는 IFEval 384개, GSM8K 512개, BBH 432개를 사용해 seed 2026의 random과 diversity를 다시 비교했습니다. 아래 구간은 세 seed를 합친 결과가 아니라, 확장 subset의 동일 example ID를 사용한 단일 seed의 paired bootstrap입니다.

| 정책 | IFEval prompt strict | IFEval instruction strict | GSM8K | BBH |
|---|---:|---:|---:|---:|
| Random (seed 2026) | 18.23% | 36.04% | 40.23% | 32.87% |
| Diversity (seed 2026) | 19.53% | 36.48% | 45.90% | 39.12% |
| Diversity - random | +1.30 pp [-2.34, 4.95] | +1.48 pp [-1.95, 4.90] | +5.66 pp [0.39, 10.94] | +6.25 pp [1.62, 11.11] |

확장 subset에서 diversity의 방향은 네 지표에서 모두 random보다 높았지만, IFEval 구간은 0을 포함했습니다. GSM8K와 BBH 구간은 이 고정된 단일 seed·subset에서 0을 포함하지 않았으나, 이는 3-seed 일반화 검정이나 full benchmark 결과가 아니며 generation-cap 진단의 영향을 받습니다. 따라서 확장 subset은 primary 결론을 뒤집는 증거가 아니라, 평가 subset 크기에 따른 robustness를 확인하는 보조 증거로 취급합니다.

### 4.5 Paired contrast

| Metric | Quality - random | Diversity - random |
|---|---:|---:|
| IFEval prompt strict | -0.78 pp [-4.43, 2.86] | -0.26 pp [-3.39, 2.60] |
| IFEval instruction strict | -2.13 pp [-5.64, 1.30] | -1.35 pp [-4.30, 1.61] |
| GSM8K accuracy | +0.98 pp [-2.93, 4.88] | +2.73 pp [-0.98, 6.45] |
| BBH accuracy | +2.78 pp [-1.85, 7.64] | +9.49 pp [4.63, 14.58] |

구간은 두 seed별 score를 example ID별로 평균한 뒤 수행한 paired 95% bootstrap confidence interval입니다. 네 contrast 중 0을 포함하지 않는 것은 secondary BBH에서 diversity가 random보다 높은 경우뿐입니다. Primary IFEval contrast에는 0이 포함되므로 주가설 H1을 지지할 근거로 사용할 수 없습니다.

### 4.5.1 Frozen base-model baseline

동일한 최종 subset과 long-generation 설정에서 adapter를 적용하지 않은 base model도 평가했습니다. Base model의 IFEval prompt strict는 13.02%, instruction strict는 38.95%, GSM8K는 43.75%, BBH는 28.70%였습니다. Base model은 IFEval과 GSM8K에서는 seed 2026 adapter보다 높았고, BBH에서는 random과 diversity adapter보다 낮았습니다. 이 baseline은 SFT 자체의 변화와 selection policy 간 차이를 분리하는 참고값이며, strategy 평균이나 paired strategy contrast에 포함하지 않았습니다.

[그림 2. Paired contrast와 95% confidence interval.]

### 4.6 보조 지표와 안정성

Diversity는 BBH에서 random보다 9.49 percentage points 높았고, 두 seed 표준편차도 0.33으로 낮았습니다. 그러나 동일한 diversity 정책이 IFEval에서는 random보다 낮았고, IFEval instruction-level strict에서도 -1.35 percentage points였습니다. 따라서 diversity의 효과는 benchmark가 측정하는 행동에 의존할 가능성이 있습니다.

Quality의 GSM8K 표준편차는 5.52로 random의 1.38보다 컸습니다. 이 차이는 quality policy가 수학 문항에 안정적인 이점을 제공한다고 보기 어렵게 합니다. 다만 두 seed만으로 분산의 신뢰할 만한 추정치를 얻기는 어렵기 때문에, 이 관찰은 후속 seed 추가를 요구하는 신호로만 보고합니다.

### 4.7 BBH task 수준 결과

BBH의 전체 평균 차이가 소수의 task에만 의존하는지 확인하기 위해 27개 task를 seed별로 비교했습니다. 각 task에는 8개 example만 있으므로 한 task의 정확도는 12.5 percentage points 단위로 변합니다. 이산적인 점수 구조를 고려하지 않고 작은 차이를 정밀한 우열로 해석하지 않기 위해, task별 방향과 동률을 함께 집계했습니다.

| 비교 단위 | Diversity > random | 동일 | Diversity < random |
|---|---:|---:|---:|
| Task-seed pair 54개 | 31 | 15 | 8 |
| Task 27개 | 방향 요약은 seed pair 기준 | - | - |

54개 task-seed 비교 중 diversity가 random보다 높은 경우는 31개, 동률은 15개, 낮은 경우는 8개였습니다. 이는 BBH 평균 상승이 한 seed의 단일 task에만 의해 만들어진 것은 아님을 시사하지만, task당 8개라는 작은 표본 때문에 강한 task별 일반화 주장으로 이어지지는 않습니다. 특히 BBH worst task는 모든 정책에서 0이었으므로, 평균 상승과 최저 task 개선은 서로 다른 결과로 보고해야 합니다.

[그림 1. 전략별 평균과 seed 표준편차.]

## 5. 논의

### 5.1 연구 질문에 대한 답

본 조건에서 quality-plus-diversity 선택은 고정된 1,000,000-token QLoRA budget 아래 primary IFEval prompt-level strict accuracy를 개선하지 못했습니다. Diversity의 평균은 random보다 0.26 percentage points 낮았고, paired confidence interval은 [-3.39, 2.60]으로 0을 포함했습니다. Quality-only 역시 primary metric에서 random보다 낮았습니다. 그러므로 H1은 지지되지 않습니다.

추가 seed를 random과 diversity에 한정해 포함해도 같은 방향입니다. 세 seed paired estimate는 -0.69 percentage points이고 95% CI는 [-3.30, 1.91]이므로, 추가 seed는 primary 결론을 뒤집지 않습니다. 다만 quality seed 2026이 없으므로 이것을 세 정책의 완전한 3-seed replication으로 해석하지 않습니다.

H2도 지지되지 않습니다. BBH minimum task accuracy는 모든 정책과 두 seed에서 0이었습니다. 이 지표는 최저 성능 task를 확인하는 stress signal로는 의미가 있지만, 현재의 task별 표본과 모델 규모에서는 정책 차이를 구분하지 못했습니다.

### 5.2 왜 BBH와 IFEval의 방향이 달랐는가

Diversity가 BBH에서만 유리하게 나타난 것은 본 실험의 관찰입니다. 가능한 설명으로는 embedding cluster round-robin이 heterogeneous reasoning pattern의 범위를 넓혀 BBH task coverage에 도움을 주었을 가능성이 있습니다. 그러나 이는 결과에 대한 inference이며, 본 연구는 cluster별 causal ablation이나 task별 coverage mediation analysis를 수행하지 않았습니다.

반대로 IFEval strict는 지시 형식과 constraint를 정확히 만족하는지를 측정합니다. Semantic diversity가 늘어도 선택된 예시가 IFEval의 특정 formatting behavior를 더 잘 가르친다는 보장은 없습니다. 현재 결과만으로 diversity가 instruction following에 해롭다고 결론 내릴 수도 없습니다. Primary subset의 generation cap 진단과 2-seed 설계가 함께 존재하기 때문입니다.

### 5.3 연구·엔지니어링 관점의 의미

이번 비교에서 가장 강한 결론은 특정 selector의 우승이 아니라 비교 방법에 관한 것입니다. Fixed-token accounting을 적용하면 row count가 다른 manifest도 공정하게 비교할 수 있습니다. Source-level license review와 revision pinning은 나중에 결과를 재생성할 수 있는 범위를 명확히 합니다. Deterministic subset, paired ID, raw output, validation report를 보존하면 음성 결과도 실험 오류와 구분할 수 있습니다.

실무적으로는 하나의 quality score에 의존하여 데이터를 줄이기보다, 목표 task의 행동과 subset 설계를 함께 검토해야 합니다. Diversity가 secondary benchmark에서 높은 점수를 얻었다는 이유만으로 데이터 선택 정책을 운영에 바로 적용하기보다, 긴 generation cap과 추가 seed로 효과를 먼저 확인해야 합니다.

### 5.4 Seed별 효과의 일관성

Seed별 결과는 평균값을 해석할 때 필요한 경계를 보여줍니다. IFEval prompt strict에서 diversity는 seed 13과 2026에서는 random보다 낮았고 seed 42에서만 높았습니다. BBH와 GSM8K의 diversity 우위는 세 seed에서 모두 나타났습니다. 따라서 “diversity가 모든 benchmark에서 안정적으로 개선된다”는 주장은 성립하지 않지만, “BBH subset에서 관찰된 차이가 seed 하나의 우연한 상승뿐이다”라고 단정하기도 어렵습니다.

이 비대칭성은 primary metric을 사전에 지정한 이유를 설명합니다. 여러 benchmark 중 유리한 결과만 선택하면 선택 정책의 효과를 과장할 수 있으므로, IFEval에서의 paired contrast와 confidence interval을 주가설의 판정 기준으로 유지했습니다. BBH의 일관된 상승은 후속 검증의 우선순위를 높이는 탐색적 신호이지, H1을 뒤집는 근거는 아닙니다.

### 5.5 생성 한도와 점수 해석의 경계

이번 결과에서 가장 중요한 운영상 위험은 점수 자체보다 생성 길이입니다. 기존 6개 adapter의 IFEval에서는 1,152개 출력 중 1,114개, GSM8K에서는 1,536개 중 1,530개, BBH에서는 1,296개 중 1,293개가 possible truncation으로 표시되었습니다. 추가 seed 2026에서도 1,328개 중 1,318개가 표시되었습니다. 이 비율은 응답이 구조적으로 파싱되었다는 사실과 문제를 끝까지 풀었다는 사실을 분리해서 보아야 함을 뜻합니다.

이 진단은 decoded text를 다시 tokenizer에 넣어 generation cap 이상인지 확인한 보수적 flag입니다. 원래 generated token ID가 보존되지 않았기 때문에 실제 truncation의 확정 판정은 아닙니다. 그러나 GSM8K와 BBH에서 cap에 도달한 비율이 특히 높다는 사실은 후속 실험에서 더 긴 generation limit을 먼저 검증해야 한다는 충분한 근거가 됩니다. 이 절차 없이 현재의 BBH 상승을 완전한 reasoning 능력의 개선으로 표현해서는 안 됩니다.

## 6. 한계와 후속 실험

첫째, 모델은 1.7B parameter에 불과하므로 더 큰 모델에 결과가 전이된다고 보장할 수 없습니다. 둘째, quality proxy는 heuristic이며 200-example human audit용 blind sheet와 rubric은 준비했지만 실제 사람의 rating은 수행하지 않았습니다. 셋째, 최종 비교는 전체 benchmark distribution이 아닌 resource-constrained deterministic subset에 기반합니다. 넷째, 3,984개 출력 중 3,937개와 seed 2026 추가 출력 1,328개 중 1,318개가 generation cap에 도달했을 가능성이 있어, 특히 GSM8K와 BBH 결과에 불완전한 응답의 영향이 있을 수 있습니다.

다섯째, seed 2026은 random과 diversity만 실행했으므로 세 정책의 완전한 3-seed replication이 아닙니다. 여섯째, frozen base-model baseline은 동일한 subset에서 수행했지만 full official benchmark baseline은 아닙니다. 일곱째, artifact·schema·token·문서 회귀검사는 추가했지만 broad helpfulness, fluency, safety를 측정하는 general-capability regression suite는 수행하지 않았습니다. 여덟째, 2배 확장 subset의 random/diversity 성능도 완료했지만 single-seed·별도 subset 결과이므로 primary 및 3-seed 결과와 분리해 해석합니다. 아홉째, 하나의 English data pool, 하나의 embedding model, 하나의 token budget만 비교했습니다. 마지막으로 BBH raw row는 conversion card의 redistribution license가 audit에서 명확하지 않아 reproduction ZIP에 포함하지 않았습니다.

남은 후속 과제는 generation cap을 더 높인 stress test, 실제 human rating, full benchmark, general-capability regression suite, 더 큰 model과 다른 data pool 재현입니다. 이 과제들은 현재 결과를 대체하지 않고 별도 protocol로 관리해야 합니다.

## 7. 결론

고정된 1,000,000 formatted-token QLoRA budget과 Qwen3-1.7B-Base 조건에서 quality-plus-diversity 선택은 사전에 정한 IFEval prompt-level strict accuracy를 안정적으로 개선하지 못했습니다. 추가 seed를 random과 diversity에 한정해도 이 primary 결론은 유지되었습니다. 확장 subset의 단일 seed robustness에서는 diversity가 IFEval·GSM8K·BBH에서 모두 높았지만, IFEval 구간은 0을 포함하고 GSM8K·BBH 결과도 generation-cap과 subset 범위를 고려해야 하므로 일반적 우위로 해석할 수 없습니다. Frozen base-model baseline은 SFT와 selector 효과를 분리하는 참고점을 제공하지만, broad capability 향상을 입증하지는 않습니다. 따라서 본 연구의 결론은 “diversity가 항상 좋다”가 아니라, 데이터 선택 정책의 효과가 task와 평가 설계에 의존하며 고정 예산·고정 subset·무결성 검사를 갖춘 비교가 필요하다는 것입니다.

본 연구는 여섯 개 primary adapter, seed 2026 random/diversity adapter, frozen base-model baseline, 확장 subset robustness 평가의 기록과 exact-token manifest, validation report, 3-seed random/diversity paired bootstrap, artifact regression report, human-audit preparation materials를 함께 제공합니다. 실제 human rating, full official benchmark, general-capability regression suite는 수행 범위 밖으로 명시합니다.

## 데이터와 코드 공개 및 재현 절차

Workspace의 `work/selection_manifests/`와 `work/selection_manifests_seed2026/`에는 frozen selection manifest와 token summary가 있습니다. `work/evaluation_subsets_final/`에는 primary subset, `work/evaluation_subsets_expanded/`에는 확장 subset manifest와 hash가 있습니다. `work/main_*`에는 adapter run summary와 training log가 있고, `work/evaluation_final_balanced/`에는 primary output, `work/evaluation_seed2026_long_generation/`에는 추가 seed output, `work/evaluation_base_long_generation/`에는 frozen base output, `work/evaluation_expanded_long_generation/`에는 확장 subset output이 있습니다. `work/results_final/`, `work/results_seed2026_long_generation/`, `work/results_seed_robustness/`, `work/results_base_long_generation/`, `work/results_expanded_long_generation/`, `work/results_reliability/`에는 processed metric과 validation 자료가 있습니다.

분석 코드는 `src/analyze_final_results.py`, `src/analyze_followup_results.py`, `scripts/analyze_seed_robustness.py`, frozen evaluation 코드는 `src/evaluate_frozen.py`, subset 생성 코드는 `src/make_eval_subset.py`, artifact 회귀검사는 `scripts/validate_followup_artifacts.py`, human audit 준비는 `scripts/prepare_human_audit.py`에 기록했습니다. Reproduction ZIP에는 manuscript, PDF, 실행 스크립트, 환경 고정 파일, processed result와 validation 자료를 포함했습니다. Raw BBH row와 raw benchmark JSONL은 redistribution license와 패키지 정책 때문에 제외했으며, 대신 revision, subset hash, task allocation, download/regeneration 절차를 포함했습니다.

재현 시 먼저 고정된 Python 환경과 model/dataset revision을 확인하고, selection manifest의 token 합계와 example ID uniqueness를 검사해야 합니다. 이후 adapter별 output completeness를 검증한 다음 분석 스크립트를 실행합니다. Partial IFEval run과 32-row pilot은 pipeline 진단 기록이므로 final table의 통계에 합산하지 않습니다.

## 부록 A. 재현 체크리스트

아래 체크리스트는 결과를 다시 만들 때 확인해야 할 순서를 실행 산출물과 연결합니다. 각 단계는 다음 단계로 넘어가기 위한 검증 조건을 함께 갖습니다. 이 순서를 지키면 selection manifest의 변경, 평가 subset의 변경, 분석 결과의 변경을 서로 구분할 수 있습니다.

| 단계 | 확인 항목 | 통과 조건 | 주요 산출물 |
|---|---|---|---|
| 1. 환경 | Python, CUDA, library version, GPU | 고정 환경과 GPU 정보 일치 | `reproduction/environment.lock` |
| 2. 선택 | source revision, hard filter, quality floor | 6개 manifest의 token 합계가 각각 1,000,000 | `work/selection_manifests/` |
| 3. 오염 검사 | IFEval train, GSM8K test, BBH test overlap | exact·near overlap 모두 0 | `work/selection_manifests/*contamination*` |
| 4. 평가 | adapter별 row 수와 score object | 18개 파일이 예상 ID를 한 번씩 포함 | `work/results_final/validation.json` |
| 5. 분석 | seed 평균, paired bootstrap, task 요약 | 분석 CSV와 figure가 동일 입력에서 재생성 | `work/results_final/` |
| 6. 후속 평가 | seed 2026, base baseline, 확장 subset, protocol 분리 | 추가 결과가 primary 결과를 덮어쓰지 않음 | `work/results_seed2026_long_generation/`, `work/results_base_long_generation/`, `work/results_expanded_long_generation/` |
| 7. 회귀검사 | token·schema·ID·문서·패키지 무결성 | `all_checks_pass: true`, ZIP `testzip=None` | `work/results_reliability/validation_followup.json` |
| 8. 문서 | 원고와 결과 파일의 수치 일치 | PDF와 원고의 결론·한계 일치 | `paper/manuscript.md`, `paper/paper.pdf` |

이 표의 통과 조건은 성능이 높다는 뜻이 아니라, 같은 입력과 같은 판정 규칙으로 결과를 다시 만들 수 있다는 뜻입니다. 특히 raw benchmark output을 공개하지 않는 환경에서는 revision, subset hash, output manifest, validation report를 함께 보존해야 결과의 provenance를 확인할 수 있습니다.

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
