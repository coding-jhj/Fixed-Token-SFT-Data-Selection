# T10 Post Training 재현 패키지

이 패키지는 고정 토큰 예산에서 영어 SFT 데이터 선택 정책을 비교한 자원 제약 실험의 최종 산출물입니다. 한국어 연구 원고와 PDF, AI 조언과 최종 판단 기록, 분석 코드, 고정 manifest와 평가 subset manifest, 처리된 결과, validation report, 환경 버전, provenance 기록을 포함합니다.

## 연구 질문

metadata에 기록한 Qwen/Qwen3-1.7B-Base revision과 adapter별 정확히 1,000,000개의 formatted training token을 고정했을 때, quality-plus-semantic-diversity 선택이 stratified random 선택 또는 quality-only 선택보다 held-out instruction-following을 개선하는지 검증합니다.

## 현재 결론

Primary IFEval prompt-level strict에서 diversity minus random 차이는 -0.26 percentage points였고, paired 95% bootstrap interval은 [-3.39, 2.60] percentage points였습니다. Primary hypothesis는 지지되지 않았습니다. Diversity는 secondary BBH subset에서 더 높았지만, 일반적 우위가 아닌 task-dependent evidence로 보고합니다. 기존 평가에서는 3,984개 decoded row 중 3,937개가 generation limit에 도달했을 가능성이 있었습니다. 후속 장문 평가 3,984개도 구조적으로 완료되었지만 3,928개가 새 limit에 도달했을 가능성이 있어, 이 결과는 sensitivity diagnostic으로만 취급합니다. 세 번째 seed 2026은 random/diversity 선택·학습을 완료했고, 평가는 1,192/1,328개까지 진행된 상태입니다.

## 재현 범위

포함된 `scripts/reproduce_analysis.ps1`은 보존된 workspace output을 사용해 validation, aggregate metric, paired bootstrap interval, PDF 생성을 다시 수행합니다. Adapter를 재학습하지는 않습니다. 전체 end-to-end 재실행에는 고정 model과 dataset snapshot, 여섯 adapter checkpoint, raw benchmark row가 추가로 필요합니다. Raw BBH row는 local audit에서 redistribution license가 명확하지 않았으므로 의도적으로 제외했고, revision, 선택 ID, hash, source 재생성 절차를 보존했습니다.

## 고정 프로토콜

- Model: `Qwen/Qwen3-1.7B-Base`, revision은 run summary에 기록하고 benchmark revision은 `data/evaluation_subset_manifest.json`에 기록했습니다.
- Training: 4-bit NF4 QLoRA, 1 epoch, maximum sequence length 2,048, exact formatted-token target 1,000,000.
- Selection policy: stratified random, quality-only, quality-plus-semantic-diversity; primary seed 13과 42, follow-up seed 2026은 random/diversity.
- Evaluation subset: IFEval 192, GSM8K 256, BBH 216; selection seed 2026.
- Primary generation cap: IFEval 512 new tokens; GSM8K와 BBH 128 new tokens. Follow-up sensitivity cap: IFEval 1,024, GSM8K/BBH 256 new tokens. 모두 greedy decoding, batch size 8.
- Bootstrap: 10,000 paired resample, example ID별 두 evaluation seed 평균, seed 2026.

## 파일 구성

- `paper/`: 한국어 manuscript, PDF, AI 조언과 최종 판단 기록.
- `src/`: selection, training, evaluation, contamination, final-analysis code.
- `data/`: selected-manifest summary/CSV와 frozen evaluation subset manifest.
- `results/`: per-seed metric, strategy summary, bootstrap interval, BBH task metric, validation, analysis metadata.
- `environment.lock`: 검증된 연구 환경 버전과 hardware 기록.
- `licenses_and_provenance.md`: source revision과 redistribution 범위.
- `../work/PROJECT_STATUS_AND_RELIABILITY_PLAN_2026-09-13.md`: 다섯 기준별 현재 상태, 신뢰도 강화 범위와 예상 시간.
