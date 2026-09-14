# Fixed Token SFT Data Selection

고정된 학습 토큰 예산에서 데이터 선택 정책이 소형 언어 모델의 영어 지시 수행에 미치는 영향을 비교한 재현 가능한 QLoRA 연구입니다.

저자: 정환주

## 연구 질문

동일한 `1,000,000` formatted training token 예산과 동일한 평가 subset을 사용할 때, `quality-plus-semantic-diversity` 선택이 stratified random 및 quality-only 선택보다 영어 instruction following을 개선하는지 검증했습니다.

## 핵심 결과

- Primary IFEval prompt-level strict accuracy에서 diversity minus random은 `-0.26 pp`였습니다.
- Paired 95% bootstrap interval은 `[-3.39, 2.60]`으로 주가설을 지지하지 않았습니다.
- Secondary BBH subset에서는 diversity minus random이 `+9.49 pp`였지만 일반적 우위가 아닌 task-dependent evidence로 해석했습니다.
- 3,984개 평가 출력은 구조적으로 유효했지만, 3,937개는 generation cap 도달 가능성이 있는 것으로 진단되었습니다.
- random/diversity seed 2026을 추가한 3-seed paired 분석에서 IFEval은 `-0.69 pp` (95% CI `[-3.30, 1.91]`), BBH는 `+7.72 pp` (95% CI `[3.40, 12.19]`)였습니다. quality seed 2026은 수행하지 않았습니다.
- 동일 subset의 frozen base-model baseline은 IFEval `13.02%`, GSM8K `43.75%`, BBH `28.70%`였습니다. adapter strategy 평균에는 포함하지 않았습니다.
- 확장 subset(seed 2026, IFEval 384/GSM8K 512/BBH 432)에서 diversity minus random은 IFEval `+1.30 pp` (95% CI `[-2.34, 4.95]`), GSM8K `+5.66 pp` (`[0.39, 10.94]`), BBH `+6.25 pp` (`[1.62, 11.11]`)였습니다. 이는 단일 seed의 별도 robustness protocol입니다.
- 여덟 개 selected manifest, 총 `8,144` rows의 automatic quality audit에서 모든 row가 message 구조 검사를 통과했고, stored heuristic 재계산 불일치는 `0`건이었습니다. 장문자 반복 flag `1`건과 8단어 미만 assistant response flag `6`건은 review 후보로만 기록했습니다.

## 실험 조건

- Base model: `Qwen/Qwen3-1.7B-Base`
- Training: 4-bit NF4 QLoRA SFT, 최대 길이 2,048 token
- Selection: stratified random, quality-only, quality-plus-semantic-diversity
- Seeds: primary `13`, `42`; follow-up `2026`은 random/diversity만
- Evaluation subset: IFEval `192`, GSM8K `256`, BBH `216`
- Follow-up subset: IFEval `384`, GSM8K `512`, BBH `432`를 별도 robustness protocol로 구성

## 재현

분석 결과와 PDF는 다음 명령으로 재생성할 수 있습니다.

```powershell
.\scripts\reproduce_analysis.ps1
```

전체 산출물과 provenance 정보는 `T10_post_training_reproduction_package_2026-09-13.zip`에 포함되어 있습니다. 원자료 JSONL, 학습 adapter 가중치, tokenizer는 용량과 재배포 라이선스 범위 때문에 저장소에 직접 포함하지 않았습니다.

## 저장소 구조

- `paper/`: 한국어 원고와 PDF
- `src/`: 선택, 학습, 평가, 오염 검사, 최종 분석 코드
- `scripts/`: 분석 재현 및 PDF 생성 스크립트
- `configs/`: 선택 설정
- `work/results_final/`: 최종 metric, paired bootstrap, validation 결과
- `work/results_seed2026_long_generation/`: seed 2026 long-generation 분석 결과
- `work/results_seed_robustness/`: random/diversity 3-seed robustness 분석
- `work/results_base_long_generation/`: frozen base-model baseline 분석 결과
- `work/results_expanded_long_generation/`: 확장 subset random/diversity 분석 결과
- `work/results_reliability/`: token·schema·ID·subset 회귀검사 결과
- `work/human_audit_200/`: blind audit 준비 자료와 automatic/AI-assisted audit의 로컬 결과(사람의 rating 없음)
- `reproduction/`: 재현 패키지 보조 문서

## 후속 검증 현황

- 기존 6개 adapter의 장문 generation 재평가(`IFEval 1,024`, `GSM8K/BBH 256`)는 `3,984/3,984`개 완료했습니다. 다만 `3,928/3,984`개가 새 cap에 도달했을 가능성이 있어 보강 진단으로만 취급합니다.
- random/diversity seed `2026`의 선택과 학습은 정확히 `1,000,000` formatted training tokens로 완료했습니다. 평가는 `1,328/1,328`개 완료했고 모든 구조 검사를 통과했습니다.
- 동일 최종 subset의 frozen base-model baseline `664/664`행을 완료했습니다. long-generation possible truncation 진단은 base `297/664`, seed 2026 adapter `1,318/1,328`입니다.
- 2배 fixed subset을 구성하고 여덟 개 training manifest에 대한 expanded benchmark contamination 검사에서 exact·near overlap `0`을 확인했습니다.
- 확장 subset 성능 평가도 2개 adapter, 3개 benchmark, `2,656/2,656`행 완료했으며 구조 검사를 통과했습니다. Possible truncation은 random `1,318/1,328`, diversity `1,322/1,328`이었습니다.
- 전체 기준별 상태와 신뢰도 강화 계획은 [`work/PROJECT_STATUS_AND_RELIABILITY_PLAN_2026-09-13.md`](work/PROJECT_STATUS_AND_RELIABILITY_PLAN_2026-09-13.md)에 기록했습니다.
- 보존된 중간 결과와 재현 명령은 [`work/HANDOFF_2026-09-13.md`](work/HANDOFF_2026-09-13.md)에 있습니다.

## 한계

현재 primary 결과는 1.7B base model, 하나의 영어 데이터 풀, 두 개 seed, 고정 subset, 제한된 generation budget에 대한 결과입니다. 추가 seed는 random/diversity에 한정되며, 확장 subset 결과도 별도 single-seed robustness evidence입니다. Automatic audit은 구조·표면 패턴·heuristic 재현성만 점검합니다. 200-example blind sheet에는 실제 human rating을 입력하지 않았고, 동일한 Qwen3 base model을 사용한 AI-assisted judge도 200개 중 55개만 두 prompt에서 parse되어 human audit이나 독립 검증으로 볼 수 없습니다. Full official benchmark·general-capability regression suite도 완료하지 않았습니다.
