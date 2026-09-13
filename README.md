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

## 실험 조건

- Base model: `Qwen/Qwen3-1.7B-Base`
- Training: 4-bit NF4 QLoRA SFT, 최대 길이 2,048 token
- Selection: stratified random, quality-only, quality-plus-semantic-diversity
- Seeds: `13`, `42`
- Evaluation subset: IFEval `192`, GSM8K `256`, BBH `216`

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
- `reproduction/`: 재현 패키지 보조 문서

## 한계

현재 결과는 1.7B base model, 하나의 영어 데이터 풀, 두 개 seed, 고정 subset, 제한된 generation budget에 대한 결과입니다. 추가 seed, 더 긴 생성 한도, human evaluation, full benchmark baseline은 후속 과제로 남아 있습니다.
