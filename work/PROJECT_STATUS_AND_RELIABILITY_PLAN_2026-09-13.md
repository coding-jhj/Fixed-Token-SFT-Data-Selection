# 프로젝트 현재 상태와 논문 신뢰도 강화 계획

스냅샷: 2026-09-13  
저자: 정환주

이 문서는 작업의 기준을 다음 다섯 항목으로 고정하여 현재 완료 범위, 남은 작업, 필요한 실험량과 예상 시간을 기록합니다.

1. 원고 최종 검토
2. 더 긴 generation 한도의 동일 평가
3. 세 번째 seed 추가
4. 논문 신뢰도 강화
5. 제출본 제작

## 한눈에 보는 현재 상태

| 기준 | 현재 상태 | 완료도 | 남은 핵심 작업 |
|---|---|---:|---|
| 원고 최종 검토 | 기존 2-seed 원고와 PDF의 문장·표·페이지 레이아웃 검토 완료. 후속 결과 통합 전 | 부분 완료 | seed 2026 결과 반영 후 초록·결과·한계 재검토 |
| 긴 generation 평가 | 기존 6개 adapter `3,984/3,984` 완료. 새 한도에서도 possible truncation `3,928/3,984` 진단 | 완료 | 필요하면 더 높은 한도의 2-policy stress test |
| 세 번째 seed | random/diversity 선택과 학습 완료. 평가 `1,192/1,328` 저장 | 진행 중 | diversity BBH `136`개를 `--resume`로 완료 |
| 신뢰도 강화 | 별도 human audit, base-model baseline, regression suite, 큰 subset은 미실행 | 미착수 | 아래 권장 범위에서 선택·실행 |
| 제출본 제작 | 한국어 기존 PDF는 A4 21쪽으로 검증 완료 | 부분 완료 | 후속 결과 통합 PDF. 영문/HWP/Word는 선택 사항 |

현재까지 실제로 완료된 후속 계산은 다음과 같습니다.

- 장문 생성 한도: IFEval `1,024`, GSM8K/BBH `256` new tokens.
- 기존 6개 adapter: 모든 출력 구조 검증 통과.
- 세 번째 seed: random `995` rows, diversity `1,027` rows. 두 adapter 모두 정확히 `1,000,000` formatted training tokens.
- seed 2026 평가: random `664/664` 완료, diversity IFEval `192/192`, GSM8K `256/256`, BBH `80/216` 완료.
- 중간 결과는 row 단위로 보존되어 있으며 `work/HANDOFF_2026-09-13.md`의 `--resume` 명령으로 재개합니다.

## 다섯 기준별 상세 상태

### 1. 원고 최종 검토

기존 한국어 원고는 저자 `정환주`, 제목, 초록, IMRaD 구조, 표와 PDF 전체 페이지를 확인했습니다. A4 고정, 표 overflow, 고아 제목과 페이지별 크기 문제도 기존 기준본에서 수정했습니다.

다만 후속 장문 평가와 seed 2026 결과를 아직 원고에 통합하지 않았습니다. 따라서 현재 PDF는 “기존 2-seed 결과의 검증된 기준본”이지, 최종 후속 결과를 포함한 제출본은 아닙니다.

필요 작업:

- seed 2026 분석 결과를 기존 2-seed primary result와 분리하여 표기
- 초록의 핵심 결과와 결론의 적용 범위 갱신
- generation cap 진단을 결과와 한계에 정확히 반영
- 제목, 저자, 표 캡션, 수치, 들여쓰기와 위계의 최종 육안 검토

예상 시간: `30~60분`.

### 2. 더 긴 generation 한도의 동일 평가

기존 6개 adapter에 대해 평가 subset을 그대로 두고 generation 한도만 늘린 평가를 완료했습니다. 그러나 3,984개 중 3,928개가 새 한도에 도달했을 가능성이 있어, “더 긴 한도로 truncation 문제가 해결되었다”고 쓸 수는 없습니다.

따라서 현재 장문 결과는 다음처럼 사용해야 합니다.

- 기존 결론을 대체하지 않는 sensitivity diagnostic
- IFEval primary conclusion과 GSM8K·BBH 변화를 별도 평가 protocol로 보고
- cap 도달 가능성이 남았다는 한계를 명시

추가로 강한 해석이 꼭 필요하다면, 모든 6개 adapter를 다시 돌리기보다 핵심 비교인 random과 diversity 두 adapter만 `IFEval 2,048`, `GSM8K/BBH 512` 한도로 stress test하는 것이 비용 대비 적절합니다. 예상 시간은 GPU 생성 `2~5시간`, 분석·검증 `30~60분`입니다. 현재 논문을 정리하는 데 필수는 아니며, 제출처가 생성 truncation 해소를 요구할 때 수행합니다.

### 3. 세 번째 seed 추가

선택과 학습은 완료되어 가장 중요한 추가 seed의 비용을 이미 지불했습니다.

- random seed 2026: 선택·학습 완료, 평가 664개 완료
- diversity seed 2026: 선택·학습 완료, 평가 528개 완료
- 남은 평가: diversity BBH 136개

예상 시간:

- 남은 GPU 평가: `25~45분`
- 분석과 validation: `5~10분`

### 4. 논문 신뢰도 강화

현재 논문의 주장을 지탱하는 가장 중요한 비교는 동일한 1,000,000-token budget에서 random, quality, diversity를 비교하는 paired evaluation입니다. 따라서 모든 추가 실험을 한꺼번에 수행할 필요는 없습니다. 다음 순서가 합리적입니다.

| 우선순위 | 작업 | 권장 범위 | 예상 시간 | 필요성 |
|---|---|---|---:|---|
| 1 | seed 2026 완료·분석 | random/diversity, 현재 subset 유지 | `30~55분` | 필수 |
| 2 | regression suite | exact-token, manifest ID, score schema, aggregate, PDF A4/overflow 자동 검사 | `1~2시간` | 강력 권장 |
| 3 | full benchmark baseline | 학습하지 않은 base model을 동일 subset·동일 decoding으로 평가 | GPU·분석 `1~2시간` | 권장 |
| 4 | quality score human audit | source·length·score 구간별 200개, 명시적 rubric, blind rating | 준비·정리 `1~2시간` + rating `3~6시간`/1인 | 조건부 권장 |
| 5 | 큰 평가 subset | 현재 subset의 2배를 random/diversity에 우선 적용 | GPU·분석 `3~6시간` | 선택 |
| 6 | full official benchmark | 공식 전체 split과 license 범위 재확인 | `8시간 이상` 가능 | 현재는 보류 |

#### Human audit의 해석 범위

human audit는 사람이 직접 점수를 매겨야 합니다. 모델 기반 자동 판정이나 작성자의 사후 확인을 human audit라고 부르면 안 됩니다. 한 명이 rating하면 `single-rater audit`으로 보고하고, 두 명이 독립 평가할 때만 inter-rater agreement를 계산합니다.

200개를 한 명이 평가하면 약 `4~8시간`, 두 명이면 총 `8~16시간`을 예상하는 것이 안전합니다. 제가 할 수 있는 범위는 표본 추출, rubric, blind sheet, 집계 코드 준비까지이며, 사람의 판단 자체를 대신 만들 수는 없습니다.

#### 권장 결론

현재 논문 제출을 위해서는 우선순위 1~3까지가 적절합니다. 즉, seed 2026을 마무리하고, regression suite와 base-model baseline을 추가한 뒤 manuscript·PDF·ZIP을 갱신합니다. Human audit는 제출처가 데이터 품질 검증을 요구하거나 사용자가 직접 평가할 시간을 확보할 때 추가합니다. 큰 subset과 full official benchmark는 generation cap과 라이선스 범위를 먼저 정리한 뒤 별도 확장 과제로 두는 것이 좋습니다.

### 5. 제출본 제작

현재 한국어 PDF는 기준본으로 완성되어 있습니다. 후속 결과를 통합하면 최종 제출본 제작에 다음 작업이 남습니다.

- 한국어 PDF 재생성 및 전체 페이지 렌더링 검사: `30~60분`
- 영문 manuscript 변환: 내용 검토 포함 `3~6시간`
- HWP/Word 변환: 원고 구조 재확인 포함 `1~3시간`

영문·HWP·Word는 제출처 요구가 있을 때만 진행합니다. 현재 필수 제출본은 후속 결과가 반영된 한국어 PDF입니다.

## 전체 예상 시간

### 최소한의 신뢰 가능한 제출본

seed 2026 완료·분석, manuscript/PDF/ZIP 통합, regression suite까지 포함합니다.

- 예상: `2~4시간`
- 결과: 현재 연구 질문에 대한 3-seed 보강과 재현·레이아웃 검증이 포함된 한국어 제출본

### 권장 신뢰도 강화 제출본

최소 제출본에 base-model baseline과 single-rater 200-example human audit를 추가합니다.

- 예상: 최소 제출본 `2~4시간` + baseline `1~2시간` + human audit `4~8시간`
- 총 예상: `7~14시간`
- 두 명의 human rater를 사용하면 총 `11~22시간`까지 늘어날 수 있습니다.

### 확장 연구본

권장 제출본에 2배 subset 또는 full official benchmark를 추가합니다.

- 2배 subset: 추가 `3~6시간`
- full benchmark: 데이터 다운로드·라이선스·생성 길이에 따라 `8시간 이상`이며, 현재 작업의 필수 범위가 아닙니다.

## 내일 재개 순서

1. `work/HANDOFF_2026-09-13.md`를 읽고 현재 파일 행 수를 확인합니다.
2. diversity BBH 남은 136개를 `--resume`로 완료합니다.
3. seed 2026 분석과 validation을 실행합니다.
4. 이 문서의 상태표와 인수인계 수치를 갱신합니다.
5. 우선 regression suite와 base-model baseline을 수행할지 결정합니다.
6. 결정된 범위만 manuscript·PDF·README·ZIP에 반영하고 전체 렌더링 검사를 수행합니다.
