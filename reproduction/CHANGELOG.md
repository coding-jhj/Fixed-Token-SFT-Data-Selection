# 변경 기록

## 2026-09-14 후속 신뢰도 강화

- seed 2026 random/diversity frozen evaluation을 `1,328/1,328`행으로 완료했습니다.
- random/diversity 3-seed robustness paired bootstrap을 추가했습니다.
- 동일 final subset에서 frozen base-model baseline `664/664`행을 완료했습니다.
- IFEval 384, GSM8K 512, BBH 432의 expanded subset과 contamination report를 생성했습니다.
- Expanded subset에서 seed 2026 random/diversity 성능 평가 `2,656/2,656`행과 별도 paired analysis를 완료했습니다.
- exact-token, output schema, ID completeness, subset, 문서·패키지 검사를 자동화하고 validation을 통과했습니다.
- 200-example human audit용 blind sheet와 rubric을 준비했지만 실제 human rating은 수행하지 않았습니다.
- 여덟 개 selected manifest, 총 8,144 rows에 대해 message 구조·표면 flag·quality heuristic 재계산 일치 여부를 automatic audit했습니다. 이 결과는 human rating이 아닙니다.
- 로컬 `Qwen3-1.7B-Base`를 이용한 200-example AI-assisted exploratory judge를 별도 실행하며, 두 prompt variant의 일치도와 parse 상태를 보존합니다. 모델은 generation/base 평가와 독립적이지 않으므로 human agreement 증거로 사용하지 않습니다.
- 두 prompt 모두 parse된 항목은 `55/200`개였고 parsed overall score의 `49/55`개가 5/5였습니다. 이 judge는 판별력이 낮아 정량 quality claim에 사용하지 않습니다.
- 결과 protocol을 primary, seed follow-up, base baseline, expanded robustness output으로 분리했습니다.

## 2026-09-13 후속 작업 현황

- 기존 6개 adapter의 장문 generation sensitivity 평가를 `3,984/3,984`개 완료했습니다.
- 장문 평가 결과와 possible truncation 진단을 별도 후속 결과 문서에 기록했습니다.
- random/diversity seed `2026`의 선택과 정확한 `1,000,000` formatted-token 학습을 완료했습니다.
- seed 2026 평가를 `1,192/1,328`개까지 row 단위로 보존하고, `--resume` 재개 절차를 기록했습니다.
- 다섯 개 작업 기준, 논문 신뢰도 강화의 필수·권장·선택 범위와 예상 시간을 `work/PROJECT_STATUS_AND_RELIABILITY_PLAN_2026-09-13.md`에 추가했습니다.

## 2026-09-13

- 여섯 adapter의 최종 balanced evaluation 완료: 3,984/3,984행, exit code 0.
- Validation과 aggregate analysis 결과, paired bootstrap interval 추가.
- 한국어 IMRaD 연구 원고와 PDF 추가.
- AI 조언과 최종 판단을 분리한 기록 추가.
- Provenance, 환경, 재현 안내 추가.
