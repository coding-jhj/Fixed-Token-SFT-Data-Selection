# T10 과제 및 post-training 연구 방향

## 첨부 과제 지침 요약
- ALEPH Studio 무료 AI 챌린지 T10은 관심 분야의 질문을 가설로 만들고, 근거·데이터·실험·결과 해석을 거쳐 첫 논문으로 완성하는 과제입니다.
- 1단계는 관심 분야, 후보 주제 3개 이상과 선정 이유, 최종 가설, 가설이 영향을 주는 대상, 측정 방법, 반증 시 결과, 데이터 기간/범위입니다.
- 2단계는 참고문헌 목록, 각 문헌의 저자·연도·핵심 주장·가설 관련 근거, 빠진 문헌과 이유, 실험 데이터가 있는 연구, 인용 목록과 본문의 일치, 출처 확인 방법입니다.
- 3단계는 변수와 고정 조건, 반복 횟수·표본 수, 측정값, 원자료와 출처/날짜, 실행 절차·스크립트, 재현 방법, 설계 변경 기록입니다.
- 4단계는 결과 표/그래프, 가설과 다른 결과, 근거 위치, 해석, 가설 수정 여부와 수정 가설, 기각 시 결과, 최종 결론입니다. 원자료에 없는 주장은 하지 않습니다.
- 5단계는 제목·작성자·작성일, 초록, 서론-방법-결과-논의, 참고문헌, 재현 방법, 원자료와 실행 절차가 든 단일 ZIP, AI 조언과 최종 판단의 분리입니다. 미완성 표시·자리표시자는 남기지 않습니다.
- 제출에는 완성 논문 1편, 재현 패키지 ZIP, AI와 나의 판단 3줄이 필요합니다. 제출 URL·소스 URL·개인정보/비밀번호는 저장하지 않습니다.

## 사용자 목표
- 사용자는 `post-training research engineer` 직무를 목표로 하며, T10 논문을 그 진로와 연결해 제대로 수행하고 싶어 합니다.
- 이후 관련 작업의 기본 방향은 데이터 큐레이션, SFT/DPO 등 post-training, 평가 설계, 재현 가능한 실험 파이프라인입니다.

## 권장 논문 방향
- 제목: `토큰 예산이 고정된 한국어 지시학습에서 품질·다양성 기반 데이터 선택이 소형 언어모델의 일반화에 미치는 영향`
- 핵심 질문: 같은 base model·학습 토큰·훈련 설정에서 무작위 선택, 품질 중심 선택, 품질+다양성 선택 중 어느 전략이 한국어 unseen-task 일반화와 worst-group 성능을 가장 높이는가?
- 권장 base: `Qwen/Qwen2.5-1.5B` base. 모델 카드상 1.54B, pretraining 단계 모델, 한국어를 포함한 29개 이상 언어 지원, post-training 적용을 권장합니다.
- 권장 방법: QLoRA/LoRA 기반 SFT를 주 실험으로 두고, 같은 토큰 예산과 seed를 유지합니다. DPO는 본 실험 완료 후의 선택적 확장입니다.
- 권장 비교군: stratified random, quality-only, quality+diversity. 데이터는 license가 확인된 한국어 instruction pool에서 만들고, task/source 중복과 평가 세트 contamination을 점검합니다.
- 핵심 평가: held-out 한국어 task macro score, worst-group score, K2-Eval 또는 동등한 한국어 평가, 형식 준수율, human rubric 평가, 영어/일반능력 회귀 여부.
- 핵심 가설 예시: 고정 토큰 예산에서 품질+다양성 선택은 품질-only와 random보다 평균 unseen-task 성능뿐 아니라 최저 task 성능을 높입니다. bootstrap/seed 결과에서 차이가 재현되지 않거나 특정 task만 개선되면 가설을 기각 또는 수정합니다.

## 실행 원칙
- 데이터·모델·라이선스·하드웨어는 실행 전에 실제 상태를 확인합니다.
- 3개 전략을 2 seed로 먼저 실행하고, 최종 비교군에는 3번째 seed를 추가합니다. 정확한 token budget과 GPU/VRAM은 smoke test 뒤 고정합니다.
- 제출 ZIP에는 raw/processed data manifest, selection code, train/eval config, seed, environment lock, result CSV/plot, README, 설계 변경 로그를 포함합니다.
