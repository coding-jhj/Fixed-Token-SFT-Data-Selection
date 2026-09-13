# 후속 평가 중간 결과

스냅샷: 2026-09-13

이 문서는 기존 논문의 결론을 즉시 대체하지 않는 후속 평가 기록입니다. 장문 생성 한도 민감도 실험은 완료되었고, 세 번째 seed 실험은 평가가 진행 중입니다.

## 1. 장문 생성 한도 민감도

기존에 고정한 평가 subset을 유지하고, 기존 6개 adapter를 다음 한도로 다시 평가했습니다.

- IFEval: `1,024` new tokens
- GSM8K/BBH: `256` new tokens
- batch size: `8`
- decoding: 기존 평가와 동일한 greedy decoding
- 출력: `3,984/3,984`개, 구조 검증 통과

전략별 평균 정확도입니다.

| 전략 | IFEval prompt strict | GSM8K | BBH |
|---|---:|---:|---:|
| random | 12.24% | 37.30% | 25.93% |
| quality | 11.20% | 35.74% | 30.56% |
| diversity | 11.98% | 50.59% | 36.81% |

`diversity - random` paired bootstrap 결과는 다음과 같습니다.

| 지표 | 차이 | 95% bootstrap interval |
|---|---:|---:|
| IFEval prompt strict | -0.26 pp | [-3.13, 2.60] pp |
| IFEval instruction strict | -1.61 pp | [-4.34, 1.13] pp |
| GSM8K | +13.28 pp | [+7.81, +18.55] pp |
| BBH | +10.88 pp | [+5.56, +16.44] pp |

다만 `3,928/3,984`개 출력이 새 generation limit에 도달했을 가능성이 있어, 이 결과는 현재 보강 분석 자료로만 취급합니다. 상세 산출물은 로컬 `work/results_long_generation/`에 있습니다.

재분석 명령:

```powershell
& "C:\Users\ghksw\anaconda3\envs\posttrain-research\python.exe" src\analyze_followup_results.py `
  --evaluation-dir work\evaluation_long_generation `
  --subset-dir work\evaluation_subsets_final `
  --output-dir work\results_long_generation `
  --adapter random_seed13=random=13 `
  --adapter quality_seed13=quality=13 `
  --adapter diversity_seed13=diversity=13 `
  --adapter random_seed42=random=42 `
  --adapter quality_seed42=quality=42 `
  --adapter diversity_seed42=diversity=42 `
  --ifeval-max-new-tokens 1024 `
  --benchmark-max-new-tokens 256 `
  --bootstrap-repetitions 10000
```

## 2. 세 번째 seed 2026

선택과 학습은 완료되었습니다.

| 전략 | 선택 행 수 | 학습 token | optimizer steps | 상태 |
|---|---:|---:|---:|---|
| random | 995 | 1,000,000 | 125 | 학습 완료 |
| diversity | 1,027 | 1,000,000 | 129 | 학습 완료 |

평가 subset은 IFEval 192, GSM8K 256, BBH 216입니다. 두 adapter 합계 1,328개 row 중 현재 1,032개가 생성되어 있습니다.

- random seed 2026: IFEval 192, GSM8K 256, BBH 216 완료
- diversity seed 2026: IFEval 192, GSM8K 176 진행
- diversity seed 2026: BBH 216 미실행

현재 평가가 종료된 뒤 이 문서의 진행 수치를 갱신하고, 두 seed 간 paired analysis를 수행해야 합니다.
