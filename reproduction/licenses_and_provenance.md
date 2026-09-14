# 라이선스와 provenance

이 실험은 dataset collection 전체를 동일한 라이선스로 간주하지 않고 source별 provenance를 기록했습니다.

| Source | Revision | 사용 목적 | 재배포 관련 기록 |
|---|---|---|---|
| `HuggingFaceTB/smoltalk` | `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa` | SFT selection용 Apache-2.0 설명 subset 네 개 | Local audit에 source별 주의사항을 기록했으며, 포함된 manifest에 source label과 ID를 보존했습니다. |
| `google/IFEval` | `966cd89545d6b6acfd7638bc708b98261ca58e84` | Contamination check와 frozen evaluation subset | License 조건은 해당 upstream dataset card와 revision metadata를 확인해야 합니다. |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | Frozen evaluation subset | Local audit에 MIT license로 기록했습니다. |
| `lukaemon/bbh` | `982bb89fd79532a8ac676a61fc42eb1aeec63f99` | Frozen evaluation subset | Audit에서 conversion card가 명확한 redistribution license를 노출하지 않아 raw BBH row를 재배포하지 않습니다. |
| `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Base model과 tokenizer | Upstream model card의 조건에 따라 직접 받아야 합니다. |

전체 audit는 workspace의 `work/data_audit.md`에 보존되어 있습니다. 이 패키지는 재생성한 subset 파일이 실험을 조용히 변경하지 않았는지 확인할 수 있도록 hash와 선택 ID를 포함합니다.

선택 데이터의 automatic quality audit summary는 `human_audit/automatic_quality_audit_summary.json`과 `human_audit/automatic_quality_audit_report.md`에 보존합니다. 이는 message 구조·heuristic 재계산·표면 flag 검사이며, 사람의 factual-quality rating이 아닙니다. 200-example blind sheet와 rubric은 실제 사람의 rating을 포함하지 않습니다.

2026-09-14 후속 검증에서는 동일한 고정 원천에서 IFEval 384, GSM8K 512, BBH 432의 expanded subset을 별도로 생성했습니다. 이 subset은 seed 2027로 선택했고, 여덟 개 SFT manifest에 대해 GSM8K·BBH exact 및 normalized near overlap(threshold 0.92)을 다시 검사했습니다. 모든 검사에서 overlap은 0이었습니다. Expanded raw benchmark JSONL은 기존 redistribution 정책과 동일하게 ZIP에 포함하지 않고, subset manifest·hash·재생성 명령만 보존합니다.

Expanded subset 성능 출력은 seed 2026 random/diversity에 대해 별도 protocol로 생성했으며, ZIP에는 raw JSONL 대신 처리된 metric·validation·paired bootstrap 결과와 raw output의 SHA-256 manifest만 포함합니다.
