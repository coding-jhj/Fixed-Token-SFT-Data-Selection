# T10 모델 선택 업데이트

- 2026-09-12 현재 공개 모델 카드를 다시 확인했습니다.
- 기본 모델은 `Qwen/Qwen3-1.7B-Base`로 변경합니다. 모델 카드상 1.7B, Apache-2.0, pretraining 단계 모델이며 119개 언어를 대상으로 학습되었습니다.
- 현재 GPU는 `NVIDIA GeForce RTX 5060, 8151 MiB`입니다. 따라서 4-bit QLoRA, `max_seq_length` 1024~2048 범위의 smoke test, 고정 token budget이 기본입니다.
- `Qwen/Qwen2.5-1.5B`는 Transformers/QLoRA 호환성 문제가 있을 때의 fallback으로 둡니다.
- 정확한 batch size, sequence length, token budget, training time은 smoke test 뒤 확정합니다.
