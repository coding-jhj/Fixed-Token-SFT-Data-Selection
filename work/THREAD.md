# T10 Post-Training Research Paper

## Objective
- 첨부된 ALEPH Studio T10 과제 지침을 사용자 요청과 분리해 구조화한다.
- `post-training research engineer` 진로에 맞는 논문 주제 1개와 실행 가능한 연구 계획을 설계한다.
- 현재 연구 흐름은 공식 논문·공식 문서로 검증한다.
- 사용자가 이후 작업에서 재사용할 수 있도록 핵심 맥락을 메모리에 남긴다.

## Constraints from attached screenshots
- T10은 관심 분야에서 가설을 세우고 직접 데이터를 모아 검증하는 첫 논문 과제이다.
- 제출물: 완성 논문 1편, 재현 패키지 ZIP, AI와 나의 판단 3줄.
- 논문 과정: 질문을 가설로 만들기 -> 논거 세우기 -> 실험 설계와 실행 -> 결과 해석 -> 논문으로 묶기.
- 논문은 가설-근거-데이터-결론이 이어져야 하며, 근거 없는 주장은 하지 않는다.
- 1단계: 관심 분야, 후보 주제 3개 이상, 각 주제 선정 이유, 최종 가설, 가설 영향, 표현 측정 방법, 반증 시 결과, 데이터 기간/범위.
- 2단계: 참고문헌 목록, 문헌별 저자/발표 연도, 핵심 주장/근거, 가설 뒷받침 부분, 빠진 문헌과 이유, 실험 데이터로 검증된 연구, 인용 문헌 목록과 일치, 출처 확인 방법.
- 3단계: 변수와 고정 조건, 반복 횟수와 표본 수, 측정값, 원자료, 출처/받은 날짜, 실행 절차/스크립트, 재현 방법, 설계 변경 기록.
- 4단계: 결과 표/그래프, 가설과 다른 결과 및 범위, 근거 위치, 가설과 다른 결과의 해석, 가설 수정 여부, 고친 가설, 기각 시 결과, 최종 결론, 원자료에 없는 주장 금지.
- 5단계: 제목/작성자/작성일, 초록, 서론-방법-결과-논의, 참고문헌, 재현 방법, 원자료와 실행 절차 포함 ZIP, AI 조언과 최종 판단 분리, 미완성 표시/자리표시자 금지.
- 제출 ZIP은 논문과 원자료·재현 방법을 포함하는 한 개의 패키지여야 한다.

## User request
- 첨부 내용을 이후 작업의 전제로 학습한다.
- 논문을 `post-training research engineer`가 되고 싶은 진로와 연결한다.
- 주제 기획과 실행 계획을 제대로 설계한다.

## Current phase
1. 첨부 지침과 사용자 요청 분리 완료.
2. 영어 논문 주제와 실험 프로토콜 확정.
3. Apache-2.0 SmolTalk 네 subset과 token-budget 설계 확정.
4. 연구 전용 `posttrain-research` 환경 생성 및 smoke-test 의존성 설치 완료.
5. Hub streaming/chat-template 확인과 2,048-token QLoRA VRAM smoke test 완료.
6. selector, contamination check, fixed-token manifest 생성 완료.
7. base-model baseline 생성과 짧은 SFT training smoke 완료.
8. GSM8K·BBH 평가 세트 고정과 benchmark contamination check 완료.
9. `random/seed13`, `quality/seed13`, `diversity/seed13` 본 학습 및 adapter generation 확인 완료.
10. `random/seed42`, `quality/seed42`, `diversity/seed42` 본 학습 및 adapter generation 확인 완료.
11. `src/evaluate_frozen.py` 구현 및 6개 adapter x 3개 benchmark x 32건 pilot 평가 완료. 최종 frozen 전체 평가 전 파이프라인·메모리·출력 형식을 검증했다.
12. 전체 IFEval 실행은 batch size 4와 `max_new_tokens=256`으로 시작했으나, 비용이 과도하고 긴 응답 truncation 가능성이 있어 graceful interrupt로 중단했다. `random_seed13` 183/541건의 부분 JSONL은 `work/evaluation_full_ifeval/`에 보존되어 있으나 최종 점수로 사용하지 않는다.
13. 다음 평가 설계는 batch size 8 메모리 smoke test, IFEval 192건 stratified subset, GSM8K 256건 고정 subset, BBH 27개 task x 8건 subset, 그리고 benchmark별 generation length를 포함한다. 전체 benchmark를 그대로 실행하지 않는다.
14. `src/make_eval_subset.py`로 최종 subset과 SHA-256 manifest를 생성했고, batch size 8 smoke test가 IFEval 512 tokens 및 GSM8K·BBH 128 tokens에서 통과했다. 이제 6개 adapter 전체 비교를 시작한다.
15. `work/evaluation_final_balanced/`에서 6개 adapter 최종 balanced evaluation을 실행 중이다. `random_seed13`의 IFEval 192건, GSM8K 256건, BBH 216건은 완료했고, `quality_seed13`을 시작했다. batch size 8 OOM은 없다.
16. `src/evaluate_frozen.py`에 `--resume`와 interrupted JSONL 복구 처리를 추가하고 컴파일을 통과했다. 인수인계 문서는 workspace와 Downloads에 저장했다.
17. 최종 balanced evaluation이 6개 adapter x 3개 benchmark 전체 3,984/3,984행으로 종료되었고 exit code 0을 확인했다. 각 출력 파일의 행 수·ID·JSON 구조·빈 응답을 검증했다.
18. `src/analyze_final_results.py`로 validation, per-seed metrics, strategy metrics, BBH task metrics, paired bootstrap 결과를 생성했다. primary IFEval strict에서 diversity-minus-random은 -0.26pp, 95% CI [-3.39, 2.60]으로 H1을 지지하지 않았다. BBH secondary는 +9.49pp, 95% CI [4.63, 14.58]이었다.
19. 응답을 재인코딩한 보수적 generation-cap 진단은 3,937/3,984행을 potential truncation으로 표시했다. 원래 generated token IDs가 보존되지 않아 확정적 truncation 판정이 아님을 원고에 명시했다.
20. 영어 IMRaD manuscript, PDF, AI 조언과 최종 판단 분리 기록, 재현 안내·provenance·환경 기록, 재현 ZIP을 생성하고 PDF 6페이지를 렌더링해 시각 검수했다.
21. 사용자 요청에 따라 서술 문서를 한국어로 전환했다. 실험 코드, 명령어, dataset/model ID, metric 이름, 수치와 참고문헌 원제목은 재현성과 출처 보존을 위해 유지했다. 한글 지원 폰트를 적용한 PDF는 5페이지로 재생성했다.
22. 사용자 제보의 페이지별 폭 차이를 `pdfinfo -f 1 -l N`으로 재검증했고, `bbox_inches="tight"`가 내용물 기준으로 각 페이지를 잘라낸 것이 원인임을 확인했다. `savefig`에서 tight cropping을 제거하고 A4 canvas를 보존했다.
23. 원고 내용 감사를 수행한 결과 기존 문서는 핵심 결과 요약본에 가까웠다. 후보 풀·quality score·diversity clustering·학습 실행·평가 subset·무결성 검사·truncation 진단·계획 대비 실제 범위·한계를 한국어 원고에 확장 반영했다.
24. 실제 Malgun 글꼴 폭 기반 줄바꿈, 본문 첫 줄 들여쓰기, 제목 단계, 자동 표 셀 줄바꿈, 표 제목과 표의 동시 페이지 배치를 PDF renderer에 적용했다.
25. 최신 원고를 재생성한 PDF는 15페이지이며 모든 페이지가 `595.44 x 841.68 pt (A4)`로 동일하다. 15개 PNG를 전체 시각 검수했고, 오른쪽 잘림·표 제목 고립·본문 위계 문제를 확인하지 않았다.
26. 최신 원고와 PDF를 reproduction ZIP에 반영했다. ZIP은 `testzip=None`, 49 entries, raw `.jsonl` 0, manuscript/PDF/저자명 포함으로 재검증했다.

## Decisions pending
- 서술 문서와 저자는 한국어·정환주로 고정한다. 실험 코드와 model/dataset ID, metric 명칭, 수치는 재현성을 위해 원문 표기를 유지한다.
- 주 실험은 SFT 데이터 선택 전략 비교로 완료했다. DPO와 추가 selector는 현재 범위에 포함하지 않는다.
- 다음 선택지는 제출용 DOCX/HWPX 변환, 문장 전면 윤문, generation cap 확대·추가 seed·human evaluation 중 하나다.

## Research evidence collected
- InstructGPT: post-training feedback can improve instruction following and reduce harmful behavior.
- DPO: preference tuning can be implemented with a lightweight classification-style objective.
- Tulu 3: data curation, staged post-training, decontamination, and unseen evaluation are central to an open recipe.
- QDIT/Deita: quality, complexity, and diversity are useful dimensions for instruction-data selection; diversity can improve worst-case robustness.
- Primary English pool: the four new Apache-2.0 `HuggingFaceTB/smoltalk` subsets (`smol-magpie-ultra`, `smol-constraints`, `smol-rewrite`, `smol-summarize`); public-source subsets remain excluded pending audit.
- Qwen3-1.7B-Base model card: 1.7B base model, Apache-2.0, 119-language pretraining; Qwen2.5-1.5B remains a compatibility fallback.

## Touched files
- work/THREAD.md
- work/research_protocol.md
- work/data_audit.md
- work/run_data_template_smoke.py
- work/run_qlora_smoke.py
- work/smoke_results.json
- src/select_data.py
- src/contamination_check.py
- src/train_sft.py
- src/generate.py
- src/freeze_benchmarks.py
- src/benchmark_contamination.py
- configs/selection_config.json
- work/selection_manifests/
- work/evaluation_sets/
- work/training_smoke_random_seed13/
- work/main_random_seed13/
- work/main_quality_seed13/
- work/main_diversity_seed13/
- work/main_random_seed42/
- work/main_quality_seed42/
- work/main_diversity_seed42/
- src/evaluate_frozen.py
- src/make_eval_subset.py
- work/evaluation_pilot_32/
- work/evaluation_full_ifeval/
- work/evaluation_subsets_final/
- work/evaluation_smoke_batch8_final/
- work/HANDOFF_2026-09-12.md
- src/analyze_final_results.py
- work/evaluation_final_balanced/
- work/results_final/
- paper/manuscript.md
- paper/paper.pdf
- paper/ai_advice_and_judgement.md
- scripts/build_paper_pdf.py
- scripts/reproduce_analysis.ps1
- scripts/build_reproduction_zip.py
- reproduction/
- T10_post_training_reproduction_package_2026-09-13.zip
- reproduction/README.md
- reproduction/licenses_and_provenance.md
- reproduction/CHANGELOG.md
- reproduction/environment.lock

## Verification
- 초기 workspace 파일 목록을 확인했고, 이후 산출물은 이 THREAD에 기록된 경로에 생성했다.
- `academic-researcher`, `fine-tuning-expert`, `llm-evaluation`, `documents` 스킬 지침을 확인했고, 원고 작성·실험 보고·PDF 렌더 검수에 적용했다.
- `C:\Users\ghksw\anaconda3\envs\posttrain-research\python.exe -m py_compile src\analyze_final_results.py scripts\build_paper_pdf.py scripts\build_reproduction_zip.py` 통과.
- `.\scripts\reproduce_analysis.ps1` 재실행 결과 `validation_all_files_complete: true`, `possible_truncation_total: 3937`.
- `pdfinfo -f 1 -l 30 paper\paper.pdf`: 최종 15 pages, non-empty PDF. 15페이지 모두 `595.44 x 841.68 pt (A4)`로 동일하며, 저자명은 정환주다.
- ZIP integrity test `zipfile.testzip()`: `None`; 49 entries; raw `.jsonl` entries 0; manuscript, PDF, analysis, processed results 포함 확인.
- 가독성 수정 후 `.\scripts\reproduce_analysis.ps1`를 재실행해 `validation_all_files_complete: true`, `possible_truncation_total: 3937`을 재확인했고 PDF와 reproduction ZIP을 갱신했다.
- 최종 ZIP 재검증: `testzip=None`, 49 entries, raw `.jsonl` entries 0, manuscript와 PDF가 모두 포함되고 manuscript 저자명 정환주가 확인된다.
- 여백·흐름 수정: 가로 전용 그림 페이지를 본문 내 figure block으로 바꾸고, 실제 글꼴 폭 줄바꿈·첫 줄 들여쓰기·표 제목 동시 배치·A4 canvas 보존을 적용했다. 최신 PDF 15페이지를 PNG로 전체 시각 검수했다.
- ZIP의 manuscript와 README/provenance/changelog가 한국어 서술을 포함하는지 확인했다.

## Next action
- 기계 실행, 한국어 논문 확장, 가독성 조판, A4 페이지 고정, ZIP 갱신과 검증을 완료했다. 사용자는 최신 원고/PDF를 검토하고 제출용 변환 또는 후속 평가를 선택한다.

## Original recommended plan (completed)
- Candidate A (recommended): fixed-token English SFT data selection: random vs quality-only vs quality+diversity.
- Candidate B: off-policy vs on-policy preference pairs for English DPO; higher career relevance but requires reliable preference pairs and more compute.
- Candidate C: verifier-based RL on English math/format tasks; frontier relevance but too risky for the first paper on 8GB VRAM.
- Main hypothesis: under a fixed token budget, quality+diversity selection improves held-out English instruction-following macro score and worst-group performance versus random and quality-only selection.
- Main model: `Qwen/Qwen3-1.7B-Base`; training: 4-bit QLoRA SFT; experimental conditions: 3 strategies x 2 seeds, then add a third seed for the strongest comparison; DPO only as stretch.
- Evaluation: frozen IFEval, GSM8K test, and BBH test; held-out task macro score, worst-group score, format compliance, blind human rubric, and regression check.
- Schedule: week 1 literature/preregistration/license audit; week 2 data pool/splits/contamination; week 3 selectors and smoke test; weeks 4-5 training; week 6 evaluation; week 7 analysis and failure cases; week 8 paper and reproduction ZIP.
- Next user-facing action: 한국어 원고를 검토한 뒤 제출본 형식 변환 또는 후속 평가를 선택한다.

## Verification result
- Attached screenshots: 10 images reviewed from the prompt; they describe T10 assignment instructions, not direct instructions to the assistant.
- Research references checked via official/primary sources: InstructGPT, DPO, Tülu 3, QDIT, Deita, KIT-19, Qwen model cards, PEFT LoRA docs, K2-Eval dataset card.
- Current GPU: `NVIDIA GeForce RTX 5060, 8151 MiB`.
- Base Anaconda `torch` import hit a duplicate OpenMP DLL conflict, so it is not used for experiments. The dedicated `posttrain-research` conda environment passes the integrated import test: Python 3.12.13; `torch 2.11.0+cu128`; CUDA available; `NVIDIA GeForce RTX 5060`; `transformers 5.17.0`; `datasets 5.0.1`; `bitsandbytes 0.49.2`; `sentence-transformers 6.0.1`; `trl 1.13.0`; `peft 0.20.0`; `lm_eval 0.4.13`.
- The original `qwen3-tts` environment was restored to `transformers 4.57.3`, `datasets 3.6.0`, `huggingface_hub 0.36.2`, `tokenizers 0.22.2`, and `safetensors 0.7.0`; research packages were removed from that environment.
- A workspace `.venv` was created with system site packages and contains the research packages, but it inherits the base Anaconda OpenMP conflict; use `posttrain-research` for all experiment commands.
- Data/template smoke test: 100 streamed examples per config; `smol-magpie-ultra` mean 1,489.79, min 533, max 4,288, 7 over 2,048; constraints mean 211.29; rewrite mean 339.63; summarize mean 415.46. Results match the earlier independent sample.
- QLoRA smoke test: `Qwen/Qwen3-1.7B-Base`, 4-bit NF4, LoRA r=16/alpha=32/dropout=.05, sequence length 2,048, one forward/backward step; loss 6.6917; peak allocated 6,584.9 MiB; peak reserved 7,516.0 MiB; elapsed 67.7 s.
- SmolTalk Hub configs verified: `all`, four Apache-only new subsets, and public-source subsets. Streaming token sample measured; `smol-magpie-ultra` mean 1489.8 and `smol-summarize` mean 415.5 formatted Qwen3 tokens over 100 examples each. Protocol context target changed from 1024 to 2048 and passed the VRAM smoke test.
- Full selection run: one shared `pool_seed=0` candidate pool accepted 555,479 rows across the four Apache-only subsets. Six manifests (`random`, `quality`, `diversity` x seeds `13`, `42`) each contain exactly 1,000,000 formatted tokens; all selected example IDs are unique and all selected rows are at most 2,048 tokens.
- IFEval contamination: the cached official `google/IFEval` train split contains 541 prompts; all six manifests have exact overlap 0 and near overlap 0 at normalized `SequenceMatcher` threshold 0.92.
- Evaluation freeze: GSM8K `main/test` has 1,319 rows at revision `740312add88f781978c0658806c59bc2815b9866`; BBH has 6,511 rows across 27 tasks at revision `982bb89fd79532a8ac676a61fc42eb1aeec63f99`. All six manifests have exact and near overlap 0 against both sets at threshold 0.92.
- Training smoke: fixed Qwen3 commit with 8 rows, batch size 1, gradient accumulation 2, and 4 optimizer steps completed; 1,074 tokens seen, loss curve saved, peak allocated 3,041.7 MiB, peak reserved 3,742.0 MiB, and LoRA adapter reloaded successfully.
- Generation smoke: the pinned base model and saved adapter both generated 64 tokens from the same prompt. The smoke adapter output is not treated as a quality result because the run is intentionally only four optimizer steps.
- Main training: `random/seed13` completed 126 optimizer steps over 1,006 rows with exactly 1,000,000 tokens in 1,674.7 seconds; `quality/seed13` completed 130 optimizer steps over 1,033 rows with exactly 1,000,000 tokens in 1,684.2 seconds. Both adapters reloaded and generated 64 tokens from the fixed prompt.
- Main training: `diversity/seed13` completed 129 optimizer steps over 1,029 rows with exactly 1,000,000 tokens in 1,540.9 seconds; loss changed from 2.5650 to 0.9765, peak allocated memory was 6,750.1 MiB, and peak reserved memory was 17,442.0 MiB. The adapter reloaded and generated 64 tokens from the fixed prompt.
- Main training: `random/seed42` completed 125 optimizer steps over 1,000 rows with exactly 1,000,000 tokens in 1,797.7 seconds; loss changed from 2.2237 to 2.4013, peak allocated memory was 6,781.8 MiB, and peak reserved memory was 15,302.0 MiB. The adapter reloaded and generated 64 tokens from the fixed prompt.
- Main training: `quality/seed42` completed 130 optimizer steps over 1,033 rows with exactly 1,000,000 tokens in 1,654.9 seconds; loss changed from 1.8829 to 1.1492, peak allocated memory was 6,747.5 MiB, and peak reserved memory was 16,248.0 MiB. The adapter reloaded and generated 64 tokens from the fixed prompt.
- Main training: `diversity/seed42` completed 128 optimizer steps over 1,021 rows with exactly 1,000,000 tokens in 1,612.1 seconds; loss changed from 1.1651 to 0.9089, peak allocated memory was 6,748.0 MiB, and peak reserved memory was 16,066.0 MiB. The adapter reloaded and generated 64 tokens from the fixed prompt.
- Frozen evaluation pilot: all six adapters completed IFEval, GSM8K, and BBH with 32 rows per benchmark and batch size 4. The aggregate is saved at `work/evaluation_pilot_32/evaluation_summary.json`; this is a pipeline smoke/pilot result, not the final statistical comparison.
- Verification: selector, contamination checker, and smoke scripts compile; manifests and JSON reports parse; recomputed token sums and uniqueness checks pass; `git diff --check` passes.
- Handoff: `work/HANDOFF_2026-09-12.md` and `C:\Users\ghksw\Downloads\T10_POST_TRAINING_HANDOFF_2026-09-12.md`; evaluator resume support compiled successfully.
- Durable notes written: `C:\Users\ghksw\.codex\memories\extensions\ad_hoc\notes\20260912-t10-post-training-paper.md` and `20260912-t10-model-update.md`.
- Cover revision: shortened the main title to `고정 토큰 예산에서의 데이터 선택 효과`, separated the scope as a subtitle, and kept author metadata as `정환주`. The cover now uses a one-line title, aligned abstract indentation, and hanging-indent key results.
- Pagination revision: reserved space for a following table when a subsection heading precedes it, so `4.4 Paired contrast` now starts with 표 8 on page 11 instead of becoming an orphan on page 10.
- Final visual QA: rendered the current 15-page PDF to PNG and checked the cover, page transitions, tables, figure page, references, and page 10/11 boundary. `pdfinfo` confirms all 15 pages are A4 (`595.44 x 841.68 pts`).
- Package refresh: rebuilt `T10_post_training_reproduction_package_2026-09-13.zip`; verification reports 49 entries, 0 raw JSONL entries, author/title present, embedded PDF size 346,877 bytes, and `testzip=None`. `python -m py_compile scripts\\build_paper_pdf.py` and `git diff --check` pass.
- GitHub publication: added the provided remote as `origin`, created root `README.md` and `.gitignore`, committed 54 curated files as `8dfbfa6`, and pushed successfully to `origin/master`. Local status is clean and tracks the remote branch.
- Publication scope: excluded `.venv`, adapter checkpoints, tokenizers, raw JSONL, and visual QA renders from GitHub because they total most of the 800 MB workspace and have redistribution/size concerns. The 763 KB reproduction ZIP remains included.
- Editorial diagnosis: current manuscript is a complete experiment/reproduction report but still feels visually and substantively light as a paper because the results are concentrated in a few aggregate tables, there is only one figure, methods lack algorithm/equation detail, and there is no per-seed/failure-case appendix or fuller related-work synthesis. A second, content-expansion commit is recommended; no unsupported results should be invented.
