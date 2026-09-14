$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\ghksw\anaconda3\envs\posttrain-research\python.exe"
$env:HF_HUB_OFFLINE = "1"

& $pythonExe -m py_compile src\analyze_final_results.py
& $pythonExe src\analyze_final_results.py `
  --evaluation-dir work\evaluation_final_balanced `
  --subset-dir work\evaluation_subsets_final `
  --output-dir work\results_final `
  --bootstrap-repetitions 10000

if (Test-Path -LiteralPath "work\evaluation_seed2026_long_generation") {
  & $pythonExe src\analyze_followup_results.py `
    --evaluation-dir work\evaluation_seed2026_long_generation `
    --subset-dir work\evaluation_subsets_final `
    --output-dir work\results_seed2026_long_generation `
    --adapter random_seed2026=random=2026 `
    --adapter diversity_seed2026=diversity=2026 `
    --ifeval-max-new-tokens 1024 `
    --benchmark-max-new-tokens 256 `
    --bootstrap-repetitions 10000
  & $pythonExe scripts\analyze_seed_robustness.py `
    --primary-dir work\evaluation_final_balanced `
    --seed2026-dir work\evaluation_seed2026_long_generation `
    --output-dir work\results_seed_robustness `
    --bootstrap-repetitions 10000
}

if (Test-Path -LiteralPath "work\evaluation_expanded_long_generation") {
  & $pythonExe src\analyze_followup_results.py `
    --evaluation-dir work\evaluation_expanded_long_generation `
    --subset-dir work\evaluation_subsets_expanded `
    --output-dir work\results_expanded_long_generation `
    --adapter random_seed2026=random=2026 `
    --adapter diversity_seed2026=diversity=2026 `
    --ifeval-max-new-tokens 1024 `
    --benchmark-max-new-tokens 256 `
    --ifeval-rows 384 `
    --gsm8k-rows 512 `
    --bbh-rows 432 `
    --bootstrap-repetitions 10000
}

if (Test-Path -LiteralPath "work\evaluation_base_long_generation") {
  & $pythonExe src\analyze_followup_results.py `
    --evaluation-dir work\evaluation_base_long_generation `
    --subset-dir work\evaluation_subsets_final `
    --output-dir work\results_base_long_generation `
    --adapter base_model=base=0 `
    --ifeval-max-new-tokens 1024 `
    --benchmark-max-new-tokens 256 `
    --bootstrap-repetitions 10000
}

if ((Test-Path -LiteralPath "work\evaluation_seed2026_long_generation") -and (Test-Path -LiteralPath "work\results_reliability")) {
  $validationArgs = @(
    "scripts\validate_followup_artifacts.py",
    "--seed-output", "work\evaluation_seed2026_long_generation",
    "--base-output", "work\evaluation_base_long_generation",
    "--subset-dir", "work\evaluation_subsets_final",
    "--expanded-subset-dir", "work\evaluation_subsets_expanded",
    "--summary", "work\selection_manifests\manifest_random_seed13.summary.json",
    "--summary", "work\selection_manifests\manifest_random_seed42.summary.json",
    "--summary", "work\selection_manifests\manifest_quality_seed13.summary.json",
    "--summary", "work\selection_manifests\manifest_quality_seed42.summary.json",
    "--summary", "work\selection_manifests\manifest_diversity_seed13.summary.json",
    "--summary", "work\selection_manifests\manifest_diversity_seed42.summary.json",
    "--summary", "work\selection_manifests_seed2026\manifest_random_seed2026.summary.json",
    "--summary", "work\selection_manifests_seed2026\manifest_diversity_seed2026.summary.json",
    "--output", "work\results_reliability\validation_followup.json"
  )
  if (Test-Path -LiteralPath "work\evaluation_expanded_long_generation") {
    $validationArgs += @("--expanded-output", "work\evaluation_expanded_long_generation")
  }
  & $pythonExe @validationArgs
}

& $pythonExe scripts\build_paper_pdf.py `
  --manuscript paper\manuscript.md `
  --figure work\results_final\strategy_metrics.png `
  --output paper\paper.pdf

Write-Output "분석과 PDF 생성이 완료되었습니다."
