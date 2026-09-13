$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\ghksw\anaconda3\envs\posttrain-research\python.exe"
$env:HF_HUB_OFFLINE = "1"

& $pythonExe -m py_compile src\analyze_final_results.py
& $pythonExe src\analyze_final_results.py `
  --evaluation-dir work\evaluation_final_balanced `
  --subset-dir work\evaluation_subsets_final `
  --output-dir work\results_final `
  --bootstrap-repetitions 10000

& $pythonExe scripts\build_paper_pdf.py `
  --manuscript paper\manuscript.md `
  --figure work\results_final\strategy_metrics.png `
  --output paper\paper.pdf

Write-Output "분석과 PDF 생성이 완료되었습니다."
