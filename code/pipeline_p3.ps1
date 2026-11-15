# Final-config ablations (p3) -> eval -> warning eval; plus national eval + throughput for the final model (p2_abls_k1_image runs)
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_p3.log"
$jl = "D:\highwaymm\experiments\_pipeline_p3_jobs.txt"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
& $py code\train\make_grid_p3.py *>> $jl
$t0 = Get-Date
L "grid_p3 start"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_p3.txt *>> $jl
L "grid_p3 done"
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p3_" -and -not (Test-Path "$($_.run_dir)\test_metrics.json") }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir)" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_p3.txt
L "evaluating $($runs.Count) runs"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_p3.txt *>> $jl
foreach ($r in $runs) { & $py code\warning\eval_warning.py --run $r.run_dir --warn_source tax *> $null }
$final = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p2_abls_k1_image_s\d$" }
foreach ($r in $final) { $env:CUDA_VISIBLE_DEVICES = "0"; & $py code\warning\national_eval.py --run $r.run_dir --splits all *> $null; L "national $($r.run_id) $LASTEXITCODE" }
$env:CUDA_VISIBLE_DEVICES = "0"
& $py code\warning\throughput.py --run ($final | Select-Object -First 1).run_dir *>> $jl
L "throughput final $LASTEXITCODE"
Remove-Item Env:CUDA_VISIBLE_DEVICES
L "P3 DONE"
