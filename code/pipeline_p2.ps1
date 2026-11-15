# Protocol v2 (final): grid_p2 (51 runs) -> test eval -> warning eval (all p2 runs) -> national eval (main S) -> throughput -> tables
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_p2.log"
$jl = "D:\highwaymm\experiments\_pipeline_p2_jobs.txt"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
& $py code\train\make_grid_p2.py *>> $jl
$t0 = Get-Date
L "grid_p2 start"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_p2.txt *>> $jl
L "grid_p2 done"
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p2_" -and ([datetime]$_.date) -ge $t0.AddMinutes(-1) }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir)" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_p2.txt
L "evaluating $($runs.Count) runs"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_p2.txt *>> $jl
L "warning eval"
foreach ($r in $runs) { & $py code\warning\eval_warning.py --run $r.run_dir --warn_source tax *> $null }
$main = $runs | Where-Object { $_.run_id -match "_p2_main_taxowarn_s_s\d$" }
foreach ($r in $main) { $env:CUDA_VISIBLE_DEVICES = "0"; & $py code\warning\national_eval.py --run $r.run_dir --splits all *> $null; L "national $($r.run_id) $LASTEXITCODE" }
L "throughput on idle GPU 0"
$env:CUDA_VISIBLE_DEVICES = "0"
$m1 = $main | Select-Object -First 1
$mlp = $runs | Where-Object { $_.run_id -match "_p2_base_mlp_s0$" } | Select-Object -First 1
if ($m1) { & $py code\warning\throughput.py --run $m1.run_dir *>> $jl }
if ($mlp) { & $py code\warning\throughput.py --run $mlp.run_dir *>> $jl }
Remove-Item Env:CUDA_VISIBLE_DEVICES
& $py code\report\make_tables.py *>> $jl
& $py code\report\warning_report.py *>> $jl
L "P2 DONE"
