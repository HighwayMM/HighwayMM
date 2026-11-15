# Stage 4: TaxoWarn-S grid (33 runs) -> test eval -> warning eval (3 sources x main runs) -> national eval -> throughput (idle GPU) -> tables
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_stage4.log"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
# (stage 3 finished at 06:57; no wait needed)
$t0 = Get-Date
L "grid_s start"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_s_v1.txt *>> D:\highwaymm\experiments\_pipeline_stage4_jobs.txt
L "grid_s done"
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.task -eq "taxowarn_s" -and $_.status -eq "completed" -and ([datetime]$_.date) -ge $t0.AddMinutes(-1) }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir)" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_s_v1.txt
L "evaluating $($runs.Count) runs"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_s_v1.txt *>> D:\highwaymm\experiments\_pipeline_stage4_jobs.txt
$main = $runs | Where-Object { $_.run_id -match "_main_taxowarn_s_s\d$" }
foreach ($r in $main) {
  & $py code\warning\eval_warning.py --run $r.run_dir --warn_source tax *> $null; L "warning eval $($r.run_id) exit $LASTEXITCODE"
  $env:CUDA_VISIBLE_DEVICES = "0"; & $py code\warning\national_eval.py --run $r.run_dir --splits all *> $null; L "national eval $($r.run_id) exit $LASTEXITCODE"
}
L "throughput (idle GPU 0)"
$env:CUDA_VISIBLE_DEVICES = "0"
if ($main) { & $py code\warning\throughput.py --run ($main | Select-Object -First 1).run_dir *>> D:\highwaymm\experiments\_pipeline_stage4_jobs.txt }
Remove-Item Env:CUDA_VISIBLE_DEVICES
& $py code\report\make_tables.py *>> D:\highwaymm\experiments\_pipeline_stage4_jobs.txt
L "STAGE4 DONE"

