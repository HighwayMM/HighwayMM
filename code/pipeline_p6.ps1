$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"; $jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_p6.log"; $jl = "D:\highwaymm\experiments\_pipeline_p6_jobs.txt"
$lab = "D:\highwaymm\HighwayMM_release\annotations\labels_v2t.csv"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
L "grid p6"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_p6.txt *>> $jl
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p6_audit" }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir) --labels $lab" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_p6.txt
L "evaluating $($runs.Count) p6 runs"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_p6.txt *>> $jl
& $py code\train\eval_gold.py --gold D:\highwaymm\HighwayMM_release\gold\gold_labels_merged.csv --out D:\highwaymm\experiments\gold_eval_p6 --match "_p[456]_" *>> $jl
L "P6 DONE"
