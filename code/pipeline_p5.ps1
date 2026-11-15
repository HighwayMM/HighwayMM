# P5: wait for detection features -> train det-fusion grid (12 runs) -> evaluate on labels_v2t -> gold eval.
# Log: experiments\_pipeline_p5.log
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_p5.log"
$jl = "D:\highwaymm\experiments\_pipeline_p5_jobs.txt"
$lab = "D:\highwaymm\HighwayMM_release\annotations\labels_v2t.csv"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
L "waiting for detection shards"
while ((Get-ChildItem D:\highwaymm\HighwayMM_release\features\det -Filter "shard*.npz" -ErrorAction SilentlyContinue).Count -lt 4) { Start-Sleep 30 }
Start-Sleep 20
L "grid p5"
& $py code\train\make_grid_p5.py *>> $jl
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_p5.txt *>> $jl
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p5_det_" }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir) --labels $lab" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_p5.txt
L "evaluating $($runs.Count) p5 runs on v2t"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_p5.txt *>> $jl
L "gold eval"
& $py code\train\eval_gold.py --gold D:\highwaymm\HighwayMM_release\gold\gold_labels_merged.csv --out D:\highwaymm\experiments\gold_eval_p5 --match "_p[45]_" *>> $jl
L "P5 DONE"
