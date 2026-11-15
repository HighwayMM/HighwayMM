# P4: add Iowa DOT RWIS summer cameras (sensor labels) -> frames/OCR mask -> features -> VLM (missing only) -> labels_v2
#     -> grid_p4 (63 runs) -> test eval -> warning eval -> tables.  Log: experiments\_pipeline_p4.log
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_p4.log"
$jl = "D:\highwaymm\experiments\_pipeline_p4_jobs.txt"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
L "waiting for RWIS sensor download"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'iowa_rwis_sensors.py' }) { Start-Sleep 20 }
L "iowa select"
& $py code\data\iowa_select.py *>> $jl
L "iowa fetch images"
& $py -I code\data\iowa_rwis_fetch.py fetch --selection D:\highwaymm\external\iowa_rwis\selection.csv *>> $jl
L "build_external (+iowa)"
& $py code\data\build_external.py *>> $jl
L "prep frames (only missing items are processed)"
& $py $jr --gpus 0,1,2 --per-gpu 2 --cmd "code\data\prep_frames.py --manifest manifest_external.csv --shard {s} --nshards 6" --grid s=0,1,2,3,4,5 *>> $jl
L "features (iowa only)"
& $py $jr --gpus 0,1,2 --cmd "code\data\extract_features.py --only iowa_ --name iowa --shard {s} --nshards 3" --grid s=0,1,2 *>> $jl
& $py $jr --gpus 0,1,2 --cmd "code\data\extract_spatial.py --only iowa_ --name iowa --shard {s} --nshards 3" --grid s=0,1,2 *>> $jl
L "vlm items + VLM (missing only)"
& $py code\data\make_vlm_items.py *>> $jl
& $py $jr --gpus 0,1,2 --cmd "code\data\vlm_label.py --items D:\highwaymm\HighwayMM_release\annotations\vlm_items.csv --out_dir D:\highwaymm\HighwayMM_release\annotations\vlm_qwen25vl7b --shard {s} --nshards 3 --bs 8 --max_side 448" --grid s=0,1,2 *>> $jl
L "labels v2"
& $py code\data\build_labels.py v2 *>> $jl
L "baselines on v2 (zero-shot, VLM)"
& $py code\train\evaluate.py --zeroshot siglip2 --labels D:\highwaymm\HighwayMM_release\annotations\labels_v2.csv *>> $jl
& $py code\train\evaluate.py --vlm --labels D:\highwaymm\HighwayMM_release\annotations\labels_v2.csv *>> $jl
L "grid_p4"
& $py code\train\make_grid_p4.py *>> $jl
$t0 = Get-Date
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_p4.txt *>> $jl
L "grid_p4 done"
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p4_" -and -not (Test-Path "$($_.run_dir)\test_metrics.json") }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir)" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_p4.txt
L "evaluating $($runs.Count) runs"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_p4.txt *>> $jl
foreach ($r in $runs) { & $py code\warning\eval_warning.py --run $r.run_dir --warn_source tax *> $null }
L "P4 DONE"
