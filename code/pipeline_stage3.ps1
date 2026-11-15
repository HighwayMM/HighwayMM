# Stage 3: after VLM labels -> labels v1 -> experiment grid (51 runs) -> test eval of every run -> zero-shot + VLM baselines
# -> ConvNeXt fine-tune baseline (3 seeds) -> warning-system eval (main runs) -> throughput -> tables/figures.
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_stage3.log"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
L "waiting for stage 2"
while (-not (Select-String -Path D:\highwaymm\experiments\_pipeline_stage2.log -Pattern "STAGE2 DONE" -Quiet)) { Start-Sleep 30 }
L "labels v1 (with VLM)"
& $py code\data\build_labels.py | Add-Content -Encoding utf8 $log
& $py code\report\dataset_figs.py | Out-Null
& $py code\train\make_grid.py | Add-Content -Encoding utf8 $log
$t0 = Get-Date
L "grid start"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\grid_v1.txt | Add-Content -Encoding utf8 $log
L "grid done exit $LASTEXITCODE"
# evaluate every taxowarn run created by this grid
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.task -eq "taxowarn" -and $_.status -eq "completed" -and ([datetime]$_.date) -ge $t0.AddMinutes(-1) }
$ev = $runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir)" }
$ev | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_v1.txt
L "evaluating $($ev.Count) runs"
& $py $jr --gpus 0,1,2 --per-gpu 3 --jobs D:\highwaymm\experiments\grids\eval_v1.txt | Add-Content -Encoding utf8 $log
L "baselines: zero-shot + VLM"
$env:CUDA_VISIBLE_DEVICES = "0"
& $py code\train\evaluate.py --zeroshot siglip2 | Select-Object -Last 3 | Add-Content -Encoding utf8 $log
& $py code\train\evaluate.py --vlm | Select-Object -Last 3 | Add-Content -Encoding utf8 $log
Remove-Item Env:CUDA_VISIBLE_DEVICES
L "ConvNeXt fine-tune baseline"
$ft = 0, 1, 2 | ForEach-Object { "code\train\main.py --config code\train\configs\finetune_convnext.yaml --tag base_ft_convnext --set seed=$_" }
$ft | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\ft_v1.txt
$t1 = Get-Date
& $py $jr --gpus 0,1,2 --jobs D:\highwaymm\experiments\grids\ft_v1.txt | Add-Content -Encoding utf8 $log
$fr = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.task -eq "finetune_img" -and $_.status -eq "completed" -and ([datetime]$_.date) -ge $t1.AddMinutes(-1) }
$fe = $fr | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir)" }
$fe | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\ft_eval_v1.txt
& $py $jr --gpus 0,1,2 --jobs D:\highwaymm\experiments\grids\ft_eval_v1.txt | Add-Content -Encoding utf8 $log
L "warning-system eval on main runs"
$main = $runs | Where-Object { $_.run_id -match "_main_taxowarn_s\d$" }
foreach ($r in $main) { & $py code\warning\eval_warning.py --run $r.run_dir | Out-Null; L "warning eval $($r.run_id) exit $LASTEXITCODE" }
L "throughput"
$env:CUDA_VISIBLE_DEVICES = "1"
if ($main) { & $py code\warning\throughput.py --run ($main | Select-Object -First 1).run_dir | Select-Object -Last 2 | Add-Content -Encoding utf8 $log }
Remove-Item Env:CUDA_VISIBLE_DEVICES
L "tables"
& $py code\report\make_tables.py | Add-Content -Encoding utf8 $log
L "STAGE3 DONE"
