# Stage 2 driver: wait for our frame prep -> external frame prep -> VLM labels -> SigLIP2 features. Logs: experiments\_pipeline_stage2.log
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_stage2.log"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
L "waiting for prep_frames (ours)"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'prep_frames.py' }) { Start-Sleep 30 }
L "ours prep finished; external prep"
& $py $jr --gpus 0,1,2 --per-gpu 2 --cmd "code\data\prep_frames.py --manifest manifest_external.csv --shard {s} --nshards 6" --grid s=0,1,2,3,4,5 | Add-Content -Encoding utf8 $log
L "external prep exit $LASTEXITCODE; build vlm items"
& $py code\data\make_vlm_items.py | Add-Content -Encoding utf8 $log
L "features"
& $py $jr --gpus 0,1,2 --cmd "code\data\extract_features.py --shard {s} --nshards 3" --grid s=0,1,2 | Add-Content -Encoding utf8 $log
L "features exit $LASTEXITCODE; preliminary labels (no VLM yet)"
& $py code\data\build_labels.py | Add-Content -Encoding utf8 $log
L "FEATURES READY"
L "VLM labelling"
& $py $jr --gpus 0,1,2 --cmd "code\data\vlm_label.py --items D:\highwaymm\HighwayMM_release\annotations\vlm_items.csv --out_dir D:\highwaymm\HighwayMM_release\annotations\vlm_qwen25vl7b --shard {s} --nshards 3 --bs 8 --max_side 448" --grid s=0,1,2 | Add-Content -Encoding utf8 $log
L "VLM exit $LASTEXITCODE; labels"
& $py code\data\build_labels.py | Add-Content -Encoding utf8 $log
L "STAGE2 DONE"
