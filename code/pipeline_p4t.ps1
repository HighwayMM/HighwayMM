# P4t: add TSBOW (Suwon CCTV, human per-video weather tags) as a TEST-ONLY Korean CCTV source -> labels_v2t
# (train/val rows identical to labels_v2) -> re-evaluate all p4 runs + zero-shot + VLM. Log: experiments\_pipeline_p4t.log
$ErrorActionPreference = "Continue"
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$jr = "D:\highwaymm\code\train\jobrunner.py"
$log = "D:\highwaymm\experiments\_pipeline_p4t.log"
$jl = "D:\highwaymm\experiments\_pipeline_p4t_jobs.txt"
$lab = "D:\highwaymm\HighwayMM_release\annotations\labels_v2t.csv"
function L($m) { "$(Get-Date -Format s) $m" | Add-Content -Encoding utf8 $log }
Set-Location D:\highwaymm
L "waiting for tsbow_frames"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'tsbow_frames.py' }) { Start-Sleep 20 }
L "build_external (+tsbow)"
& $py code\data\build_external.py *>> $jl
& $py $jr --gpus 0,1,2 --per-gpu 2 --cmd "code\data\prep_frames.py --manifest manifest_external.csv --shard {s} --nshards 6" --grid s=0,1,2,3,4,5 *>> $jl
L "features (tsbow only)"
& $py $jr --gpus 0,1,2 --cmd "code\data\extract_features.py --only tsbow_ --name tsbow --shard {s} --nshards 3" --grid s=0,1,2 *>> $jl
& $py $jr --gpus 0,1,2 --cmd "code\data\extract_spatial.py --only tsbow_ --name tsbow --shard {s} --nshards 3" --grid s=0,1,2 *>> $jl
L "VLM (missing only)"
& $py code\data\make_vlm_items.py *>> $jl
& $py $jr --gpus 0,1,2 --cmd "code\data\vlm_label.py --items D:\highwaymm\HighwayMM_release\annotations\vlm_items.csv --out_dir D:\highwaymm\HighwayMM_release\annotations\vlm_qwen25vl7b --shard {s} --nshards 3 --bs 8 --max_side 448" --grid s=0,1,2 *>> $jl
L "labels v2t"
& $py code\data\build_labels.py v2t *>> $jl
& $py -c "import pandas as pd; a=pd.read_csv(r'D:\highwaymm\HighwayMM_release\annotations\labels_v2.csv',keep_default_na=False); b=pd.read_csv(r'$lab',keep_default_na=False); b=b[b.source!='tsbow']; c=[x for x in a.columns]; print('v2 vs v2t (non-tsbow) identical:', a[c].astype(str).reset_index(drop=True).equals(b[c].astype(str).reset_index(drop=True)))" *>> $jl
L "baselines v2t"
& $py code\train\evaluate.py --zeroshot siglip2 --labels $lab *>> $jl
& $py code\train\evaluate.py --vlm --labels $lab *>> $jl
$runs = Import-Csv D:\highwaymm\experiments\runs_index.csv | Where-Object { $_.status -eq "completed" -and $_.run_id -match "_p4_" }
$runs | ForEach-Object { "code\train\evaluate.py --run $($_.run_dir) --labels $lab" } | Set-Content -Encoding utf8 D:\highwaymm\experiments\grids\eval_p4t.txt
L "re-evaluating $($runs.Count) p4 runs on v2t"
& $py $jr --gpus 0,1,2 --per-gpu 2 --jobs D:\highwaymm\experiments\grids\eval_p4t.txt *>> $jl
L "P4T DONE"
