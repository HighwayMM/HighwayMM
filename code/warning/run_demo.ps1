# Replay demos used in the supplement and the GitHub README (3 GPUs; each line also runs alone on one GPU or the CPU).
# Inputs are built by make_demo_inputs.py (hard links; originals untouched).
param([string]$py = ".\analysis\.venv\Scripts\python.exe")
$ErrorActionPreference = "Continue"
$demo = "code\warning\live_demo.py"
New-Item -ItemType Directory -Force demo\logs | Out-Null
$ours = Get-Content demo\inputs\ours_winter.txt
$runs = @(
  @{gpu = "0"; log = "nysdot";         args = @("--input", "demo\inputs\nysdot", "--gt", "demo\inputs\nysdot_gt.csv", "--out", "demo\out\nysdot", "--reset_gap", "720", "--gif")},
  @{gpu = "1"; log = "floodcam_all";   args = @("--input", "demo\inputs\floodcam", "--gt", "demo\inputs\floodcam_gt.csv", "--out", "demo\out\floodcam_all", "--gif")},
  @{gpu = "1"; log = "floodcam_focus"; args = @("--input", "demo\inputs\floodcam", "--gt", "demo\inputs\floodcam_gt.csv", "--out", "demo\out\floodcam_focus", "--focus", "flooding", "--tau", "0.9", "--gif")},
  @{gpu = "2"; log = "ours_winter";    args = @("--input") + $ours + @("--camera", "per_file", "--window", "10", "--layout", "anytext", "--out", "demo\out\ours_winter", "--gif")}
)
$procs = foreach ($r in $runs) {
  $env:CUDA_VISIBLE_DEVICES = $r.gpu
  $al = @($demo) + $r.args | ForEach-Object { if ($_ -match " ") { '"' + $_ + '"' } else { $_ } }
  Start-Process -FilePath $py -ArgumentList $al -NoNewWindow -PassThru `
    -RedirectStandardOutput "demo\logs\$($r.log).out" -RedirectStandardError "demo\logs\$($r.log).err"
}
$procs | Wait-Process
foreach ($r in $runs) { Write-Output "== $($r.log)"; Get-Content "demo\logs\$($r.log).out" | Select-String "\[demo\]" }
