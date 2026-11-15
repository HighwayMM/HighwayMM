# Multi-GPU launcher for native Windows (gloo DDP via torchrun).
#   powershell -File code\train\launch.ps1 -Gpus 3 code\train\main.py --config code\train\configs\synthetic.yaml
#   powershell -File code\train\launch.ps1 -Gpus 1 -Visible 0 code\train\bench.py --mode all
# Launcher options (-Gpus N, -Visible "0,1", -Threads N) must come BEFORE the script path;
# everything from the script path on is passed through verbatim.
# -Visible restricts CUDA_VISIBLE_DEVICES (e.g. "0,1" to keep GPU 2 / the display GPU free).
$ErrorActionPreference = "Stop"
$Gpus = 3; $Visible = ""; $Threads = 4
$i = 0
while ($i -lt $args.Count) {
    switch -Regex ($args[$i]) {
        '^-Gpus$'    { $Gpus = [int]$args[$i + 1]; $i += 2; continue }
        '^-Visible$' { $Visible = [string]$args[$i + 1]; $i += 2; continue }
        '^-Threads$' { $Threads = [int]$args[$i + 1]; $i += 2; continue }
        default      { break }
    }
    if ($args[$i] -notmatch '^-(Gpus|Visible|Threads)$') { break }
}
$Rest = @($args[$i..($args.Count - 1)])
$py = "D:\highwaymm\analysis\.venv\Scripts\python.exe"
$env:USE_LIBUV = "0"                 # Windows torch has no libuv TCPStore
$env:OMP_NUM_THREADS = "$Threads"
$env:PYTHONUNBUFFERED = "1"
if ($Visible) { $env:CUDA_VISIBLE_DEVICES = $Visible }
$port = Get-Random -Minimum 29500 -Maximum 29999
# torchrun is broken on Windows torch 2.5 (libuv TCPStore); spawn.py implements the same env contract.
$argv = @("D:\highwaymm\code\train\spawn.py", "--nproc", "$Gpus", "--port", "$port") + $Rest
$env:LAUNCH_CMD = "launch.ps1 -Gpus $Gpus" + $(if ($Visible) { " -Visible $Visible" } else { "" }) + " " + ($Rest -join " ")
Push-Location D:\highwaymm
try { & $py @argv; $code = $LASTEXITCODE } finally { Pop-Location }
exit $code
