# Detached launcher: survives the shell that started it.
# usage: powershell -File launch.ps1 <gpu> <logname> <python args...>
param([string]$gpu, [string]$log)
$env:CUDA_VISIBLE_DEVICES = $gpu
$env:HF_HUB_OFFLINE = "1"
$env:PYTHONUNBUFFERED = "1"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$work = if ($env:ESC_WORK) { $env:ESC_WORK } else { "C:\esc_work" }
New-Item -ItemType Directory -Force (Join-Path $work "logs") | Out-Null
$out = Join-Path $work "logs\$log.log"
$err = Join-Path $work "logs\$log.err"
$p = Start-Process -FilePath "python" -ArgumentList $args -WorkingDirectory $root `
    -RedirectStandardOutput $out -RedirectStandardError $err -WindowStyle Hidden -PassThru
Write-Output "pid $($p.Id) -> $out"

