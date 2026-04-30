$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$python = Join-Path $backend ".venv\Scripts\python.exe"
$chromeCandidates = @(
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
)
$chromeExe = $chromeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

function Test-Port {
    param(
        [int]$Port
    )

    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $iar = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $iar.AsyncWaitHandle.WaitOne(500)) {
            $client.Close()
            return $false
        }

        $client.EndConnect($iar)
        $client.Close()
        return $true
    }
    catch {
        return $false
    }
}

function Wait-Port {
    param(
        [int]$Port,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port -Port $Port) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }

    return $false
}

# Avoid Unicode console issues on Windows when the backend prints Chinese text.
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Port -Port 8002)) {
    if (-not (Test-Path $python)) {
        throw "Backend Python not found: $python"
    }

    Start-Process -FilePath $python -ArgumentList @("-m", "src.server") -WorkingDirectory $backend
    Write-Host "Backend starting on http://127.0.0.1:8002"
}
else {
    Write-Host "Backend already running on port 8002"
}

if (-not (Wait-Port -Port 8002 -TimeoutSeconds 30)) {
    throw "Backend did not become ready on port 8002."
}

if (-not (Test-Port -Port 3000)) {
    Start-Process -FilePath "npm.cmd" -ArgumentList @("run", "dev", "--", "--host", "0.0.0.0") -WorkingDirectory $frontend
    Write-Host "Frontend starting on http://127.0.0.1:3000"
}
else {
    Write-Host "Frontend already running on port 3000"
}

if (-not (Wait-Port -Port 3000 -TimeoutSeconds 30)) {
    throw "Frontend did not become ready on port 3000."
}

$cacheBust = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$targetUrl = "http://127.0.0.1:3000/?v=$cacheBust"
if ($chromeExe) {
    Start-Process -FilePath $chromeExe -ArgumentList @(
        "--new-tab",
        $targetUrl
    )
}
else {
    Start-Process $targetUrl
}
