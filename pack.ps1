param(
    [string]$OutputPath = (Join-Path $PSScriptRoot ("data-agent_{0}.zip" -f (Get-Date -Format "yyyyMMdd_HHmmss")))
)

$ErrorActionPreference = "Stop"

$root = $PSScriptRoot
$outputFullPath = [System.IO.Path]::GetFullPath($OutputPath)

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.IO.Compression

function Test-ExcludedPath {
    param(
        [string]$Path
    )

    $full = [System.IO.Path]::GetFullPath($Path)
    $relative = $full.Substring($root.Length).TrimStart('\', '/')
    $relativeLower = $relative.ToLowerInvariant()

    $excludedPrefixes = @(
        ".git\",
        ".claude\",
        ".playwright-cli\",
        ".pytest_cache\",
        "test-results\",
        "backend\.venv\",
        "backend\.pytest_cache\",
        "backend\temp_data\",
        "backend\static\images\",
        "frontend\build\",
        "frontend\node_modules\"
    )

    foreach ($prefix in $excludedPrefixes) {
        if ($relativeLower.StartsWith($prefix)) {
            return $true
        }
    }

    $excludedExact = @(
        ".env",
        "backend\.env",
        "backend\backend.out.log",
        "backend\backend.err.log",
        "frontend\frontend.out.log",
        "frontend\frontend.err.log"
    )

    if ($excludedExact -contains $relativeLower) {
        return $true
    }

    if ($relativeLower.EndsWith(".log")) {
        return $true
    }

    if ($relativeLower -like "*.log.*") {
        return $true
    }

    if ($relativeLower -like "*\__pycache__\*") {
        return $true
    }

    if ($relativeLower -like "*\.pytest_cache\*") {
        return $true
    }

    if ($relativeLower -like "*.pyc") {
        return $true
    }

    if ($relativeLower -like "public\*.png") {
        if ($relativeLower -ne "public\show.png" -and $relativeLower -ne "public\fig.png") {
            return $true
        }
    }

    return $false
}

function Get-RelativePath {
    param(
        [string]$BasePath,
        [string]$TargetPath
    )

    $base = [System.IO.Path]::GetFullPath($BasePath).TrimEnd('\', '/')
    $target = [System.IO.Path]::GetFullPath($TargetPath)
    return $target.Substring($base.Length).TrimStart('\', '/')
}

if (Test-Path $outputFullPath) {
    Remove-Item -LiteralPath $outputFullPath -Force
}

$files = Get-ChildItem -LiteralPath $root -Recurse -File -Force |
    Where-Object {
        $full = $_.FullName
        $full -ne $outputFullPath -and -not (Test-ExcludedPath -Path $full)
    }

$zip = [System.IO.Compression.ZipFile]::Open($outputFullPath, [System.IO.Compression.ZipArchiveMode]::Create)

try {
    foreach ($file in $files) {
        $relativePath = Get-RelativePath -BasePath $root -TargetPath $file.FullName
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip,
            $file.FullName,
            $relativePath,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
}
finally {
    $zip.Dispose()
}

Write-Host ("Archive created: {0}" -f $outputFullPath)
Write-Host ("Included files: {0}" -f $files.Count)
