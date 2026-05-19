$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $output = & git @Args 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw (($output | Out-String).Trim())
    }
    return $output
}

function Read-NonEmptyInput {
    param(
        [string]$Prompt,
        [string]$DefaultValue = ""
    )

    $value = Read-Host $Prompt
    if ([string]::IsNullOrWhiteSpace($value)) {
        return $DefaultValue
    }
    return $value.Trim()
}

Write-Host "Repository: $root"

$remoteNames = Invoke-Git -Args @("remote")
if (-not ($remoteNames -contains "github")) {
    throw "Git remote 'github' 不存在，请先配置 github 远端。"
}

$currentBranch = (Invoke-Git -Args @("branch", "--show-current") | Select-Object -First 1).Trim()
if ([string]::IsNullOrWhiteSpace($currentBranch)) {
    throw "无法识别当前分支。"
}

if ($currentBranch -ne "main") {
    $confirmNonMain = Read-Host "当前分支是 '$currentBranch'，是否将当前 HEAD 推送到 github/main？输入 y 继续"
    if ($confirmNonMain -ne "y") {
        Write-Host "已取消。"
        exit 0
    }
}

$status = Invoke-Git -Args @("status", "--short")
$hasChanges = ($status | Measure-Object).Count -gt 0

if ($hasChanges) {
    Write-Host ""
    Write-Host "检测到以下改动："
    $status | ForEach-Object { Write-Host $_ }
    Write-Host ""

    $defaultMessage = "Update project on $(Get-Date -Format "yyyy-MM-dd HH:mm")"
    $commitMessage = Read-NonEmptyInput -Prompt "输入 commit message（直接回车用默认值）" -DefaultValue $defaultMessage

    Invoke-Git -Args @("add", "-A")
    Invoke-Git -Args @("commit", "-m", $commitMessage) | Out-Null
    Write-Host "Commit 完成：$commitMessage"
}
else {
    Write-Host "没有未提交改动，准备直接推送当前提交。"
}

Write-Host "正在推送到 github/main ..."
Invoke-Git -Args @("push", "github", "HEAD:main") | Out-Null

$latestCommit = (Invoke-Git -Args @("rev-parse", "--short", "HEAD") | Select-Object -First 1).Trim()
Write-Host "推送完成，当前提交：$latestCommit"
Read-Host "按回车结束"
