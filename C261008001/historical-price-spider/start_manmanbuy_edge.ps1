param(
    [int]$Port = 9222,
    [string]$ProductUrl = "https://item.jd.com/100041149481.html"
)

$edgeCandidates = @(
    "$env:ProgramFiles (x86)\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:LOCALAPPDATA\Microsoft\Edge\Application\msedge.exe"
)
$edgePath = $edgeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $edgePath) {
    throw "未找到 Microsoft Edge"
}

$profilePath = Join-Path $PSScriptRoot ".browser-profile\manmanbuy-edge"
$queryUrl = "https://tool.manmanbuy.com/HistoryLowest.aspx?url=$([uri]::EscapeDataString($ProductUrl))"
New-Item -ItemType Directory -Path $profilePath -Force | Out-Null
Start-Process -FilePath $edgePath -WindowStyle Hidden -ArgumentList @(
    "--remote-debugging-address=127.0.0.1",
    "--remote-debugging-port=$Port",
    "--user-data-dir=$profilePath",
    $queryUrl
)
Write-Host "已启动专用 Edge。首次使用请在窗口中手动登录慢慢买。"
