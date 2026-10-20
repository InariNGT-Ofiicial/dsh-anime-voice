param(
    [Parameter(Mandatory=$true)][string]$SkillRoot,
    [switch]$Update
)
$ErrorActionPreference = 'Stop'
$names = @('anime-voice-line-extractor', 'anime-voice-clone-training')
$root = [IO.Path]::GetFullPath($SkillRoot)
New-Item -ItemType Directory -Force -Path $root | Out-Null
foreach ($name in $names) {
    $dest = Join-Path $root $name
    if ((Test-Path -LiteralPath $dest) -and -not $Update) {
        throw "$dest already exists; use -Update after reviewing the changes."
    }
}
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
foreach ($name in $names) {
    $dest = Join-Path $root $name
    if (Test-Path -LiteralPath $dest) {
        $backup = Join-Path $root "_anime-voice-backups/$stamp/$name"
        New-Item -ItemType Directory -Force -Path (Split-Path $backup) | Out-Null
        Copy-Item -LiteralPath $dest -Destination $backup -Recurse
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item -Path (Join-Path $PSScriptRoot "skills/$name/*") -Destination $dest -Recurse -Force
    Write-Output "Installed $name -> $dest"
}
