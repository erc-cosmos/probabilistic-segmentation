# Dot-source this file: . .\activate.ps1
$CosmosBin = Join-Path $PSScriptRoot '.tools/bin'
if (!(Test-Path (Join-Path $CosmosBin 'pixi.exe'))) { throw 'Run bootstrap.ps1 first.' }
$env:PATH = "$CosmosBin;$env:PATH"
$env:PIXI_CACHE_DIR = Join-Path $PSScriptRoot '.cache/pixi'
Remove-Variable CosmosBin
