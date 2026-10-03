param(
    [ValidateSet('bootstrap','setup','add','update','upgrade','run','help')]
    [string]$Action = 'help',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$RunArgs
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Pixi = Join-Path $Root '.tools/bin/pixi.exe'
$Version = '0.81.0'
$env:PIXI_CACHE_DIR = Join-Path $Root '.cache/pixi'
function Invoke-Pixi {
    & $Pixi @args
    if ($LASTEXITCODE -ne 0) { throw "Pixi failed (exit $LASTEXITCODE). See the message above." }
}
if ($Action -eq 'bootstrap') {
    if (!(Test-Path $Pixi) -or ((& $Pixi --version) -ne "pixi $Version")) {
        $Installer = Join-Path ([IO.Path]::GetTempPath()) (([guid]::NewGuid().ToString()) + '.ps1')
        try {
            Invoke-WebRequest -UseBasicParsing "https://raw.githubusercontent.com/prefix-dev/pixi/v$Version/install/install.ps1" -OutFile $Installer
            $env:PIXI_VERSION = $Version
            $env:PIXI_HOME = Join-Path $Root '.tools'
            $env:PIXI_NO_PATH_UPDATE = '1'
            & $Installer -PixiVersion $Version -PixiHome $env:PIXI_HOME -NoPathUpdate
        } finally { Remove-Item $Installer -ErrorAction SilentlyContinue }
    }
    Invoke-Pixi --version
    Write-Host 'Pixi is ready. Next: .\cosmos.ps1 setup'
    exit 0
}
if ($Action -eq 'help') {
    Write-Host '.\cosmos.ps1 {bootstrap|setup|add|update|upgrade|run COMMAND [ARGS...]}'
    exit 0
}
if (!(Test-Path $Pixi)) { throw 'Run .\bootstrap.ps1 first.' }
if ((& $Pixi --version) -ne "pixi $Version") { throw 'Pixi version mismatch. Run .\bootstrap.ps1.' }
$Manifest = Join-Path $Root 'pixi.toml'
switch ($Action) {
    'setup' { Invoke-Pixi install --locked --manifest-path $Manifest }
    'update' { Invoke-Pixi install --manifest-path $Manifest }
    'upgrade' { Invoke-Pixi update --manifest-path $Manifest }
    'add' {
        $Package = Read-Host 'Package name (e.g. numpy)'
        $VersionPin = Read-Host 'Exact version (e.g. 1.26.4)'
        $Source = Read-Host 'Source: conda or pypi [conda]'
        if ($Package -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]*$') { throw 'Invalid package name.' }
        if ($VersionPin -notmatch '^[0-9][A-Za-z0-9.!+_-]*$') { throw 'Enter an exact version, without comparison signs.' }
        if (!$Source -or $Source -eq 'conda') {
            Invoke-Pixi add --manifest-path $Manifest "$Package==$VersionPin"
        } elseif ($Source -eq 'pypi') {
            Invoke-Pixi add --manifest-path $Manifest --pypi "$Package==$VersionPin"
        } else { throw 'Source must be conda or pypi.' }
    }
    'run' {
        if (!$RunArgs) { throw 'Example: .\cosmos.ps1 run python your_script.py' }
        Invoke-Pixi run --locked --manifest-path $Manifest -- @RunArgs
    }
}
