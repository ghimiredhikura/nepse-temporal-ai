$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot

if ($env:CONDA_DEFAULT_ENV -ne "nepalai" -or -not $env:CONDA_PREFIX) {
    throw "Activate the nepalai conda environment before running this script."
}
$pythonExecutable = Join-Path $env:CONDA_PREFIX "python.exe"
if (-not (Test-Path -LiteralPath $pythonExecutable)) {
    throw "Python was not found at $pythonExecutable"
}

function Invoke-LoggedStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$PythonArguments,

        [Parameter(Mandatory = $true)]
        [string]$LogPath
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"
    & $script:pythonExecutable @PythonArguments 2>&1 |
        Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Invoke-LoggedStep `
    -Name "1/2 Year-specific inference, calibration, and decision utility" `
    -PythonArguments @(
        "scripts\finalize_surveillance_evidence.py",
        "--config",
        "configs\experiments\surveillance_journal_robustness.json"
    ) `
    -LogPath "data\manifests\surveillance_journal_robustness.log"

Invoke-LoggedStep `
    -Name "2/2 Temporal explanation parameter-randomization sanity check" `
    -PythonArguments @(
        "scripts\validate_explanation_sanity.py",
        "--config",
        "configs\experiments\surveillance_explanation_sanity.json"
    ) `
    -LogPath "data\manifests\surveillance_explanation_sanity.log"

Write-Host ""
Write-Host "Journal-completion analyses finished successfully."
