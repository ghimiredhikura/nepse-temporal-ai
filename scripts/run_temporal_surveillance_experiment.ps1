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

function Invoke-ExperimentStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string[]]$PythonArguments
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host $Name
    Write-Host "============================================================"
    & $script:pythonExecutable @PythonArguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Invoke-ExperimentStep `
    -Name "1/4 Frozen 2024-2025 surveillance baselines" `
    -PythonArguments @(
        "scripts\run_surveillance_baselines.py",
        "--config",
        "configs\experiments\surveillance_baselines_2024_2025.json",
        "--resume"
    )

Invoke-ExperimentStep `
    -Name "2/4 Four-stage temporal GRU evaluation" `
    -PythonArguments @(
        "scripts\run_temporal_surveillance.py",
        "--config",
        "configs\experiments\temporal_surveillance_2024_2025.json",
        "--resume"
    )

Invoke-ExperimentStep `
    -Name "3/4 Calibration, uncertainty, regimes, and block bootstrap" `
    -PythonArguments @(
        "scripts\analyze_temporal_surveillance.py",
        "--config",
        "configs\experiments\temporal_surveillance_analysis.json"
    )

Invoke-ExperimentStep `
    -Name "4/4 Temporal occlusion and LightGBM TreeSHAP" `
    -PythonArguments @(
        "scripts\explain_temporal_surveillance.py",
        "--config",
        "configs\experiments\surveillance_explainability.json"
    )

Write-Host ""
Write-Host "Temporal surveillance experiment completed successfully."
