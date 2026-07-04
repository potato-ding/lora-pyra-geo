$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$ExperimentName = "exp_E_parallel_lk_psa025"
$OutputDir = "src/checkpoint/student/$ExperimentName"
$LogDir = "logs"
$LogPath = "$LogDir/$ExperimentName.log"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$TrainArgs = @(
    "src/training/student_train.py",
    "--output_dir", $OutputDir,
    "--enable_lk_adapter", "true",
    "--enable_psa_tiny", "true",
    "--psa_ratio", "0.25",
    "--adapter_fusion_mode", "parallel"
)

& python @TrainArgs @args 2>&1 | Tee-Object -FilePath $LogPath
exit $LASTEXITCODE
