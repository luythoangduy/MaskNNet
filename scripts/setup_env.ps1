param(
    [string]$PythonExe = "python",
    [string]$VenvDir = ".venv",
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

Write-Host "[1/5] Create virtual environment at $VenvDir"
& $PythonExe -m venv $VenvDir

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

Write-Host "[2/5] Upgrade pip/setuptools/wheel"
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "[3/5] Install PyTorch"
if ($CpuOnly) {
    & $VenvPip install torch torchvision torchaudio
} else {
    & $VenvPip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
}

Write-Host "[4/5] Install project dependencies"
& $VenvPip install -r requirements.txt
& $VenvPip install gdown

Write-Host "[5/5] Install repo in editable mode"
& $VenvPip install -e .

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\$VenvDir\Scripts\Activate.ps1"
