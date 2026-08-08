$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$venvDir = Join-Path $projectRoot ".venv-neural"
$python = Join-Path $venvDir "Scripts\python.exe"

# Base interpreter used to create the venv (Python 3.10-3.12 recommended for
# torch/transformers wheels). Override via MEDIGRAPH_BASE_PYTHON.
function Find-BasePython {
    if ($env:MEDIGRAPH_BASE_PYTHON) { return $env:MEDIGRAPH_BASE_PYTHON }
    foreach ($ver in @("3.12", "3.11", "3.10")) {
        $py = Get-Command py -ErrorAction SilentlyContinue
        if ($py) {
            & py "-$ver" -c "pass" 2>$null
            if ($LASTEXITCODE -eq 0) { return "py -$ver" }
        }
    }
    return "python"
}

if (-not (Test-Path -LiteralPath $python)) {
    $basePython = Find-BasePython
    Write-Host "Creating neural runtime venv: $venvDir (base: $basePython)"
    if ($basePython -like "py -*") {
        $ver = $basePython.Split(" ")[1]
        & py $ver -m venv $venvDir
    } else {
        & $basePython -m venv $venvDir
    }
    if (-not (Test-Path -LiteralPath $python)) {
        throw "Failed to create venv. Install Python 3.10-3.12 or set MEDIGRAPH_BASE_PYTHON."
    }
}

Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip

Write-Host "Installing CPU PyTorch..."
& $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

Write-Host "Installing MediGraph requirements..."
& $python -m pip install -r (Join-Path $projectRoot "requirements.txt")

Write-Host "Neural runtime ready: $python"
& $python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__)"
