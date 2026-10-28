param(
    [ValidateSet("auto", "cpu", "cu128")]
    [string]$Backend = "auto"
)

$ErrorActionPreference = "Stop"

if ($Backend -eq "auto") {
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        $Backend = "cu128"
    }
    else {
        $Backend = "cpu"
    }
}

Write-Host "Installing the course environment with PyTorch backend: $Backend"
uv sync --extra $Backend

if ($Backend -eq "cu128") {
    uv run --extra cu128 python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not detected')"
}
else {
    uv run --extra cpu python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
}
