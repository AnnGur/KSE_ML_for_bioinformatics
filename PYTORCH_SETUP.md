# PyTorch setup for the practicals

The Week 6 practical and the Lecture 6 examples use PyTorch. The repository defines two mutually exclusive uv extras:

- "cpu" installs the CPU-only PyTorch wheel.
- "cu128" installs the PyTorch wheel built for CUDA 12.8 and is the recommended choice for a recent NVIDIA GPU.

The lockfile contains both alternatives, but students install only the extra they select.

## Recommended installation

From the repository root, after installing uv:

    uv sync --extra cpu

Use this option for laptops without an NVIDIA GPU, macOS, AMD GPUs, or when GPU setup is uncertain.

For a recent NVIDIA GPU on Windows or Linux:

    uv sync --extra cu128

This installs the same course environment with the CUDA-enabled PyTorch build. A compatible NVIDIA driver is required. The full CUDA toolkit is not normally needed for these notebooks because the PyTorch wheel provides the CUDA runtime libraries it needs.

## Automatic PowerShell setup

On Windows, the included script selects "cu128" when nvidia-smi is available and otherwise selects "cpu":

    .\setup_pytorch.ps1

The backend can also be selected explicitly:

    .\setup_pytorch.ps1 -Backend cpu
    .\setup_pytorch.ps1 -Backend cu128

If PowerShell blocks local scripts, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

and then run the setup script again.

## Verify the installation

CPU environment:

    uv run --extra cpu python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

CUDA environment:

    uv run --extra cu128 python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU not detected')"

For the CUDA setup, torch.cuda.is_available() should print True. If it prints False, update the NVIDIA driver and rerun the verification. If the computer has an older driver that cannot run the CUDA 12.8 wheel, use the CPU setup or select a compatible CUDA build using the official PyTorch installer instructions.

The environment uses Python 3.11, as specified in .python-version. Do not install both "cpu" and "cu128"; uv intentionally marks them as conflicting alternatives.

Official references:

- uv PyTorch integration: https://docs.astral.sh/uv/guides/integration/pytorch/
- PyTorch local installation selector: https://pytorch.org/get-started/locally/
