POETRY ?= poetry
PYTHON ?= python

CONDA_ENV := ml-cnn
TORCH_VERSION := 2.14.0

CPU_INDEX := https://download.pytorch.org/whl/cpu
CUDA_INDEX := https://download.pytorch.org/whl/cu126
ROCM_INDEX := https://download.pytorch.org/whl/rocm7.2


.PHONY: help check-env lock check install-base \
        install-mac install-cpu install-cuda install-rocm \
        install-windows-cuda install-linux-cuda install-linux-rocm \
        test lint format device


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo ""
	@echo "Breast-CNN"
	@echo "=========="
	@echo ""
	@echo "Activate Conda first:"
	@echo "  conda activate $(CONDA_ENV)"
	@echo ""
	@echo "Installation:"
	@echo "  make install-mac           macOS / Apple Silicon / MPS"
	@echo "  make install-cpu           Windows/Linux CPU"
	@echo "  make install-windows-cuda  Windows / NVIDIA"
	@echo "  make install-linux-cuda    Linux / NVIDIA"
	@echo "  make install-linux-rocm    Linux / AMD"
	@echo ""
	@echo "Development:"
	@echo "  make test"
	@echo "  make lint"
	@echo "  make format"
	@echo "  make device"
	@echo ""


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

check-env:
	@$(PYTHON) -c "import os, sys; env = os.environ.get('CONDA_DEFAULT_ENV'); sys.exit(0 if env == '$(CONDA_ENV)' else 'ERROR: run: conda activate $(CONDA_ENV)')"


# ---------------------------------------------------------------------------
# Poetry
# ---------------------------------------------------------------------------

lock: check-env
	$(POETRY) lock


check: check-env
	$(POETRY) check


install-base: check-env
	$(POETRY) install --with dev,experiment


# ---------------------------------------------------------------------------
# PyTorch
# ---------------------------------------------------------------------------

# macOS / Apple Silicon / MPS
install-mac: install-base
	$(PYTHON) -m pip install "torch==$(TORCH_VERSION)"


# Windows/Linux CPU
install-cpu: install-base
	$(PYTHON) -m pip install "torch==$(TORCH_VERSION)+cpu" --index-url $(CPU_INDEX)


# Windows/Linux NVIDIA CUDA 12.6
install-cuda: install-base
	$(PYTHON) -m pip install "torch==$(TORCH_VERSION)+cu126" --index-url $(CUDA_INDEX)


# Linux AMD ROCm 7.2
install-rocm: install-base
	$(PYTHON) -m pip install "torch==$(TORCH_VERSION)+rocm7.2" --index-url $(ROCM_INDEX)


install-windows-cuda: install-cuda

install-linux-cuda: install-cuda

install-linux-rocm: install-rocm


# ---------------------------------------------------------------------------
# Tests / quality
# ---------------------------------------------------------------------------

test: check-env
	$(PYTHON) -m pytest


lint: check-env
	$(PYTHON) -m ruff check src tests


format: check-env
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests


# ---------------------------------------------------------------------------
# Hardware diagnostics
# ---------------------------------------------------------------------------

device: check-env
	$(PYTHON) -c "import platform, torch; print('OS:', platform.platform()); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('ROCm/HIP version:', torch.version.hip); print('MPS available:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"