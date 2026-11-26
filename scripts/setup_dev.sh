#!/usr/bin/env bash
# Development environment setup for the CNN classifier project.
# Installs Python dependencies and prepares the backend environment.
# Run this once after cloning the repository.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"

echo "========================================="
echo "Setting up development environment"
echo "Repo root: ${REPO_ROOT}"
echo "Backend:   ${BACKEND_DIR}"
echo "========================================="

cd "${BACKEND_DIR}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "✗ Python 3 not found. Please install Python 3.11+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: ${PYTHON_VERSION}"

# Create virtual environment if it doesn't exist
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate venv
# shellcheck source=/dev/null
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install backend package in development mode with dev dependencies
echo "Installing backend package..."
pip install -e ".[dev]"

# Check if CUDA is available and install appropriate torch
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "No NVIDIA GPU detected, using CPU-only PyTorch (already installed via pyproject.toml)"
fi

# Create necessary directories
mkdir -p checkpoints data logs

echo ""
echo "========================================="
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Train a model: ./scripts/train_first_model.sh"
echo "  2. Run the app:   cd ${REPO_ROOT} && flutter run -d linux"
echo "========================================="