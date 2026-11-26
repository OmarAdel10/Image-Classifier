#!/usr/bin/env bash
# Train the first model checkpoint for the CNN classifier.
# Run this once after setting up the environment.
# Usage: ./scripts/train_first_model.sh [--model resnet18|cnn_scratch] [--epochs N]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_DIR="${REPO_ROOT}/backend"

MODEL="resnet18"
EPOCHS=10
EXTRA_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --lr)
            EXTRA_ARGS+=("--lr" "$2")
            shift 2
            ;;
        --batch-size)
            EXTRA_ARGS+=("--batch-size" "$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model resnet18|cnn_scratch] [--epochs N] [--lr LR] [--batch-size BS]"
            echo ""
            echo "Trains a CIFAR-10 model and saves checkpoint to backend/checkpoints/best_model.pt"
            echo ""
            echo "Options:"
            echo "  --model MODEL      Model to train: resnet18 (transfer, default) or cnn_scratch"
            echo "  --epochs N         Number of epochs (default: 10 for resnet18, 30 for cnn_scratch)"
            echo "  --lr LR            Learning rate"
            echo "  --batch-size BS    Batch size"
            echo "  -h, --help         Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default epochs based on model if not explicitly provided
if [[ $# -eq 0 ]] && [[ "$MODEL" == "cnn_scratch" ]]; then
    EPOCHS=30
fi

cd "${BACKEND_DIR}"

echo "========================================="
echo "Training first model: ${MODEL}"
echo "Epochs: ${EPOCHS}"
echo "Checkpoint: ${BACKEND_DIR}/checkpoints/best_model.pt"
echo "Labels:     ${BACKEND_DIR}/checkpoints/labels.json"
echo "========================================="

# Create checkpoint directory
mkdir -p checkpoints

# Activate venv if it exists, otherwise use system python
if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
fi

# Run training (PYTHONPATH includes src/ so cnn_classifier package is importable)
export PYTHONPATH="${BACKEND_DIR}/src:${PYTHONPATH:-}"

if [[ "$MODEL" == "cnn_scratch" ]]; then
    python scripts/train_scratch.py \
        --epochs "${EPOCHS}" \
        --model-name "cnn_scratch" \
        "${EXTRA_ARGS[@]}"
else
    python scripts/train_transfer.py \
        --epochs "${EPOCHS}" \
        --model-name "resnet18" \
        "${EXTRA_ARGS[@]}"
fi

# Verify checkpoint was created
if [[ -f "checkpoints/best_model.pt" && -f "checkpoints/labels.json" ]]; then
    echo ""
    echo "========================================="
    echo "✓ Training complete!"
    echo "Checkpoint: checkpoints/best_model.pt"
    echo "Labels:     checkpoints/labels.json"
    echo ""
    echo "Next steps:"
    echo "  cd ${REPO_ROOT}"
    echo "  flutter run -d linux"
    echo "========================================="
else
    echo ""
    echo "========================================="
    echo "✗ Training failed - checkpoint not found"
    echo "Check the logs above for errors"
    echo "========================================="
    exit 1
fi