"""CLI entry: train a from-scratch CIFAR-10 CNN (Task 4 spec)."""

import argparse

from cnn_classifier.training.cli import run_training


def parse_args(argv=None):
    """Parse CLI args: --epochs, --lr, --model-name."""
    parser = argparse.ArgumentParser(description="Train a from-scratch CIFAR-10 classifier")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--model-name", type=str, default="cnn_scratch")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    from cnn_classifier.config import get_config

    run_training(get_config(), args.model_name, args.epochs, args.lr)


if __name__ == "__main__":
    main()
