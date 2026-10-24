"""CLI entry: train a from-scratch CIFAR-10 CNN (Task 4 spec)."""

import argparse
import logging

from cnn_classifier.training.cli import run_training


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value):
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return number


def parse_args(argv=None):
    """Parse CLI args: --epochs, --lr, --model-name."""
    parser = argparse.ArgumentParser(description="Train a from-scratch CIFAR-10 classifier")
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--model-name", type=str, default="cnn_scratch")
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    from cnn_classifier.config import get_config

    run_training(get_config(), args.model_name, args.epochs, args.lr)


if __name__ == "__main__":
    main()
