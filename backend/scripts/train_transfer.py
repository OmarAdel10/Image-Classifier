"""CLI entry: fine-tune a pretrained transfer model (Task 4 spec)."""

import argparse

from cnn_classifier.training.cli import run_training


def parse_args(argv=None):
    """Parse CLI args with transfer defaults (resnet18, lr 1e-4)."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained transfer model on CIFAR-10"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[224, 224],
        metavar=("H", "W"),
    )
    args = parser.parse_args(argv)
    args.input_size = tuple(args.input_size)
    return args


def main() -> None:
    args = parse_args()
    from cnn_classifier.config import Config

    config = Config(input_size=tuple(args.input_size))
    run_training(config, args.model_name, args.epochs, args.lr)


if __name__ == "__main__":
    main()
