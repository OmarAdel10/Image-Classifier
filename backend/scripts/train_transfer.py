"""CLI entry: fine-tune a pretrained transfer model (Task 4 spec)."""

import argparse

from cnn_classifier.training.cli import run_training


def parse_args(argv=None):
    """Parse CLI args with transfer defaults (resnet18, lr 1e-4).

    --input-size defaults to CIFAR-10's native 32x32: get_dataloaders
    pins (32, 32), and ResNet18's AdaptiveAvgPool handles 32x32 fine
    (standard CIFAR transfer practice). Override for other sizes.
    """
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
        default=[32, 32],
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
