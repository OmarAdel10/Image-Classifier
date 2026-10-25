"""CLI entry: fine-tune a pretrained transfer model (Task 4 spec)."""

import argparse
import logging
import math

from cnn_classifier.training.cli import run_training


def _positive_int(value):
    number = int(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return number


def parse_args(argv=None):
    """Parse CLI args with transfer defaults (resnet18, lr 1e-4).

    --input-size defaults to CIFAR-10's native 32x32: get_dataloaders
    pins (32, 32), and ResNet18's AdaptiveAvgPool handles 32x32 fine
    (standard CIFAR transfer practice).
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained transfer model on CIFAR-10"
    )
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--lr", type=_positive_float, default=1e-4)
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
    if args.input_size != (32, 32):
        parser.error(
            "--input-size currently supports only 32 32 "
            "(CIFAR-10 native); resize support is a follow-up"
        )
    return args


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    from cnn_classifier.config import Config

    config = Config(input_size=tuple(args.input_size))
    run_training(config, args.model_name, args.epochs, args.lr)


if __name__ == "__main__":
    main()
