"""CLI entry: fine-tune a pretrained transfer model (Task 4 spec)."""

import argparse
import logging
import os

from cnn_classifier.cli_args import positive_float, positive_int
from cnn_classifier.training.cli import run_training


def parse_args(argv=None):
    """Parse CLI args with transfer defaults (resnet18, lr 1e-4).

    --input-size defaults to CIFAR-10's native 32x32: get_dataloaders
    pins (32, 32), and ResNet18's AdaptiveAvgPool handles 32x32 fine
    (standard CIFAR transfer practice).
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained transfer model on CIFAR-10"
    )
    parser.add_argument("--epochs", type=positive_int, default=30)
    parser.add_argument("--lr", type=positive_float, default=1e-4)
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[32, 32],
        metavar=("H", "W"),
        help="only 32 32 currently supported (CIFAR-10 native)",
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
    os.environ.setdefault("CNN_MODEL_PATH", f"checkpoints/best_model_{args.model_name}.pt")
    from cnn_classifier.config import Config

    config = Config(input_size=tuple(args.input_size))
    run_training(config, args.model_name, args.epochs, args.lr)


if __name__ == "__main__":
    main()
