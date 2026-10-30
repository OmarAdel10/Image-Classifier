#!/usr/bin/env python3
"""Export a trained PyTorch model to ONNX format for deployment."""

import argparse

import torch

from cnn_classifier.config import Config
from cnn_classifier.inference.checkpoints import load_checkpoint
from cnn_classifier.models.registry import get_model


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: str = None,
    input_size: tuple = None,
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> None:
    """
    Export a trained PyTorch checkpoint to ONNX format.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        output_path: Path where the ONNX model will be saved.
        model_name: Model architecture name (e.g., 'cnn_scratch', 'resnet18').
                    If None, inferred from checkpoint.
        input_size: Input resolution (H, W). If None, inferred from checkpoint.
        opset_version: ONNX opset version to use.
        dynamic_batch: Whether to make batch dimension dynamic.
    """
    checkpoint = load_checkpoint(checkpoint_path)

    # Infer model name and input size from checkpoint if not provided
    if model_name is None:
        model_name = checkpoint["config"].model_name
    if input_size is None:
        input_size = checkpoint["config"].input_size

    # Create model and load weights
    config = Config(
        model_name=model_name,
        input_size=input_size,
        device="cpu",
    )
    model = get_model(model_name, config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Dynamic axes for batch dimension
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )

    print(f"Exported ONNX model to {output_path}")
    print(f"Model: {model_name}, Input size: {input_size}")
    print(f"Opset version: {opset_version}")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the .pt checkpoint file",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to save the ONNX model",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model architecture name (e.g., cnn_scratch, resnet18). Inferred from checkpoint if not provided.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("H", "W"),
        help="Input resolution (H W). Inferred from checkpoint if not provided.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--no-dynamic-batch",
        action="store_true",
        help="Disable dynamic batch dimension in ONNX",
    )

    args = parser.parse_args()

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model_name,
        input_size=tuple(args.input_size) if args.input_size else None,
        opset_version=args.opset_version,
        dynamic_batch=not args.no_dynamic_batch,
    )


if __name__ == "__main__":
    main()
