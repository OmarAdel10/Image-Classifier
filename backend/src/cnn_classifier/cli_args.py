"""Shared CLI argument validators for the training entry points."""

import argparse
import math


def positive_int(value: str) -> int:
    """Argparse type: a positive integer (rejects 0 and negatives)."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_float(value: str) -> float:
    """Argparse type: a positive finite float (rejects 0, negatives, nan, inf)."""
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return number
