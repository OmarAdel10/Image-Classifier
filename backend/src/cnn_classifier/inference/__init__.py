"""Inference pipeline: preprocessing and prediction (Task 5)."""

from cnn_classifier.inference.predictor import Prediction, Predictor
from cnn_classifier.inference.preprocess import preprocess_image

__all__ = ["preprocess_image", "Prediction", "Predictor"]
