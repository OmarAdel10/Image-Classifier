"""Model definitions for the CNN classifier (Task 3)."""

from cnn_classifier.models.cnn_scratch import CNNScratch
from cnn_classifier.models.registry import MODEL_REGISTRY, get_model
from cnn_classifier.models.transfer import TransferModel

__all__ = ["CNNScratch", "TransferModel", "get_model", "MODEL_REGISTRY"]
