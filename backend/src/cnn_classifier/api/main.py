"""FastAPI application: health, prediction, and model metadata (Task 6)."""

import io
import logging
import time
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from cnn_classifier.api.dependencies import ModelUnavailableError, get_predictor
from cnn_classifier.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionItem,
    PredictionResponse,
)
from cnn_classifier.inference.predictor import Predictor
from cnn_classifier.inference.preprocess import preprocess_image

logger = logging.getLogger(__name__)

# Match the Flutter client's 10 MiB upload ceiling.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Dev-only: allows any localhost/127.0.0.1 port. Production should use explicit allowlist.
_LOCAL_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"

# Rate limiter: 30 requests/minute per IP for /predict (prevents DoS via concurrent 10MB uploads).
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title="Image Classifier API", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_LOCAL_ORIGIN_REGEX,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(ModelUnavailableError)
async def model_unavailable(request: Request, exc: ModelUnavailableError) -> JSONResponse:
    """Map checkpoint load failures to a sanitized 503 response."""
    logger.error("model unavailable at %s: %s", request.url.path, exc.__cause__)
    return JSONResponse(status_code=503, content={"detail": "model unavailable"})


@app.exception_handler(ValidationError)
async def validation_error(request: Request, exc: ValidationError) -> JSONResponse:
    """Map response validation errors to a sanitized 500 response."""
    logger.error("response validation error at %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "model inference failed"})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness probe; never loads the model."""
    return HealthResponse()


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("30/minute")
def predict(
    request: Request,
    file: UploadFile = File(...),
    predictor: Predictor = Depends(get_predictor),
) -> PredictionResponse:
    """Classify one uploaded image and return ranked predictions.

    The full ranked distribution is returned; clients slice their top-k.

    Rate limited: 30 requests/minute per IP.
    """
    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="file must be an image")

    data = file.file.read(_MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty image file")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image file too large")

    started = time.perf_counter()
    try:
        with Image.open(io.BytesIO(data)) as image:
            tensor = preprocess_image(image, predictor.input_size)
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="invalid image file") from exc

    try:
        predictions = predictor.predict(tensor)
    except (RuntimeError, ValueError, OSError) as exc:
        logger.exception("model inference failed")
        raise HTTPException(status_code=500, detail="model inference failed") from exc

    latency_ms = (time.perf_counter() - started) * 1000
    return PredictionResponse(
        predictions=[
            PredictionItem(class_name=item.class_name, confidence=item.confidence)
            for item in predictions
        ],
        latency_ms=latency_ms,
        model=predictor.model_name,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(predictor: Predictor = Depends(get_predictor)) -> ModelInfoResponse:
    """Return metadata about the checkpoint currently served.

    ``trained_at`` is the checkpoint file's mtime: checkpoints do not store
    a training timestamp, so this is the best available proxy.
    """
    trained_at = None
    if predictor.checkpoint_path is not None:
        try:
            trained_at = datetime.fromtimestamp(predictor.checkpoint_path.stat().st_mtime, tz=UTC)
        except OSError:
            trained_at = None
    return ModelInfoResponse(
        model=predictor.model_name,
        input_size="x".join(str(dimension) for dimension in predictor.input_size),
        classes=list(predictor.class_names),
        device=predictor.device,
        trained_at=trained_at,
    )
