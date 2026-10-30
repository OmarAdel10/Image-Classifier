# Image Classifier

A CIFAR-10 image classification system with a PyTorch/FastAPI backend and a Flutter Linux desktop frontend.

## Architecture

```
┌─────────────────┐     HTTP/JSON      ┌──────────────────┐
│  Flutter App    │ ◄─────────────────► │  FastAPI Backend │
│  (Linux Desktop)│   REST API + CORS   │  (PyTorch Model) │
└─────────────────┘                     └──────────────────┘
        │                                        │
        │ Drag & drop / file picker              │
        ▼                                        ▼
┌─────────────────┐                     ┌──────────────────┐
│ Image Preview   │                     │ PyTorch Model    │
│ Prediction Bars │                     │ CIFAR-10 Classes │
│ Model Info Card │                     │ 10 Classes       │
└─────────────────┘                     └──────────────────┘
```

## Project Structure

```
.
├── backend/                    # FastAPI + PyTorch backend
│   ├── src/
│   │   └── cnn_classifier/
│   │       ├── api/            # FastAPI routes (health, predict, model-info)
│   │       ├── config.py       # Configuration (env-configurable)
│   │       ├── inference/      # Checkpoint loading, preprocessing, prediction
│   │       ├── models/         # Model registry (cnn_scratch, resnet18, vgg16)
│   │       ├── training/       # Training loop, checkpointing
│   │       └── inference/      # ONNX export script
│   ├── scripts/
│   │   ├── train_scratch.py    # Train from scratch
│   │   └── export_onnx.py      # Export to ONNX
│   ├── tests/                  # pytest suite (159 tests)
│   └── pyproject.toml
├── lib/                        # Flutter app
│   ├── config.dart             # Config (--dart-define overrides)
│   ├── main.dart               # App entry point
│   ├── models/                 # Data models (PredictionItem, ModelInfo)
│   ├── providers/              # PredictionProvider (ChangeNotifier)
│   ├── screens/                # HomeScreen, ResultScreen
│   ├── services/               # ImageService, ApiClient
│   └── widgets/                # DropZone, PredictionBars, ModelInfoCard, LoadingOverlay
├── test/                       # Flutter widget tests
├── .github/workflows/
│   ├── backend-ci.yml          # Python CI (ruff, black, pytest)
│   └── flutter-ci.yml          # Flutter CI (format, analyze, test)
└── README.md
```

## Backend Setup

### Prerequisites
- Python 3.11+
- pip
- (Optional) CUDA for GPU training

### Installation
```bash
cd backend
pip install -e .[dev]
```

### Training
```bash
# Train from scratch (CNN)
python scripts/train_scratch.py --epochs 20 --lr 1e-3 --batch-size 128

# Train with pretrained ResNet18
python scripts/train_pretrained.py --model resnet18 --epochs 10 --lr 1e-3
```

Checkpoints are saved to `backend/checkpoints/best_model.pt`.

### Running the API Server
```bash
uvicorn cnn_classifier.api.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/model-info` | Model metadata (name, input size, classes, device, trained_at) |
| POST | `/predict` | Classify image (multipart/form-data, max 10MB) |

#### Example: Predict
```bash
curl -X POST -F "file=@test_image.png" http://localhost:8000/predict
```

Response:
```json
{
  "predictions": [
    {"class_name": "cat", "confidence": 0.92},
    {"class_name": "dog", "confidence": 0.05},
    ...
  ],
  "latency_ms": 42.5,
  "model": "cnn_scratch"
}
```

### Export to ONNX (Optional)
```bash
python backend/scripts/export_onnx.py \
    backend/checkpoints/best_model.pt \
    model.onnx \
    --model-name cnn_scratch \
    --input-size 32 32 \
    --opset-version 17
```

## Flutter App Setup

### Prerequisites
- Flutter SDK 3.24+
- Linux desktop environment

### Configuration
The app uses `--dart-define` for configuration:
```bash
flutter run -d linux \
  --dart-define=API_BASE_URL=http://localhost:8000 \
  --dart-define=REQUEST_TIMEOUT_SEC=30
```

### Running the App
```bash
flutter pub get
flutter run -d linux
```

### Building Release Binary
```bash
flutter build linux --release
```
The binary will be at `build/linux/x64/release/bundle/image_classifier`.

## Model Details

| Model | Type | Parameters | CIFAR-10 Accuracy |
|-------|------|------------|-------------------|
| cnn_scratch | Custom CNN | ~1.2M | ~85% |
| resnet18 | Pretrained ResNet18 | ~11M | ~92% |
| vgg16 | Pretrained VGG16 | ~138M | ~93% |

### CIFAR-10 Classes
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

## API Usage Examples

### Health Check
```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### Model Info
```bash
curl http://localhost:8000/model-info
```

### Predict (curl)
```bash
curl -X POST -F "file=@cat.jpg" http://localhost:8000/predict
```

### Predict (Python)
```python
import requests

with open("cat.jpg", "rb") as f:
    resp = requests.post("http://localhost:8000/predict", files={"file": f})
    print(resp.json())
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: cnn_classifier` | Run `pip install -e .[dev]` from `backend/` |
| `Connection refused` | Ensure FastAPI server is running on port 8000 |
| `File too large` | Max upload size is 10MB (configurable via `--dart-define`) |
| `Model unavailable` | Ensure `backend/checkpoints/best_model.pt` exists |
| Flutter build fails | Run `flutter clean && flutter pub get` |

## Development

### Code Quality
```bash
# Backend
cd backend
ruff check src/ tests/
black --check src/ tests/
pytest tests/

# Flutter
flutter analyze
dart format --set-exit-if-changed lib test
flutter test
```

### CI/CD
GitHub Actions workflows:
- `.github/workflows/backend-ci.yml` - Python (ruff, black, pytest)
- `.github/workflows/flutter-ci.yml` - Flutter (format, analyze, test)

## License

MIT License - see LICENSE file for details.