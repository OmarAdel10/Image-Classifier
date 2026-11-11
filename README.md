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
│   │       └── training/       # Training loop, checkpointing
│   ├── scripts/
│   │   ├── fetch_cifar10.sh    # Verified CIFAR-10 download (used by CI)
│   │   ├── train_scratch.py    # Train from scratch
│   │   ├── train_transfer.py   # Fine-tune a pretrained model
│   │   └── export_onnx.py      # Export to ONNX
│   ├── tests/                  # pytest suite (402 tests)
│   └── pyproject.toml
├── lib/                        # Flutter app
│   ├── config.dart             # Config (--dart-define overrides)
│   ├── main.dart               # App entry point
│   ├── models/                 # Data models (PredictionItem, ModelInfo)
│   ├── providers/              # PredictionProvider (ChangeNotifier)
│   ├── screens/                # HomeScreen, ResultScreen, SettingsScreen
│   ├── services/               # ImageService, ApiClient
│   └── widgets/                # DropZone, PredictionBars, ModelInfoCard, LoadingOverlay
├── test/                       # Flutter widget tests
├── .github/workflows/
│   ├── backend-ci.yml          # Python CI (ruff, black, pytest)
│   ├── flutter-ci.yml          # Flutter CI (format, analyze, test) + gitleaks
│   ├── image-classifier-ci.yml # Flutter + Python backend CI
│   └── image-classifier-cd.yml # Release build (master only)
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
python scripts/train_scratch.py --epochs 20 --lr 1e-3 --model-name cnn_scratch

# Fine-tune a pretrained ResNet18
python scripts/train_transfer.py --model-name resnet18 --epochs 10 --lr 1e-4
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

#### API Base URL Resolution Order

The backend address is resolved in this order:

1. **Compile-time override** — `--dart-define=API_BASE_URL=...`. Always wins.
   Such a build is **pinned**: the settings screen disables **Save** and the
   app refuses to store a runtime override, so the address cannot change
   behind the flag.
2. **Runtime override** — set in the app via the ⚙ *Backend Settings* screen
   (gear icon in the app bar). Persisted with `shared_preferences`. Only a
   build without a compile-time override can store one.
3. **Platform default** — depends on the target:
   - Linux / Windows / macOS desktop → `http://localhost:8000`
   - Android emulator → `http://10.0.2.2:8000`
     (the emulator's `localhost` is the emulator itself, not your computer)

The settings screen has a **Test Connection** button that probes `/health`
against the typed URL before you save it, and reports the address it tried
when the connection fails. Any credentials in that address are removed
first, so the banner cannot repeat a password.

#### Reaching the Backend

| Where the app runs | URL to use | Backend must be started as |
|---|---|---|
| Linux/Windows/macOS desktop | `http://localhost:8000` | `uvicorn ... --host 0.0.0.0 --port 8000` |
| Android emulator | `http://10.0.2.2:8000` | `uvicorn ... --host 0.0.0.0 --port 8000` |
| Physical phone/tablet | `http://<your-LAN-IP>:8000` | `uvicorn ... --host 0.0.0.0 --port 8000` |
| Web browser | `http://<host>:8000` | `uvicorn ... --host 0.0.0.0 --port 8000` (CORS must allow the origin) |

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
| `Cannot reach http://...` | No response arrived. The address in the error is the one that failed: check the backend is running there, and that it is bound to `0.0.0.0` rather than `127.0.0.1` (required for emulator/device access). Open ⚙ *Backend Settings* → **Test Connection**. On Android use `http://10.0.2.2:8000`; on a physical device use your computer's LAN IP. |
| `Request to ... timed out after 15s` | The connection opened but no response arrived in time (`--dart-define=REQUEST_TIMEOUT_SEC`, default 15 s). Check the server log for a stalled request. |
| `Could not read the response from ...` | The status line arrived but the body did not: the server closed the connection early, stopped sending before the request timeout (`REQUEST_TIMEOUT_SEC`, default 15 s) expired, or the response exceeded the 10 MiB response budget. Check the server log. |
| `the file is larger than the 10485760 byte limit` | The app rejects images over 10 MiB before upload. Crop or re-encode the image. |
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
- `.github/workflows/flutter-ci.yml` - Flutter (format, analyze, test) + gitleaks
- `.github/workflows/image-classifier-ci.yml` - Flutter + Python backend CI
- `.github/workflows/image-classifier-cd.yml` - release build, `master` only

## License

MIT License - see LICENSE file for details.