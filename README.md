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

### Startup Flow (Auto-Connect)

```
flutter run -d linux
        │
        ▼
┌─────────────────────┐
│  BackendController  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     127.0.0.1:8000     ┌──────────────────┐
│  Managed Backend?   │ ─────────────────────► │  Backend Process │
│  (default: yes)     │     python -m uvicorn  │  (lifespan load) │
└─────────┬───────────┘     cnn_classifier.    │  /health → /ready│
          │             api.main:app            └──────────────────┘
          ▼
┌─────────────────────┐
│  Poll /health (30s) │──► 200 OK + service=cnn-classifier
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Poll /ready (60s)  │──► 200 OK + model_loaded=true
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Connect & Show UI  │
└─────────────────────┘
```

## Project Structure

```
.
├── backend/                    # FastAPI + PyTorch backend
│   ├── src/
│   │   └── cnn_classifier/
│   │       ├── api/            # FastAPI routes (health, ready, predict, model-info)
│   │       ├── config.py       # Configuration (env-configurable)
│   │       ├── inference/      # Checkpoint loading, preprocessing, prediction
│   │       ├── logging_config.py # Structured logging with request IDs
│   │       ├── models/         # Model registry (cnn_scratch, resnet18, vgg16)
│   │       └── training/       # Training loop, checkpointing
│   ├── scripts/
│   │   ├── fetch_cifar10.sh    # Verified CIFAR-10 download (used by CI)
│   │   ├── train_scratch.py    # Train from scratch
│   │   ├── train_transfer.py   # Fine-tune a pretrained model
│   │   ├── run_backend.py      # Backend launcher (stdin watchdog)
│   │   └── export_onnx.py      # Export to ONNX
│   ├── tests/                  # pytest suite (414 tests)
│   └── pyproject.toml
├── lib/                        # Flutter app
│   ├── config.dart             # Config (--dart-define overrides)
│   ├── main.dart               # App entry point
│   ├── models/                 # Data models (PredictionItem, ModelInfo, BackendStatus)
│   ├── providers/              # BackendController, PredictionProvider
│   ├── screens/                # HomeScreen, ResultScreen, SettingsScreen, StartupScreen
│   ├── services/               # ImageService, ApiClient, BackendLocator, BackendProcessManager
│   └── widgets/                # DropZone, PredictionBars, ModelInfoCard, LoadingOverlay
├── test/                       # Flutter widget tests (341 tests)
├── scripts/                    # Helper scripts
│   ├── setup_dev.sh            # Development environment setup
│   └── train_first_model.sh    # Train first model checkpoint
├── .github/workflows/
│   ├── backend-ci.yml          # Python CI (ruff, black, pytest)
│   ├── flutter-ci.yml          # Flutter CI (format, analyze, test) + gitleaks
│   ├── image-classifier-ci.yml # Flutter + Python backend CI
│   └── image-classifier-cd.yml # Release build (master only)
└── README.md
```

## Quick Start (First Working Version)

### Prerequisites
- Python 3.11+
- Flutter SDK 3.24+
- Linux desktop environment

### One-Time Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/OmarAdel10/Image-Classifier.git
cd Image-Classifier

# 2. Run the setup script (installs Python deps, creates venv, installs CUDA torch if GPU detected)
./scripts/setup_dev.sh

# 3. Train the first model checkpoint (resnet18 transfer, ~3-5 min on GPU, ~15-20 min on CPU)
./scripts/train_first_model.sh
# Or for from-scratch training: ./scripts/train_first_model.sh --model cnn_scratch --epochs 20
```

### Run the App

```bash
# From the repo root
flutter run -d linux
```

That's it! The app will:
1. Auto-start the FastAPI backend on `127.0.0.1:8000`
2. Wait for the model to load (`/ready` endpoint)
3. Connect and show the UI
4. Stop the backend when you close the app

No manual backend startup needed!

---

## Helper Scripts

| Script | Description |
|--------|-------------|
| `./scripts/setup_dev.sh` | Creates Python venv, installs deps, detects CUDA, installs CUDA torch if GPU present |
| `./scripts/train_first_model.sh` | Trains model and saves to canonical `backend/checkpoints/best_model.pt`. Options: `--model resnet18|cnn_scratch`, `--epochs N`, `--lr LR`, `--batch-size BS` |

---

## Backend Setup (Advanced)

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

Checkpoints are saved to `backend/checkpoints/best_model.pt` (canonical path).

### Running the API Server Manually
```bash
# Using the launcher (recommended - handles stdin watchdog, logging)
python scripts/run_backend.py

# Or directly with uvicorn
uvicorn cnn_classifier.api.main:app --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

### Configuration (Environment Variables)
```bash
# Server
CNN_PORT=8000
CNN_HOST=127.0.0.1

# Model paths (relative to project root)
CNN_MODEL_PATH=checkpoints/best_model.pt
CNN_LABELS_PATH=checkpoints/labels.json

# Device: cpu, cuda, mps, or auto (auto-detects best available)
CNN_DEVICE=auto

# Logging
CNN_LOG_LEVEL=INFO        # DEBUG, INFO, WARNING, ERROR, CRITICAL
CNN_LOG_JSON=false        # true for JSON format (CI/parsers)
```

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

The app uses `--dart-define` for compile-time configuration:
```bash
flutter run -d linux \
  --dart-define=API_BASE_URL=http://my-backend:8000 \
  --dart-define=REQUEST_TIMEOUT_SEC=30
```

#### Connection Modes

The app supports three connection modes (in priority order):

| Mode | Description | How to Enable |
|------|-------------|---------------|
| **Managed (default)** | App auto-starts backend on `127.0.0.1:8000`, waits for model load, connects automatically | No `--dart-define`, no saved URL in Settings |
| **Compile-time override** | Uses `--dart-define=API_BASE_URL=...` for a fixed external backend. Settings Save disabled. | Pass `--dart-define=API_BASE_URL=...` at build/run time |
| **Runtime override** | Set custom URL in ⚙ Settings screen. Persisted via `shared_preferences`. | Enter URL in Settings → Save |

The settings screen has a **Test Connection** button that probes `/health` against the typed URL before you save it, and reports the address it tried when the connection fails. Any credentials in that address are removed first, so the banner cannot repeat a password.

#### Managed Backend Details

- **Auto-start**: `python scripts/run_backend.py` (uses stdin watchdog — backend dies when app closes)
- **Health probe**: Polls `/health` every 250ms for up to 30s
- **Readiness probe**: Polls `/ready` every 250ms for up to 60s (waits for model load)
- **Service marker**: `/health` returns `{"service": "cnn-classifier"}` so app never attaches to wrong service on port 8000
- **Port fallback**: If 8000 busy, tries 8001-8010, then OS-assigned ephemeral port
- **Logs**: Backend stdout/stderr piped to app console with `[backend]` prefix

#### Running the App
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

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness probe (always 200 if process alive) |
| GET | `/ready` | Readiness probe (200 when model loaded, 503 otherwise) |
| GET | `/model-info` | Model metadata (name, input size, classes, device, trained_at) |
| POST | `/predict` | Classify image (multipart/form-data, max 10MB) |

### Health Check
```bash
curl http://127.0.0.1:8000/health
# {"status": "ok", "service": "cnn-classifier"}
```

### Readiness Check
```bash
curl http://127.0.0.1:8000/ready
# Model loaded:
# {"status": "ready", "service": "cnn-classifier", "model_loaded": true,
#  "model_path": "/path/to/checkpoints/best_model.pt", "device": "cuda", "classes": 10}
#
# Model not loaded:
# {"status": "not_ready", "service": "cnn-classifier", "model_loaded": false,
#  "error": "Checkpoint not found"}
```

### Model Info
```bash
curl http://127.0.0.1:8000/model-info
```

### Predict (curl)
```bash
curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/predict
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

### Predict (Python)
```python
import requests

with open("cat.jpg", "rb") as f:
    resp = requests.post("http://127.0.0.1:8000/predict", files={"file": f})
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
| **Backend fails to start** | Check console for `[backend]` prefixed logs. Common: Python not in PATH, missing deps (`./scripts/setup_dev.sh`), port conflict. |
| **Stuck on "Starting Backend"** | Port 8000 may be occupied. Check `lsof -i :8000`. The app tries 8001-8010 then falls back. |
| **Stuck on "Loading Model"** | Check `/ready` endpoint. If 503, model checkpoint missing: run `./scripts/train_first_model.sh`. |
| **Backend logs not showing** | Ensure `CNN_LOG_LEVEL` not set to `WARNING` or `ERROR`. Default is `INFO`. |
| **`Model unavailable` on /predict** | Checkpoint not found at `backend/checkpoints/best_model.pt`. Run `./scripts/train_first_model.sh`. |

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

# Helper scripts
./scripts/setup_dev.sh
./scripts/train_first_model.sh
```

### CI/CD
GitHub Actions workflows:
- `.github/workflows/backend-ci.yml` - Python (ruff, black, pytest)
- `.github/workflows/flutter-ci.yml` - Flutter (format, analyze, test) + gitleaks
- `.github/workflows/image-classifier-ci.yml` - Flutter + Python backend CI
- `.github/workflows/image-classifier-cd.yml` - release build, `master` only

## License

MIT License - see LICENSE file for details.