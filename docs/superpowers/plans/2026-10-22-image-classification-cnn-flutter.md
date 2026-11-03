# Image Classifier — CNN + Flutter GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:implement-review-ship (enforced) for every phase. Steps use checkbox (`- [ ]`) syntax for tracking.

**Project Name:** `Image Classifier`

**Goal:** Build an end-to-end image classification system: a from-scratch CNN trained on CIFAR-10, served via FastAPI (PyTorch), with a Flutter Linux desktop GUI for drag-and-drop inference and result visualization.

**Architecture:** Two-process architecture. (1) Python backend: FastAPI + PyTorch model (trainable, loadable, inferencable) exposing `/predict` and `/health` endpoints. (2) Flutter Linux desktop app: Material 3 UI, image picker (file dialog + drag-drop), displays top-k predictions with confidence bars, model metadata, and latency. Communication over localhost HTTP. Model checkpoint and class labels versioned together.

**Tech Stack:** Python 3.11+, PyTorch 2.x, torchvision, FastAPI, Uvicorn, Pydantic; Flutter 3.24+ (Linux desktop), Dart 3.5+, `image_picker`, `http`, `provider`, `fl_chart` for confidence bars.

**Workflow Enforcement:** This plan follows the `implement-review-ship` skill — every phase branches from `origin/development`, implements with TDD, passes FAST gate, then undergoes **mandatory fresh reviewer loop** (QA + coverage reviewers in parallel, zero Critical/High/Medium, score >90, zero coverage gaps) before PR creation. Use skills/subagents for maximum speed and correctness.

---

## Current State (Already Done)

- ✅ Git initialized, `development` branch created and pushed to remote
- ✅ Flutter project created at repository root (`/mnt/ALL/UniProjects/deep-learning/`)
- ✅ Backend folder created at `/mnt/ALL/UniProjects/deep-learning/backend/`
- ✅ `CHANGELOG.md` created at root
- ✅ `backend/pyproject.toml` created (base dependencies)
- ✅ All committed and pushed to remote

---

## Global Constraints

- Python ≥3.11, PyTorch ≥2.3, Flutter ≥3.24 (Linux desktop stable)
- Model input: 32×32×3 (CIFAR-10) or 224×224×3 (transfer learning); normalize with ImageNet stats for pretrained, CIFAR stats for scratch
- Class labels: 10 fixed classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) — stored in `labels.json` alongside checkpoint
- Backend port: 8000 (configurable via env `PORT`); Flutter targets `http://localhost:8000`
- No external cloud dependencies; fully offline after initial `pip install` / `flutter pub get`
- Deterministic training: seed 42 for torch, numpy, python random
- Code style: `black`, `ruff` (Python); `dart format`, `flutter analyze` (Flutter)
- Tests: pytest (backend), `flutter test` (Flutter); CI on every branch
- **Every commit via `emoji-commit` skill; FAST gate before commit (format, analyze, no debug prints, no secrets)**

---

## Review Focus

| Input / Condition | Expected Behavior | Test Location |
|-------------------|-------------------|---------------|
| Corrupt / non-image file dropped | Graceful error toast, no crash | `test_widgets_drop_invalid.dart` |
| Model file missing at startup | Clear error banner, retry button | `test_backend_startup_missing_model.py` |
| Image wrong size / channels | Auto-resize + center-crop to model input; log warning | `test_preprocess_resize.py` |
| Backend not running when Flutter starts | Friendly connection-error screen with "Retry" | `test_widgets_backend_down.dart` |
| GPU OOM on large batch (transfer learning) | Fallback to CPU with warning, not crash | `test_train_oom_fallback.py` |

---

## File Structure (Target State)

```
/mnt/ALL/UniProjects/deep-learning/
├── backend/
│   ├── pyproject.toml              ✅ exists
│   ├── .env.example
│   ├── src/
│   │   └── cnn_classifier/
│   │       ├── __init__.py
│   │       ├── config.py
│   │       ├── data/
│   │       │   ├── __init__.py
│   │       │   ├── dataset.py
│   │       │   └── custom_dataset.py
│   │       ├── models/
│   │       │   ├── __init__.py
│   │       │   ├── cnn_scratch.py
│   │       │   ├── transfer.py
│   │       │   └── registry.py
│   │       ├── training/
│   │       │   ├── __init__.py
│   │       │   ├── train.py
│   │       │   └── evaluate.py
│   │       ├── inference/
│   │       │   ├── __init__.py
│   │       │   ├── predictor.py
│   │       │   └── preprocess.py
│   │       └── api/
│   │       │   ├── __init__.py
│   │       │   ├── main.py
│   │       │   ├── schemas.py
│   │       │   └── dependencies.py
│   ├── tests/
│   │   ├── conftest.py
│   │   ├── test_config.py
│   │   ├── test_dataset.py
│   │   ├── test_preprocess.py
│   │   ├── test_model_forward.py
│   │   ├── test_api_predict.py
│   │   └── test_train_smoke.py
│   ├── scripts/
│   │   ├── train_scratch.py
│   │   ├── train_transfer.py
│   │   └── export_onnx.py
│   ├── checkpoints/
│   │   ├── best_model.pt
│   │   └── labels.json
│   └── Dockerfile
├── pubspec.yaml                    ✅ exists (Flutter project)
├── CHANGELOG.md                    ✅ exists
├── lib/
│   ├── main.dart
│   ├── config.dart
│   ├── models/
│   │   ├── prediction.dart
│   │   └── model_info.dart
│   ├── services/
│   │   ├── api_client.dart
│   │   └── image_service.dart
│   ├── providers/
│   │   └── prediction_provider.dart
│   ├── screens/
│   │   ├── home_screen.dart
│   │   └── result_screen.dart
│   ├── widgets/
│   │   ├── drop_zone.dart
│   │   ├── prediction_bars.dart
│   │   ├── model_info_card.dart
│   │   └── loading_overlay.dart
│   └── utils/
│       └── formatters.dart
├── test/
│   ├── widget_test.dart
│   ├── api_client_test.dart
│   └── image_service_test.dart
├── linux/
├── .github/workflows/
│   ├── backend-ci.yml
│   └── flutter-ci.yml
└── README.md
```

---

## Task Breakdown

### Task 1: Backend Config & Scaffold Completion

**Status:** `backend/pyproject.toml` exists — complete missing pieces.

**Files:**
- Create: `backend/.env.example`
- Create: `backend/src/cnn_classifier/__init__.py`
- Create: `backend/src/cnn_classifier/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `Config` class (Pydantic `BaseSettings`) with fields: `port`, `model_path`, `labels_path`, `device`, `input_size`, `batch_size`, `num_workers`, `seed`

- [ ] **Step 1: Write failing test for config loading**

```python
# backend/tests/test_config.py
def test_config_loads_defaults(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "checkpoints/best_model.pt")
    monkeypatch.setenv("LABELS_PATH", "checkpoints/labels.json")
    from cnn_classifier.config import Config
    cfg = Config()
    assert cfg.port == 8000
    assert cfg.device in ("cpu", "cuda", "mps")
    assert cfg.input_size == (32, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_config.py::test_config_loads_defaults -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement `config.py` and `.env.example`**

`config.py`: Pydantic `BaseSettings` with `model_config = SettingsConfigDict(env_file=".env", extra="ignore")`. Add `class_names` property returning the 10 CIFAR-10 classes.

`.env.example`: `MODEL_PATH=checkpoints/best_model.pt`, `LABELS_PATH=checkpoints/labels.json`, `PORT=8000`, `DEVICE=cpu`, `INPUT_SIZE=32,32`, `BATCH_SIZE=64`, `NUM_WORKERS=4`, `SEED=42`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_config.py::test_config_loads_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/.env.example backend/src/cnn_classifier/config.py backend/tests/test_config.py
git commit -m "feat(backend): config management with Pydantic settings"
```

---

### Task 2: Data Module — CIFAR-10 Loaders & Transforms

**Files:**
- Create: `backend/src/cnn_classifier/data/__init__.py`
- Create: `backend/src/cnn_classifier/data/dataset.py`
- Create: `backend/src/cnn_classifier/data/custom_dataset.py`
- Test: `backend/tests/test_dataset.py`

**Interfaces:**
- Consumes: `Config` (input_size, batch_size, num_workers, seed)
- Produces: `get_dataloaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader]` (train, val, test); `CIFAR10_MEAN`, `CIFAR10_STD` constants; `CustomImageFolder` class for folder-based datasets

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_dataset.py
def test_cifar10_loaders_shapes():
    from cnn_classifier.data.dataset import get_dataloaders
    from cnn_classifier.config import Config
    cfg = Config(batch_size=4, num_workers=0)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32)
    assert y.shape == (4,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long

def test_custom_dataset_loads_folder(tmp_path):
    from cnn_classifier.data.custom_dataset import CustomImageFolder
    # create fake class folders with dummy images
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `dataset.py` and `custom_dataset.py`**

`dataset.py`: `get_dataloaders` downloads CIFAR-10 via `torchvision.datasets.CIFAR10`, applies transforms: `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, `ToTensor`, `Normalize(CIFAR10_MEAN, CIFAR10_STD)` for train; `ToTensor`, `Normalize` for val/test. Returns `(train_loader, val_loader, test_loader)` with stratified 90/10 train/val split.

`custom_dataset.py`: `CustomImageFolder(root, transform)` wraps `torchvision.datasets.ImageFolder` for user-provided folder structure (`root/class_name/*.jpg`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cnn_classifier/data/ backend/tests/test_dataset.py
git commit -m "feat(backend): CIFAR-10 loaders with augmentation and custom dataset support"
```

---

### Task 3: Model Definitions — Scratch CNN & Transfer Learning

**Files:**
- Create: `backend/src/cnn_classifier/models/__init__.py`
- Create: `backend/src/cnn_classifier/models/cnn_scratch.py`
- Create: `backend/src/cnn_classifier/models/transfer.py`
- Create: `backend/src/cnn_classifier/models/registry.py`
- Test: `backend/tests/test_model_forward.py`

**Interfaces:**
- Consumes: `Config` (input_size, num_classes=10, device)
- Produces: `CNNScratch` class (from-scratch CNN architecture), `TransferModel` class (ResNet18/VGG16 with replaced head), `get_model(name: str, config: Config) -> nn.Module` factory

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_model_forward.py
import torch
from cnn_classifier.models.registry import get_model
from cnn_classifier.config import Config

def test_cnn_scratch_forward():
    cfg = Config(input_size=(32, 32), device="cpu")
    model = get_model("cnn_scratch", cfg)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (2, 10)
    assert not torch.isnan(logits).any()

def test_transfer_resnet18_forward():
    cfg = Config(input_size=(224, 224), device="cpu")
    model = get_model("resnet18", cfg)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_model_forward.py -v`
Expected: FAIL

- [ ] **Step 3: Implement model files**

`cnn_scratch.py`: From-scratch CNN — Conv(3→32, 3×3) → ReLU → MaxPool(2) → Conv(32→64, 3×3) → ReLU → MaxPool(2) → Flatten → FC(64×8×8→512) → ReLU → Dropout(0.5) → FC(512→10). Weight init: Kaiming normal for conv/linear.

`transfer.py`: `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`, replace `fc` with `nn.Linear(512, num_classes)`. Freeze backbone by default; `unfreeze_backbone()` method. Same for VGG16.

`registry.py`: `MODEL_REGISTRY = {"cnn_scratch": CNNScratch, "resnet18": lambda c: TransferModel("resnet18", c), "vgg16": lambda c: TransferModel("vgg16", c)}`; `get_model` instantiates.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_model_forward.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cnn_classifier/models/ backend/tests/test_model_forward.py
git commit -m "feat(backend): CNN from scratch + transfer learning models (ResNet18, VGG16)"
```

---

### Task 4: Training Loop & Evaluation

**Files:**
- Create: `backend/src/cnn_classifier/training/__init__.py`
- Create: `backend/src/cnn_classifier/training/train.py`
- Create: `backend/src/cnn_classifier/training/evaluate.py`
- Create: `backend/scripts/train_scratch.py`
- Create: `backend/scripts/train_transfer.py`
- Test: `backend/tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `Config`, `get_dataloaders`, `get_model`, `nn.CrossEntropyLoss`, `torch.optim.AdamW`
- Produces: `train_one_epoch(model, loader, optimizer, criterion, device) -> dict` (loss, acc); `validate(model, loader, criterion, device) -> dict`; `evaluate(model, test_loader, device, class_names) -> dict` (acc, per-class metrics, confusion matrix); checkpoint dict with `model_state`, `optimizer_state`, `epoch`, `best_val_acc`, `config`, `class_names`

- [ ] **Step 1: Write failing smoke test**

```python
# backend/tests/test_train_smoke.py
def test_train_one_epoch_runs():
    from cnn_classifier.training.train import train_one_epoch
    from cnn_classifier.models.registry import get_model
    from cnn_classifier.config import Config
    import torch
    cfg = Config(device="cpu", batch_size=2, num_workers=0)
    model = get_model("cnn_scratch", cfg)
    loader = [(torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    metrics = train_one_epoch(model, loader, opt, crit, "cpu")
    assert "loss" in metrics and "acc" in metrics
    assert metrics["loss"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_train_smoke.py::test_train_one_epoch_runs -v`
Expected: FAIL

- [ ] **Step 3: Implement training & evaluation**

`train.py`: Standard loop with `model.train()`, gradient clipping (max_norm=1.0), cosine annealing LR scheduler, early stopping (patience=10), checkpoint save on best val acc. Log to console + optional TensorBoard.

`evaluate.py`: `model.eval()`, compute top-1 acc, per-class precision/recall/F1, confusion matrix (sklearn or manual). Save `classification_report.json`.

`train_scratch.py`: CLI entry — `argparse` for epochs, lr, model name; loads config, data, model, trains, saves `checkpoints/best_model.pt` + `labels.json`.

`train_transfer.py`: Same but default model=resnet18, input_size=224, smaller lr (1e-4), freeze backbone initially.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_train_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cnn_classifier/training/ backend/scripts/train_*.py backend/tests/test_train_smoke.py
git commit -m "feat(backend): training loop, evaluation, and CLI entry points"
```

---

### Task 5: Inference Module & Preprocessing

**Files:**
- Create: `backend/src/cnn_classifier/inference/__init__.py`
- Create: `backend/src/cnn_classifier/inference/preprocess.py`
- Create: `backend/src/cnn_classifier/inference/predictor.py`
- Test: `backend/tests/test_preprocess.py`, `backend/tests/test_predictor.py`

**Interfaces:**
- Consumes: `Config` (input_size, device, model_path, labels_path), PIL Image or bytes
- Produces: `preprocess_image(image: Image.Image, input_size: tuple) -> torch.Tensor` (1, C, H, W); `Predictor` class with `predict(tensor: Tensor) -> list[Prediction]` where `Prediction = namedtuple("Prediction", "class_name confidence")`; `Predictor.from_checkpoint(path: str, config: Config) -> Predictor`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_preprocess.py
from PIL import Image
from cnn_classifier.inference.preprocess import preprocess_image

def test_preprocess_resizes_and_normalizes():
    img = Image.new("RGB", (100, 100), color="red")
    tensor = preprocess_image(img, (32, 32))
    assert tensor.shape == (1, 3, 32, 32)
    assert tensor.min() >= -3 and tensor.max() <= 3

# backend/tests/test_predictor.py
def test_predictor_loads_and_predicts(tmp_path):
    from cnn_classifier.inference.predictor import Predictor
    from cnn_classifier.config import Config
    import torch
    cfg = Config(device="cpu", input_size=(32, 32))
    model = get_model("cnn_scratch", cfg)
    ckpt = {"model_state": model.state_dict(), "class_names": cfg.class_names, "config": cfg.model_dump()}
    torch.save(ckpt, tmp_path / "model.pt")
    pred = Predictor.from_checkpoint(str(tmp_path / "model.pt"), cfg)
    x = torch.randn(1, 3, 32, 32)
    results = pred.predict(x)
    assert len(results) == 10
    assert all(0 <= r.confidence <= 1 for r in results)
    assert abs(sum(r.confidence for r in results) - 1.0) < 1e-4
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_preprocess.py tests/test_predictor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement preprocessing & predictor**

`preprocess.py`: `Resize(input_size)`, `CenterCrop(input_size)`, `ToTensor()`, `Normalize(mean, std)` — mean/std from config (CIFAR or ImageNet). Handles RGBA → RGB conversion.

`predictor.py`: Loads checkpoint, extracts `class_names`, instantiates model via registry, loads `model_state`, `model.eval()`, `torch.no_grad()`, `softmax` on logits, returns sorted `Prediction` list.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_preprocess.py tests/test_predictor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cnn_classifier/inference/ backend/tests/test_preprocess.py backend/tests/test_predictor.py
git commit -m "feat(backend): inference pipeline with preprocessing and predictor"
```

---

### Task 6: FastAPI Backend — Routes & Schemas

**Files:**
- Create: `backend/src/cnn_classifier/api/__init__.py`
- Create: `backend/src/cnn_classifier/api/schemas.py`
- Create: `backend/src/cnn_classifier/api/dependencies.py`
- Create: `backend/src/cnn_classifier/api/main.py`
- Test: `backend/tests/test_api_predict.py`

**Interfaces:**
- Consumes: `Predictor`, `Config`, `preprocess_image`
- Produces: FastAPI `app` with routes: `GET /health`, `POST /predict` (multipart file → `PredictionResponse`), `GET /model-info`; Pydantic models: `PredictionItem(class_name: str, confidence: float)`, `PredictionResponse(predictions: list[PredictionItem], latency_ms: float, model: str)`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_api_predict.py
from fastapi.testclient import TestClient
from cnn_classifier.api.main import app
from PIL import Image
import io

client = TestClient(app)

def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_endpoint_returns_topk(monkeypatch):
    # mock predictor dependency
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_api_predict.py -v`
Expected: FAIL

- [ ] **Step 3: Implement API layer**

`schemas.py`: Pydantic models as above.

`dependencies.py`: `get_predictor()` — singleton `Predictor` loaded at startup via `@lru_cache`; `get_config()`.

`main.py`: `FastAPI(title="Image Classifier API")`, CORS middleware (allow localhost), routes with dependency injection. `/predict` accepts `UploadFile`, validates content-type (`image/*`), reads bytes, `PIL.Image.open(io.BytesIO(...))`, preprocess, predict, return JSON. Error handling: 400 for bad file, 500 for model errors with message.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_api_predict.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cnn_classifier/api/ backend/tests/test_api_predict.py
git commit -m "feat(backend): FastAPI routes for prediction, health, and model info"
```

---

### Task 7: Flutter App — Scaffold Completion

**Status:** Flutter project exists at root with `pubspec.yaml` — complete missing pieces.

**Files:**
- Modify: `pubspec.yaml` (add dependencies)
- Create: `lib/main.dart`
- Create: `lib/config.dart`
- Test: `test/widget_test.dart`

**Interfaces:**
- Produces: Flutter app entry point, `Config` class with `apiBaseUrl`, `requestTimeout`, `maxImageSizeBytes`

- [ ] **Step 1: Write failing widget test**

```dart
// test/widget_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:image_classifier/main.dart';

void main() {
  testWidgets('App loads home screen', (WidgetTester tester) async {
    await tester.pumpWidget(const MyApp());
    expect(find.text('Image Classifier'), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `flutter test test/widget_test.dart`
Expected: FAIL

- [ ] **Step 3: Implement scaffold**

`pubspec.yaml`: Add dependencies: `http`, `provider`, `image_picker`, `file_picker`, `fl_chart`, `path`, `mime`.

`config.dart`: `class Config { static const apiBaseUrl = 'http://localhost:8000'; static const requestTimeout = Duration(seconds: 30); static const maxImageSizeBytes = 10 * 1024 * 1024; }`

`main.dart`: `MaterialApp` with `home: HomeScreen()`, theme: Material 3, dark/light support, app title "Image Classifier".

- [ ] **Step 4: Run test to verify it passes**

Run: `flutter test test/widget_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pubspec.yaml lib/main.dart lib/config.dart test/widget_test.dart
git commit -m "feat(flutter): scaffold app with config and Material 3 theme"
```

---

### Task 8: Flutter Models & API Client

**Files:**
- Create: `lib/models/prediction.dart`
- Create: `lib/models/model_info.dart`
- Create: `lib/services/api_client.dart`
- Test: `test/api_client_test.dart`

**Interfaces:**
- Consumes: `Config.apiBaseUrl`, `http.Client`
- Produces: `PredictionItem`, `PredictionResponse`, `ModelInfo`; `ApiClient` with `Future<PredictionResponse> predict(XFile image)`, `Future<ModelInfo> getModelInfo()`, `Future<bool> healthCheck()`

- [ ] **Step 1: Write failing tests**

```dart
// test/api_client_test.dart
import 'package:http/testing.dart';
import 'package:image_classifier/services/api_client.dart';

void main() {
  test('predict returns parsed response', () async {
    final client = MockClient((request) async => Response(jsonEncode({
      'predictions': [{'class_name': 'cat', 'confidence': 0.9}],
      'latency_ms': 42.0,
      'model': 'cnn_scratch'
    }), 200));
    final api = ApiClient(client: client);
    final result = await api.predict(mockXFile);
    expect(result.predictions.length, 1);
    expect(result.predictions.first.className, 'cat');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `flutter test test/api_client_test.dart`
Expected: FAIL

- [ ] **Step 3: Implement models & client**

`prediction.dart`: Manual `fromJson`/`toJson` for `PredictionItem`, `PredictionResponse`.

`model_info.dart`: `ModelInfo(String modelName, String inputSize, List<String> classes, DateTime? trainedAt)`.

`api_client.dart`: `multipartRequest` for `/predict` with image file; JSON parsing; timeout handling; maps HTTP errors to custom exceptions.

- [ ] **Step 4: Run test to verify it passes**

Run: `flutter test test/api_client_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/models/ lib/services/api_client.dart test/api_client_test.dart
git commit -m "feat(flutter): data models and API client with error handling"
```

---

### Task 9: Flutter Image Service & Provider

**Files:**
- Create: `lib/services/image_service.dart`
- Create: `lib/providers/prediction_provider.dart`
- Test: `test/image_service_test.dart`, `test/prediction_provider_test.dart`

**Interfaces:**
- Consumes: `ApiClient`, `image_picker`, `file_picker`
- Produces: `ImageService.pickImage() -> Future<XFile?>`, `ImageService.validateAndCompress(XFile) -> Future<XFile>`; `PredictionProvider` (ChangeNotifier) with `predict(XFile)`, `state` (idle/loading/result/error), `clear()`

- [ ] **Step 1: Write failing tests**

```dart
// test/image_service_test.dart
test('compress resizes large image under limit', () async {
  final service = ImageService();
  final largeFile = createLargeTestImage(20 * 1024 * 1024);
  final compressed = await service.validateAndCompress(largeFile);
  expect(await compressed.length(), lessThan(10 * 1024 * 1024));
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `flutter test test/image_service_test.dart test/prediction_provider_test.dart`
Expected: FAIL

- [ ] **Step 3: Implement services & provider**

`image_service.dart`: `pickImage()` uses `ImagePicker` (gallery) + `FilePicker` (drag-drop fallback). `validateAndCompress` checks file size, MIME type, decodes image, resizes if > `maxImageSizeBytes` (max dimension 1024), re-encodes JPEG 85%.

`prediction_provider.dart`: `ChangeNotifier` with `PredictionState { idle, loading, result(PredictionResponse), error(String) }`. `predict(file)` calls `ApiClient.predict`, updates state, notifies listeners.

- [ ] **Step 4: Run tests to verify they pass**

Run: `flutter test test/image_service_test.dart test/prediction_provider_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/services/image_service.dart lib/providers/prediction_provider.dart test/image_service_test.dart test/prediction_provider_test.dart
git commit -m "feat(flutter): image picking/compression and prediction state management"
```

---

### Task 10: Flutter UI — Home Screen, Drop Zone, Results

**Files:**
- Create: `lib/widgets/drop_zone.dart`
- Create: `lib/widgets/prediction_bars.dart`
- Create: `lib/widgets/model_info_card.dart`
- Create: `lib/widgets/loading_overlay.dart`
- Create: `lib/screens/home_screen.dart`
- Create: `lib/screens/result_screen.dart`
- Test: `test/widget_test.dart` (extend)

**Interfaces:**
- Consumes: `PredictionProvider`, `ImageService`, `ApiClient`
- Produces: Composable widgets; `HomeScreen` orchestrates flow

- [ ] **Step 1: Write failing widget tests**

```dart
// test/widget_test.dart (additions)
testWidgets('Drop zone shows file picker on tap', (tester) async {
  await tester.pumpWidget(ProviderScope(overrides: [imageServiceProvider.overrideWithValue(mockImageService)], child: MyApp()));
  await tester.tap(find.byKey(Key('drop_zone')));
  // verify file picker invoked
});

testWidgets('Prediction bars render top-5 with correct widths', (tester) async {
  // pump PredictionBars with mock data
  // verify bar widths proportional to confidence
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `flutter test test/widget_test.dart`
Expected: FAIL

- [ ] **Step 3: Implement UI widgets**

`drop_zone.dart`: `DragTarget` + `GestureDetector` for file dialog; visual feedback (border highlight on hover/drag); shows selected image preview thumbnail.

`prediction_bars.dart`: Horizontal bar chart using `fl_chart` `BarChart` — top-5 classes, confidence 0–1, colored by confidence (green→yellow→red), labels with class name + percentage.

`model_info_card.dart`: Displays model name, input resolution, number of classes, last trained timestamp (from `/model-info`).

`loading_overlay.dart`: Full-screen semi-transparent overlay with `CircularProgressIndicator` and "Classifying..." text.

`home_screen.dart`: `Consumer<PredictionProvider>`; shows drop zone + preview; on image selected → `provider.predict()`; on result → `Navigator.push(ResultScreen)`.

`result_screen.dart`: Large image preview + `PredictionBars` + `ModelInfoCard` + "Classify Another" button.

- [ ] **Step 4: Run tests to verify they pass**

Run: `flutter test test/widget_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/widgets/ lib/screens/ test/widget_test.dart
git commit -m "feat(flutter): complete UI — drop zone, prediction bars, result screen, model info"
```

---

### Task 11: End-to-End Integration & Polish

**Files:**
- Modify: `backend/src/cnn_classifier/api/main.py` (add CORS for Flutter origin)
- Modify: `lib/config.dart` (allow override via `--dart-define`)
- Create: `backend/scripts/export_onnx.py` (optional)
- Create: `.github/workflows/backend-ci.yml`
- Create: `.github/workflows/flutter-ci.yml`
- Create: `README.md`
- Test: Full backend pytest suite, full flutter test suite, manual E2E

**Interfaces:**
- Backend serves on `0.0.0.0:8000`; Flutter builds Linux release binary

- [ ] **Step 1: Configure CORS & env override**

`main.py`: `app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:*", "http://127.0.0.1:*"], ...)`

`config.dart`: `const apiBaseUrl = String.fromEnvironment('API_BASE_URL', defaultValue: 'http://localhost:8000');`

- [ ] **Step 2: Write CI workflows**

`backend-ci.yml`: `ubuntu-latest`, `python: 3.11`, `pip install -e .[dev]`, `ruff check .`, `black --check .`, `pytest -v`.

`flutter-ci.yml`: `ubuntu-latest`, `flutter: 3.24`, `flutter pub get`, `dart format --set-exit-if-changed .`, `flutter analyze`, `flutter test`, `flutter build linux --release`.

- [ ] **Step 3: Train a baseline model & verify E2E**

```bash
cd backend && pip install -e .
python scripts/train_scratch.py --epochs 20 --lr 1e-3
# verify checkpoints/best_model.pt exists
uvicorn cnn_classifier.api.main:app --host 0.0.0.0 --port 8000 &
cd .. && flutter run -d linux
# drag a test image → verify predictions appear
```

- [ ] **Step 4: Write README**

Sections: Project overview, Architecture diagram (text), Backend setup (venv, train, run), Flutter setup (run, build), API usage (curl example), Model details, Troubleshooting.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/ backend/scripts/export_onnx.py README.md
git commit -m "feat: CI pipelines, ONNX export script, README, E2E verified"
```

---

## Execution Protocol (Enforced by implement-review-ship)

**For EACH task above (or logical phase grouping):**

1. **Branch:** `git checkout -b feature/task-N-name origin/development`
2. **Implement:** Follow task steps (write test → run fail → implement → run pass → commit via `emoji-commit`)
3. **FAST gate:** `dart format --set-exit-if-changed lib test` + `flutter analyze` + `flutter test` (or `pytest` for backend) before every commit
4. **Review loop:** Dispatch **both** fresh reviewers in parallel:
   - `agent: qa-post-implementation` (runs `qa-post-implementation` skill on branch diff)
   - `agent: general` (runs `test-coverage-analysis` skill on branch diff)
5. **Gate:** Zero Critical/High/Medium, score >90, zero coverage gaps — else fix round with **brand-new reviewers** (max 3 rounds)
6. **PR:** `gh pr create --base development` with final scores, deferred Low/Info notes, coverage summary
7. **Repeat** for next task

**Subagent usage:** Use `subagent` tool with `agent: "qa-post-implementation"` and `agent: "general"` for reviews. Use specialized agents (`engineering-senior-developer`, `engineering-minimal-change-engineer`, etc.) for implementation when beneficial.

---

**Plan saved to `/home/omaradel/.opencode/plan/2026-10-22-image-classification-cnn-flutter.md`. Ready to begin with Task 1.**