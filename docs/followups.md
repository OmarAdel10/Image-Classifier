# Follow-ups

Out-of-scope findings discovered while working on other branches. Each entry has a
matching Plane work item.

## IC-67 — Missing trained CIFAR-10 checkpoint

- **Plane:** IC-67
- **Location:** `backend/checkpoints/best_model.pt` (missing), `backend/src/cnn_classifier/config.py:115`
- **Finding:** No model artifact exists in the repo. With the FastAPI backend running,
  `/health` returns `200` but `/model-info` and `/predict` return `503 model unavailable`,
  so classification cannot succeed until a checkpoint is produced.
- **Fix:** Run `python scripts/train_scratch.py --epochs 20 --lr 1e-3 --batch-size 128`
  from `backend/` (downloads CIFAR-10), or place a pre-trained checkpoint at
  `backend/checkpoints/best_model.pt`, then verify `GET /model-info` returns `200`.

## CI-TESTS — CI test step is non-blocking

- **Plane:** IC-73
- **Location:** `.github/workflows/image-classifier-ci.yml` (test step)
- **Finding:** The `flutter test` step is followed by `|| echo`, so a failing suite
  still reports CI green. Local `flutter test` is the only authoritative signal.
  Pre-existing issue, not introduced by any current branch.
- **Fix:** Remove the `|| echo` fallback so test failures fail the workflow.

## SETTINGS-DEAD-BRANCH — unreachable ApiException arm in _testConnection banner

- **Plane:** IC-72
- **Location:** `lib/screens/settings_screen.dart:108`
- **Finding:** `e is ApiException ? e.message : e` — the `ApiException` arm is dead:
  `healthStatusCode()` catches everything and returns null, so it never throws
  `ApiException` here.
- **Fix:** Simplify to `$e`, or wire the arm if `healthStatusCode` ever starts
  throwing `ApiException`.

## DOC-DRIFT — stale Config.apiBaseUrl references and empty CHANGELOG

- **Plane:** IC-75
- **Location:** `docs/superpowers/plans/2026-10-22-image-classification-cnn-flutter.md:569`,
  `CHANGELOG.md` (0 bytes), `lib/config.dart`
- **Finding:** The public `Config.apiBaseUrl` getter and `Config.effectiveApiBaseUrl()`
  were removed from `lib/config.dart` (no runtime caller remains — `lib/main.dart:16`
  only calls `Config.validateStartupApiBaseUrl()`), but the implementation plan still
  documents `Config.apiBaseUrl`, and `CHANGELOG.md` is empty, so the removal is not
  discoverable in either the plan or the changelog.
- **Fix:** Update the plan document to reference `Config.resolveApiBaseUrl()`, and
  add a `CHANGELOG.md` entry for the removed getters.
