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
- **Location:** `lib/screens/settings_screen.dart:109`
- **Finding:** `e is ApiException ? e.message : e` — the `ApiException` arm is dead:
  `healthStatusCode()` catches everything and returns null, so it never throws
  `ApiException` here.
- **Fix:** Simplify to `$e`, or wire the arm if `healthStatusCode` ever starts
  throwing `ApiException`.

## ERROR-BODY — _parseErrorMessage reports the raw body, not the status code

- **Plane:** IC-68
- **Location:** `lib/services/api_client.dart:260`
- **Finding:** The backend error body is shown verbatim to the user instead of the
  HTTP status code, so a proxy or HTML error page leaks through the message.
- **Fix:** Report the response status code and keep the body only for structured
  JSON error payloads.

## CI-DEDUP — duplicated Flutter CI workflows with divergent torch cache keys

- **Plane:** IC-69
- **Location:** `.github/workflows/backend-ci.yml:86`,
  `.github/workflows/image-classifier-ci.yml:124`
- **Finding:** `backend-ci.yml` keys the torch weight cache on
  `torch-pretrained-tv-0.29.1-sha256-<cifar sha256>` while
  `image-classifier-ci.yml` uses the bare key `torch-pretrained-tv-0.29.1`, so
  the two workflows cache different weights under different keys. Neither
  workflow integrity-checks the cached weights: the `Verify torch cache
  integrity` step in `backend-ci.yml` only echoes two strings, and the weights
  are loaded through `torch.load(path, weights_only=True)`
  (`backend/src/cnn_classifier/inference/checkpoints.py:107`) and
  `resnet18/vgg16(weights=...)`
  (`backend/src/cnn_classifier/models/transfer.py:53,58`).
- **Fix:** Merge the two workflows into one, or give both the same key and
  verify the cached weights in both.

## TORCH-CACHE-VERIFY — torch cache integrity step verifies nothing

- **Plane:** IC-79
- **Location:** `.github/workflows/backend-ci.yml:89-96`
- **Finding:** The `Verify torch cache integrity` step only echoes two strings;
  it never hashes or compares the cached weights. Its comments claim the
  weights are loaded with `torch.hub.load_state_dict_from_url(check_hash=True)`
  and that the cache key "transitively verifies the dataset integrity" —
  neither is true: no `check_hash`, `torch.hub` or `load_state_dict_from_url`
  call exists in the repository, and the key embeds the CIFAR-10 *dataset*
  hash, not a digest of the weights.
- **Fix:** Replace the echo with a real check (hash the cached files and
  compare them against a pinned digest), or delete the step together with its
  misleading comments.

## FLAKY-HOME — flaky home_screen snackbar test

- **Plane:** IC-71
- **Location:** `test/home_screen_test.dart:266`
- **Finding:** The `prediction errors offer a Settings action that navigates`
  test was observed to fail intermittently: the Settings-action SnackBar is
  queued from an async error path, so the widget lookup can run before the
  snackbar is mounted.
- **Fix:** Synchronise the test with the snackbar queue instead of relying on
  the current pump timing.

## CI-CIFAR-POISON — Backend CI CIFAR-10 cache poisoning

- **Plane:** IC-74
- **Location:** `.github/workflows/backend-ci.yml:48-80`
- **Finding:** The failed job re-saved a truncated archive under the same
  immutable cache key, so every later run kept failing on its own poisoned
  cache entry.
- **Fix:** **Delivered by PR #30** — split `actions/cache` into `restore` + a
  hash-gated `save`, version the key, and download through
  `backend/scripts/fetch_cifar10.sh`.

## CI-CACHE-KEY — one shared CIFAR cache key, immutable v2 key, unpinned capture

- **Plane:** IC-76
- **Location:** `.github/workflows/backend-ci.yml:76-80`,
  `.github/workflows/image-classifier-ci.yml:114-118`
- **Finding:** (1) Both workflows save the same key on the same ref, so the
  second one hits a reservation race and re-downloads 162 MB; (2) the `v2` key
  is immutable, so a future corruption can never be repaired in place; (3) the
  mirror selector `web/2020id_/` is a closest-capture selector rather than a
  pinned capture id.
- **Fix:** Use one cache key per workflow, allow the save step to write a
  repair key, and pin the archive.org capture id.

## API-MESSAGES — misleading exception messages

- **Plane:** IC-77
- **Location:** `lib/services/api_client.dart:214`,
  `lib/services/api_client.dart:226`
- **Finding:** A `Uri.parse` `FormatException` is reported as
  `Invalid response from ...` although it is a request-construction error, a
  `MultipartFile.fromPath` `ArgumentError` is reported as
  `Invalid backend address ...` although the file is missing, and
  `${e.message}` can render `null`.
- **Fix:** Map each exception to the message that describes its real cause.
- **Note:** The port bound at `lib/config.dart:201` (`uri.port > 65535`) is
  load-bearing and must stay: Dart's `Uri` parser does not range-check ports,
  so `http://example.com:99999` parses successfully and only this check
  rejects it.

## README-PINNED — stale pinned-build documentation

- **Plane:** IC-78
- **Location:** `README.md:141-143`, `docs/followups.md` (SETTINGS-DEAD-BRANCH)
- **Finding:** The README still says the runtime override is ignored when a
  compile-time override is present, but a pinned build can no longer create one
  (Save is disabled and `Config` throws), and the `SETTINGS-DEAD-BRANCH` entry
  still describes the old `healthStatusCode` contract.
- **Fix:** Document the pinned-build behaviour and refresh the stale rationale.

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

## STORAGE-REFUSAL — shared_preferences caches a refused write

- **Plane:** IC-81
- **Location:** `lib/config.dart:137` (`setApiBaseUrl`),
  `lib/config.dart:159` (`_rollbackStoredUrl`),
  `lib/config.dart:178` (`clearApiBaseUrl`),
  `lib/screens/settings_screen.dart:187` (`_resetToDefault`)
- **Finding:** `shared_preferences` 2.5.6 updates its in-memory cache *before* the
  platform call: `_setValue` writes `_preferenceCache[key]` and `remove` deletes
  `_preferenceCache[key]` before the platform answers. A refused write therefore
  left the rejected URL effective for the whole session while the UI reported
  `Could not save`, and a refused removal left the override inactive while the
  UI claimed it was "still in use". The platform may also restore a refused
  removal after a restart, which Dart cannot prevent.
- **Fix:** **Delivered by PR #30** — a refused write now reloads the persisted
  state (`Config._rollbackStoredUrl`), so the app keeps using the address that
  is actually stored instead of the rejected one or the platform default, and
  the refusal is reported with its cause. A refused removal is reported as "not
  active now, but it may return after a restart". The remaining platform-level
  gap (confirming a removal by re-reading the persisted value) is tracked in
  IC-81.

## DEAD-CONFIG-BRANCHES — unreachable API-configuration branches and weak assertions

- **Plane:** IC-80
- **Location:** `lib/screens/settings_screen.dart` (save path),
  `lib/config.dart` (`validateStartupApiBaseUrl`),
  `lib/services/api_client.dart` (`healthStatusCode`)
- **Finding:** Three branches cannot be reached from the shipped app: (1) the
  `compileTimeOverride` argument of the settings save path is unreachable
  because Save is disabled in a pinned build; (2) the normalisation inside
  `Config.validateStartupApiBaseUrl` is outcome-equivalent to the validation
  that follows it, because `isValidApiBaseUrl` already trims and a trailing
  slash does not affect validity; (3) the `ApiException` arm in the settings
  screen catch is unreachable because `healthStatusCode` swallows transport
  errors (cross-ref IC-72). The last review also flagged weak assertions: a
  redundant whitespace-only URL case, a test that taps a disabled button, and
  snackbar replace-versus-queue checks that rely on text finders only.
- **Fix:** Remove the dead branches or make them reachable, and tighten the
  flagged assertions.
