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
- **Location:** `lib/screens/settings_screen.dart:114` (`_testConnection` catch)
- **Finding:** `e is ApiException ? e.message : e` — the `ApiException` arm was
  dead: `healthStatusCode()` reports an unreachable backend as a null status
  code and never throws `ApiException` here.
- **Fix:** **Delivered by PR #30** — the catch renders the failure with the
  shared `_describeError` helper, so the unreachable arm is gone. A test pins
  the message of an unexpected probe failure.

## ERROR-BODY — _parseErrorMessage reports the raw body, not the status code

- **Plane:** IC-68
- **Location:** `lib/services/api_client.dart:276`
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
- **Location:** `test/home_screen_test.dart:268`
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
  mirror selector `web/2020id_/` was a closest-capture selector rather than a
  pinned capture id — PR #30 pinned the capture id
  (`web/20201130004040id_/`), so the mirror cannot start serving different
  bytes later.
- **Fix:** Use one cache key per workflow and allow the save step to write a
  repair key. The pinned capture id is delivered by PR #30.

## API-MESSAGES — misleading exception messages

- **Plane:** IC-77
- **Location:** `lib/services/api_client.dart:239`,
  `lib/services/api_client.dart:244`,
  `lib/services/api_client.dart:251`
- **Finding:** A `Uri.parse` `FormatException` is reported as
  `Invalid response from ...` although it is a request-construction error, and
  `${e.message}` can render `null`. A missing image file raises a
  `PathNotFoundException` (an `IOException`), not an `ArgumentError`, so it was
  reported as `Connection to ... failed` and the user was sent to look at the
  backend instead of the image.
- **Fix:** Map each exception to the message that describes its real cause.
- **Delivered in PR #30:** a missing image file is now reported as
  `Could not read the image file: ...` (status 400) before the request is
  sent, a file that exists but cannot be read is reported the same way when
  the request body is streamed (`lib/services/api_client.dart:138`), and an
  `ArgumentError` is reported as `Invalid request to ...` with a null-safe
  message. The remaining part of the finding — the `FormatException` arm of the
  request builder — stays open in IC-77.
- **Note:** The port bound at `lib/config.dart:239` (`uri.port > 65535`) is
  load-bearing and must stay: Dart's `Uri` parser does not range-check ports,
  so `http://example.com:99999` parses successfully and only this check
  rejects it.

## README-PINNED — stale pinned-build documentation

- **Plane:** IC-78
- **Location:** `README.md:140-146`
- **Finding:** The README still says the runtime override is ignored when a
  compile-time override is present, but a pinned build can no longer create one
  (Save is disabled and `Config` throws).
- **Fix:** **Delivered by PR #30** — the README now states that a pinned build
  disables Save, refuses to store a runtime override, and that only a build
  without a compile-time override can store one.

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
- **Location:** `lib/config.dart:102` (`setApiBaseUrl`),
  `lib/config.dart:175` (`_rollbackStoredUrl`),
  `lib/config.dart:210` (`clearApiBaseUrl`),
  `lib/screens/settings_screen.dart:188` (`_resetToDefault`)
- **Finding:** `shared_preferences` 2.5.6 updates its in-memory cache *before* the
  platform call: `_setValue` writes `_preferenceCache[key]` and `remove` deletes
  `_preferenceCache[key]` before the platform answers. A refused write therefore
  left the rejected URL effective for the whole session while the UI reported
  `Could not save`, and a refused removal left the override inactive while the
  UI claimed it was "still in use". The platform may also restore a refused
  removal after a restart, which Dart cannot prevent.
- **Fix:** **Delivered by PR #30** — a refused write now restores the persisted
  state (`Config._rollbackStoredUrl`): it reloads the stored preferences and, if
  the reload itself fails, writes the previously stored address back, so the app
  keeps using the address that is actually stored instead of the rejected one
  or the platform default. The refusal is reported with its cause. A refused
  removal is reported as "not active now, but it may return after a restart".
  The remaining platform-level gap (confirming a removal by re-reading the
  persisted value) is tracked in IC-81.

## DEAD-CONFIG-BRANCHES — unreachable API-configuration branches and weak assertions

- **Plane:** IC-80
- **Location:** `lib/screens/settings_screen.dart` (save path),
  `lib/config.dart` (`validateStartupApiBaseUrl`),
  `lib/services/api_client.dart` (`healthStatusCode`)
- **Finding:** Four branches cannot be reached from the shipped app: (1) the
  `compileTimeOverride` argument of the settings save path is unreachable
  because Save is disabled in a pinned build; (2) the normalisation inside
  `Config.validateStartupApiBaseUrl` is outcome-equivalent to the validation
  that follows it, because `isValidApiBaseUrl` already trims and a trailing
  slash does not affect validity; (3) the `ApiException` arm in the settings
  screen catch was unreachable because `healthStatusCode` reports an
  unreachable backend as a null status code — PR #30 removed that arm
  (cross-ref IC-72); (4) the `ArgumentError` arm of the settings
  screen `_describeError` helper was unreachable because `_save` validates the
  URL with the same predicate before calling `Config`, so `Config` only ever
  raises a `StateError` there — PR #30 removed that arm. The generic
  `error.toString()` fallback of `_describeError` is **reachable**: the
  `ApiClient` provider read now happens inside the guarded `try`, so a missing
  provider raises a `ProviderNotFoundException` that reaches the fallback, and
  a test pins the error type in the message. The last review also
  flagged weak assertions: a redundant whitespace-only URL case, a test that
  taps a disabled button, and snackbar replace-versus-queue checks that rely
  on text finders only.
- **Fix:** Remove the dead branches or make them reachable, and tighten the
  flagged assertions.

## PREDICTION-ERROR-CATCH — the provider catches every Error subtype

- **Plane:** IC-82
- **Location:** `lib/providers/prediction_provider.dart:61`
- **Finding:** The `on Error` arm of the prediction request catches every
  `Error` subtype — `AssertionError`, `OutOfMemoryError` and a failing
  `TypeError` alike — and surfaces `e.toString()` to the user.
- **Fix:** Narrow the arm to the errors that can actually escape the request
  path, or map the caught error to a user-facing message that does not expose
  the raw error text. The arm must stay broad enough to keep the provider out
  of a permanently stuck loading state.
- **Delivered by PR #30:** the provider now catches a `TypeError` with the
  fixed message "The backend returned an unexpected response." and every other
  `Error` with "The prediction failed with an unexpected error." The raw error
  text is no longer shown, and the provider still never stays in a stuck
  loading state.

## CURL-HARNESS-RANGE — the stubbed curl ignores the requested byte range

- **Plane:** IC-83
- **Location:** `backend/tests/test_ci_cifar_script.py` (stub `curl`)
- **Finding:** The stub appended the whole payload when `-C` was requested,
  instead of serving the bytes from the requested offset the way a real
  `curl -C -` does. A resumed download in the harness was therefore always
  corrupt, so the resume path could only be asserted as a failure.
- **Fix:** **Delivered by PR #30** — the stub now serves the payload from the
  requested byte offset, so `curl -C -` resumes the way the real client does,
  and a test asserts that a resumed download completes the archive and matches
  both pinned digests.

## CONFIG-PREV-READ — the previous-URL read is outside the guarded try

- **Plane:** IC-84
- **Location:** `lib/config.dart:137`
- **Finding:** `Config.setApiBaseUrl` reads the previously stored URL with
  `prefs.getString(apiBaseUrlPrefsKey)` outside the guarded try. A non-string
  value under that key would raise a raw `TypeError` instead of the documented
  `StateError`. The shipped app only ever writes a string under the key, so
  the path is not reachable today.
- **Fix:** Move the read inside the guarded region, or read it with a type
  check that falls back to the documented `StateError`.
- **Delivered in PR #30:** the read is guarded now, so a value of another type
  falls back to "nothing was stored" instead of escaping as a `TypeError`. The
  rollback then drops the refused URL, which is the safe outcome.

## CURL-RETRY-RANGE — curl's automatic retry is not byte-safe with -C -

- **Plane:** IC-86
- **Location:** `backend/scripts/fetch_cifar10.sh` (the `curl -fL -C -` call)
- **Finding:** Some curl builds re-send bytes when an automatic retry follows a
  partial transfer, so a resumed download may not be byte-exact. The archive
  integrity is unaffected: the pinned md5 and sha256 reject a corrupt file and
  the per-source purge discards it. The impact is a wasted download budget,
  never a wrong archive.
- **Fix:** Download the remaining range in explicit chunks, or drop the
  automatic retries and rely on the per-source attempt loop, which already
  resumes from the partial file.
- **Delivered by PR #30:** the automatic retries are dropped (`--retry` and
  `--retry-all-errors` are no longer passed), so a resumed transfer is never
  spliced. The per-source attempt loop retries with a fresh resume, and a test
  pins that the automatic retry flags stay absent.

## CURL-TIMEOUT-KILL — the per-attempt timeout waits for a stubborn child

- **Plane:** IC-85
- **Location:** `backend/scripts/fetch_cifar10.sh` (the `timeout --kill-after=30
  "$TIMEOUT"` wrapper)
- **Finding:** `timeout` without `--kill-after` keeps waiting for a child that
  ignores `SIGTERM`. `curl` handles `SIGTERM`, so the practical risk is low,
  but a wedged child would hold the job past its budget.
- **Fix:** Add `--kill-after` to the `timeout` call so the attempt is bounded
  even if the child does not exit on `SIGTERM`.
- **Delivered by PR #30:** the wrapper now uses
  `timeout --kill-after=30 "$TIMEOUT"`, so an attempt cannot outlive its
  budget by 30 s even when the child ignores `SIGTERM`.
