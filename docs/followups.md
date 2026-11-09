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
- **Location:** `lib/services/api_client.dart:340`
- **Finding:** The backend error body is shown verbatim to the user instead of the
  HTTP status code, so a proxy or HTML error page leaks through the message.
- **Fix:** Report the response status code and keep the body only for structured
  JSON error payloads.
- **Delivered by PR #30:** `_parseErrorMessage` now takes the status code and
  falls back to `Request failed with status <code>`
  (`lib/services/api_client.dart:340`); only a JSON payload with a `detail`
  field is used as the message, so an HTML error page is no longer echoed to the
  user. The status code of the response is the status of the `ApiException`
  (`lib/services/api_client.dart:311`).

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
- **Location:** `lib/services/api_client.dart:247`,
  `lib/services/api_client.dart:255`,
  `lib/services/api_client.dart:340`
- **Finding:** A `Uri.parse` `FormatException` is reported as
  `Invalid response from ...` although it is a request-construction error, and
  `${e.message}` can render `null`. A missing image file raises a
  `PathNotFoundException` (an `IOException`), not an `ArgumentError`, so it was
  reported as `Connection to ... failed` and the user was sent to look at the
  backend instead of the image.
- **Fix:** Map each exception to the message that describes its real cause.
- **Delivered in PR #30:** a missing image file is now reported as
  `Could not read the image file.` (status 400) before the request is
  sent, a file that exists but cannot be read is reported the same way when
  the request body is streamed (`lib/services/api_client.dart:145`), a response
  body that cannot be read is reported as `Could not read the response from
  ...` (status 502) instead of `Connection to ... failed`
  (`lib/services/api_client.dart:153`, `lib/services/api_client.dart:202`),
  a response body that never ends is reported as a response problem (status
  502, `Could not read the response from ...`) instead of hanging the request
  or blaming the backend, and an `ArgumentError` is reported as
  `Invalid request to ...` with a null-safe message that carries no raw
  argument text.
- **Delivered in PR #30:** every transport failure now carries one status code
  (503) and one fixed message (`lib/services/api_client.dart:247-254`,
  `lib/services/api_client.dart:300`), so a TLS failure no longer shows
  `HandshakeException: ...` and the backend is not blamed for a file problem.
  The file messages carry no local path, and the `FormatException` arm
  (`lib/services/api_client.dart:255`) uses fixed wording without the raw
  exception text.
- **Still open in IC-77:** the `FormatException` arm covers a
  request-construction failure and a response-parse failure at the same time,
  so the wording is still wrong for the first case. The two failures need two
  messages.
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
- **Location:** `lib/providers/prediction_provider.dart:59`
- **Finding:** The `on Error` arm of the prediction request catches every
  `Error` subtype — `AssertionError`, `OutOfMemoryError` and a failing
  `TypeError` alike — and surfaces `e.toString()` to the user.
- **Fix:** Narrow the arm to the errors that can actually escape the request
  path, or map the caught error to a user-facing message that does not expose
  the raw error text. The arm must stay broad enough to keep the provider out
  of a permanently stuck loading state.
- **Delivered by PR #30:** the provider now catches a `TypeError` with the
  fixed message "The backend returned an unexpected response.", every other
  `Error` with "The prediction failed with an unexpected error." and a plain
  `Exception` with "The prediction failed.". The raw error text is no longer
  shown, and the provider still never stays in a stuck loading state.

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

## SNACKBAR-QUEUE — hiding a snackbar does not clear the queued ones

- **Plane:** IC-87
- **Location:** `lib/screens/home_screen.dart:251`,
  `lib/screens/home_screen.dart:268`
- **Finding:** `ScaffoldMessengerState.hideCurrentSnackBar()` reverse-animates
  the visible bar but does not remove it from the queue, so a burst of errors
  can leave more than one bar queued and the new message waits for the exit
  animation of the old one.
- **Fix:** Use `clearSnackBars()` before showing a message, or await the future
  of `hideCurrentSnackBar()`.

## CI-NETWORK-PATH — CI never exercises the download over the network

- **Plane:** IC-88
- **Location:** `.github/workflows/backend-ci.yml`
- **Finding:** Every green CI run restored the CIFAR-10 cache, so the `curl`
  path is covered only by the stubbed contract tests and by one local
  end-to-end run. A regression that only shows up against the real mirror
  would stay invisible until a cold cache happens.
- **Fix:** Add a scheduled or manual cold-cache job that runs
  `backend/scripts/fetch_cifar10.sh` against an empty cache.

## SCRIPT-SHEBANG — the download script carries a shebang but is not executable

- **Plane:** IC-89
- **Location:** `backend/scripts/fetch_cifar10.sh:1`
- **Finding:** The file starts with `#!/usr/bin/env bash` but is committed with
  mode `100644` on purpose — a test pins that it must not be executable — so
  the shebang suggests an invocation the contract forbids.
- **Fix:** Drop the shebang, or state in the usage header that the script is
  run with `bash`.

## LOCK-PROVENANCE — requirements.lock does not list pyyaml as a direct dependency

- **Plane:** IC-90
- **Location:** `backend/requirements.lock:997`
- **Finding:** `pyyaml` was added as a direct dev dependency in
  `backend/pyproject.toml`, but the lock still records it as a transitive
  dependency. The resolution is unchanged (`6.0.3`) and no workflow consumes
  the lock, so the drift is cosmetic.
- **Fix:** Regenerate the lock with
  `uv pip compile pyproject.toml --extra dev --generate-hashes -o requirements.lock`.

## TRANSPORT-STATUS — transport failures use two different status codes

- **Plane:** IC-91
- **Location:** `lib/services/api_client.dart:247-254`
- **Finding:** A transport failure is reported with status `502` when the
  client raises an `http.ClientException` and with status `503` when it
  raises another `IOException`, so the same user-visible problem (the backend
  is not reachable) carries two different codes. The messages differ as well:
  the `ClientException` arm names the exception message, the `IOException` arm
  names the whole exception.
- **Fix:** Map every transport failure to one status code and one message.
- **Delivered by PR #30:** a `SocketException`, an `http.ClientException`, an
  `HttpException` and an `IOException` all raise the same `ApiException`
  (status 503, `lib/services/api_client.dart:300`), so one unreachable backend
  now carries one code and one message.

## IMAGE-SIZE — the rejection message does not name the size of the file

- **Plane:** IC-92
- **Location:** `lib/services/image_service.dart`
- **Finding:** The image validation rejects a file with `the file is larger than
  the 10485760 byte limit`, so the user sees the limit but not how large the
  selected file actually was.
- **Fix:** Name the size of the rejected file in the message.

## SETTINGS-EMPTY-FIELD — a cleared URL field is refilled with the stored URL

- **Plane:** IC-93
- **Location:** `lib/screens/settings_screen.dart:49-56`
- **Finding:** The stored URL is loaded asynchronously. If the user types a URL
  and clears the field before the load finishes, the late load writes the
  stored URL back into the field.
- **Fix:** Track whether the user edited the field and skip the late write
  when they did.

## SCRIPT-LINT — the download script is not linted by shellcheck

- **Plane:** IC-94
- **Location:** `backend/scripts/fetch_cifar10.sh`
- **Finding:** The script is covered by the contract tests in
  `backend/tests/test_ci_cifar_script.py`, but no shell linter runs over it, so
  a shellcheck finding only shows up during review.
- **Fix:** Add a shellcheck step to the backend CI workflow.

## GITLEAKS-PIN — the secrets scan action is not pinned to a commit SHA

- **Plane:** IC-95
- **Location:** `.github/workflows/flutter-ci.yml:34`
- **Finding:** The Secrets Scan job uses `gitleaks/gitleaks-action@master`, so
  the action can change without a review. Every other action that this branch
  touches is pinned to a commit SHA.
- **Fix:** Pin the action to a commit SHA and add a workflow contract test that
  holds it there.

## TEMP-DIR-CLEANUP — the temporary image directory is never deleted

- **Plane:** IC-96
- **Location:** `lib/services/image_service.dart:128`
- **Finding:** `Directory.systemTemp.createTempSync('image_classifier_')` creates
  a directory for every compressed image and nothing deletes it, so the
  directories and the JPEG inside them accumulate in the system temp folder.
- **Fix:** Delete the directory once the caller has read the file, or document
  who owns it and when it is removed.

## LINT-SCOPE — the CI lint scope differs from the local gate

- **Plane:** IC-97
- **Location:** `.github/workflows/backend-ci.yml:42`,
  `.github/workflows/image-classifier-ci.yml:80`
- **Finding:** The workflows lint `src/ tests/ scripts/ conftest.py` while the
  local gate runs the tools over the whole backend directory. The CI scope
  already reaches every Python file, so nothing is missed today; the only
  difference is that the local gate additionally inspects
  `backend/pyproject.toml` for unused `noqa` directives.
- **Fix:** Use the same scope in the workflow and in the local gate, so a new
  top level module cannot slip through unnoticed.

## PERCENT-URL — every percent sign in the URL is rejected

- **Plane:** IC-98
- **Location:** `lib/config.dart:233`
- **Finding:** The URL validator rejects the whole address when it contains a
  `%` anywhere, so a legitimate percent-encoded path is refused even though
  the rule was meant to stop an encoded host or an encoded path segment.
- **Fix:** Reject a percent escape only where it can change the host or the
  path of the request, and accept the rest.

## PARTIAL-PURGE — a partial download survives a killed script run

- **Plane:** IC-99
- **Location:** `backend/scripts/fetch_cifar10.sh` (the download loop)
- **Finding:** The script removes the partial archive when the last attempt of
  a source is used up, but a run that is killed while a transfer is in flight
  leaves the partial file behind. The next run starts the transfer with
  `-C -`, so it resumes from the partial file of the previous run.
- **Fix:** Remove the partial file when the script starts, or verify that a
  resumed file belongs to the current source before it is used.

## PROBE-URL — a malformed address is reported as an unreachable backend

- **Plane:** IC-100
- **Location:** `lib/services/api_client.dart:82-94` (`healthStatusCode`)
- **Finding:** The connection probe returns no status code for a malformed base
  URL and for an unreachable backend, so the settings screen reports the same
  cause for both. The user cannot tell a typo from a backend that is down.
- **Fix:** Report the malformed address and the unreachable backend as two
  different causes.

## DISPOSE-NOOP — dispose() does nothing

- **Plane:** IC-101
- **Location:** `lib/services/api_client.dart:228`
- **Finding:** `dispose()` is a no-op because the client never owns the
  `http.Client`. The method is part of the public contract and suggests a
  cleanup that never happens.
- **Fix:** Remove the method, or make it close a client that the instance owns.

## TRAILING-SLASH — the normalized address drops every trailing slash

- **Plane:** IC-102
- **Location:** `lib/config.dart:51` (`normalizeApiBaseUrl`)
- **Finding:** The normalization removes all trailing slashes, so an address
  that names a path (`http://host:8000/api/`) loses it and the request goes to
  a different route.
- **Fix:** Remove only the slashes that the request builder adds itself, or
  keep the path of the address.

## SETTINGS-ERROR-TEXT — the settings banner echoed the raw error and the whole address

- **Plane:** IC-103
- **Location:** `lib/screens/settings_screen.dart:78` (`_describeError`),
  `lib/screens/settings_screen.dart:89` (`_displayUrl`)
- **Finding:** The banner described an unknown failure with
  `error.toString()`, so a raw Dart error object reached the user, and it
  echoed the typed address without a limit, so a long or control-character
  address could push the message out of the banner.
- **Fix:** Describe the failure from an allow-list of known causes and shorten
  the address that the banner shows.
- **Delivered by PR #30:** `_describeError` now answers from an allow-list (a
  `StateError` shows its message, a missing provider names the provider, every
  other failure reads `an unexpected error occurred`), and `_displayUrl`
  removes control characters and shortens the address to 119 characters plus
  an ellipsis.

## SEND-ABORT — a request that stalls while it is sent keeps its socket open

- **Plane:** IC-105
- **Location:** `lib/services/api_client.dart` (`client.send(request).timeout(_timeout)`
  in `predict` and `predictBytes`, `client.send(...)` in `getModelInfo`)
- **Finding:** `Future.timeout` does not cancel the work it gives up on, and
  `package:http` 1.2.2 has no `Abortable` client (that contract only exists
  from http 1.6.0 on), so a request that stalls while it is being sent keeps
  its socket open until the server or the client gives up. The response body
  is now read through an explicit subscription that is cancelled when the
  deadline expires, but the send itself cannot be aborted.
- **Fix:** Upgrade `package:http` to a version that exposes `Abortable`, or
  close the client when the deadline of a send expires.

## PREEXISTING-COVERAGE — untouched UI units have untested branches

- **Plane:** IC-104
- **Location:** `lib/screens/home_screen.dart` (`_onClassify`, `_clearPrediction`,
  `_onViewDetails`, the second navigation, the two result buttons, the model
  info success path, the model info spinner, the confidence bands),
  `lib/services/image_service.dart` (`_defaultPicker`)
- **Finding:** A coverage review of this branch reported these units as
  branches without a test. They are pre-existing code that this branch does
  not change, so they are outside the scope of the branch.
- **Fix:** Add the missing tests for the untouched units.

## HARNESS-BRANCHES — the download script test harness has untested branches

- **Plane:** IC-106
- **Location:** `backend/tests/test_ci_cifar_script.py` (the harness that
  creates the cache directory, the skip of a blank line in the stub log, the
  assertion that looks a workflow step up by name)
- **Finding:** Three branches of the test harness itself are never taken. They
  are branches of the tests, not of the shipped script, and the script contract
  itself is covered.
- **Fix:** Cover the branches or remove the dead ones.

## RANGE-LESS-SOURCE — a download source without ranged requests cannot be resumed

- **Plane:** IC-107
- **Location:** `backend/scripts/fetch_cifar10.sh` (the curl call with `-C -`
  and the purge of the partial file on the last attempt)
- **Finding:** `curl -C -` asks for the missing bytes with a range request. A
  source that ignores the range answers with the whole file, so curl aborts
  and every attempt of that source fails until the partial file is purged.
  The partial is purged only when the attempts of a source are used up, so a
  source that refuses a range request burns the whole budget. Both pinned
  sources answer a range request with `206`, so the risk is latent.
- **Fix:** Purge the partial file before the first attempt of a source, or
  retry without `-C -` when curl reports a range error.

## CLEAR-CAUSE — a failed removal of the stored URL names no cause

- **Plane:** IC-108
- **Location:** `lib/config.dart` (`clearApiBaseUrl`), the reset banner of
  `lib/screens/settings_screen.dart`
- **Finding:** `clearApiBaseUrl` answers every failure with a single
  `false`, so a refused removal and an unavailable plugin are told apart by
  nothing. The banner therefore names no cause, although the same screen
  names the cause of every other failure.
- **Fix:** Return the cause next to the result and render it in the banner.

## BYTE-UPLOAD-MESSAGE — the byte upload names a file that does not exist

- **Plane:** IC-109
- **Location:** `lib/services/api_client.dart` (the file fault of the byte
  upload path)
- **Finding:** The byte upload path reports `Could not read the image file.`
  although it builds its body from memory and reads no file. The arm is
  reachable only through a custom client.
- **Fix:** Use a message that fits the byte path.

## IMAGE-LIMIT-UNIT — the image size limit is rendered as a raw byte count

- **Plane:** IC-110
- **Location:** `lib/services/image_service.dart` (the size limit in the
  rejection message)
- **Finding:** The rejection message names the limit as `10485760 byte
  limit`, which the user has to convert by hand.
- **Fix:** Render the limit in MiB.

## REGISTER-POLISH — the register and the contract harness need polish

- **Plane:** IC-111
- **Location:** `docs/followups.md` (the register mixes open, delivered and
  rejected items without a status field), `lib/services/image_service.dart`
  (the `maxBytes` parameter is a public test seam of a shipped service),
  `backend/tests/test_ci_cifar_script.py` (every test stubs curl, so the real
  tool semantics of `--speed-limit`, `--speed-time`, `--proto` and `-C -` are
  only asserted through the arguments)
- **Finding:** Three small findings of the round 25 review. The register is
  accurate but hard to triage, the test seam widens the public surface of the
  image service, and the contract suite never runs the real tool.
- **Fix:** Add a status field to the register, narrow the test seam and
  exercise the real binary in the contract suite.

## REDIRECT-HOST-ALLOWLIST — the download follows any HTTPS redirect host

- **Plane:** IC-112
- **Location:** `backend/scripts/fetch_cifar10.sh:124`
- **Finding:** The round 26 review (Info severity) found that the download
  runs `curl -fL` and pins only the scheme (`--proto-redir '=https'`), so a
  redirect may send the transfer to any HTTPS host. The pinned md5 and sha256
  gate keeps the content safe, so the exposure is availability only.
- **Fix:** Resolve the redirect chain first and allow only the known mirror
  hosts, then fetch the archive from an allowed host.
