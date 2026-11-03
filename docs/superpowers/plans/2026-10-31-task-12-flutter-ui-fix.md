# Task 12 — Flutter UI Fix: Image Selection, Drag & Drop, Prediction Flow

**Status:** Ready for implementation in a new session
**Branch:** `feature/task-12-flutter-ui-fix` (from `origin/development`)
**Base at planning time:** `d958774` (development, clean)

## Problem (user report)

> "in the UI, there isn't any buttons or anything and the drag and drop doesn't work."

## Verified root causes

1. **Wrong screen is mounted.** `lib/main.dart:42-52` defines a duplicate placeholder `HomeScreen`, and `lib/main.dart:37` uses it (`home: const HomeScreen()`). The real screen at `lib/screens/home_screen.dart` is never instantiated. The running app only shows the static text "Drag and drop an image to classify" — no buttons, no drop zone.
2. **No providers at app root.** `lib/main.dart` has no `MultiProvider`. The real screen requires `ImageService` (`lib/widgets/drop_zone.dart:21`) and `PredictionProvider` (`lib/screens/home_screen.dart:18`). Without them it throws `ProviderNotFoundException`. `ApiClient` also needs an `http.Client` (`lib/services/api_client.dart:29`).
3. **Picker is unimplemented.** `lib/services/image_service.dart:14-17` — `pickImage()` throws `UnimplementedError('pickImage not implemented yet')`.
4. **Drag & drop can never fire.** `lib/widgets/drop_zone.dart:35` uses `DragTarget<File>`. Flutter's `DragTarget` only receives in-app `Draggable` data; native OS file drag-and-drop on Linux requires `desktop_drop` (`DropTarget`, `DropDoneDetails.files` are `XFile`). There is no drag source in the app, so the target never matches.
5. **Selection is private and unused.** `lib/widgets/drop_zone.dart:17,24-27` stores the picked file in local state only. It never notifies `HomeScreen`, never calls `PredictionProvider.predict()`, and there is no Classify button.
6. **Placeholder result navigation.** `lib/screens/home_screen.dart:182-202` pushes `ResultScreen` with `XFile('')` and an empty `ModelInfo`, so `lib/screens/result_screen.dart:118` calls `Image.file(File(''))` and fails.
7. **Row overflow.** `lib/screens/home_screen.dart:135-171` builds a `Row` with `Expanded` + `Spacer` + `FilledButton`; this overflows at narrow widths and is one source of the widget-test render failures.
8. **Result image unguarded.** `lib/screens/result_screen.dart:118` assumes a valid file path; no `errorBuilder`, no empty-path guard.
9. **Flutter tests are red.** Baseline: **75 pass / 7 fail** (`flutter test --reporter expanded`):
   - `PredictionBars widget shows class name and percentage labels`
   - `ModelInfoCard widget displays model name, input size, classes count, device`
   - `HomeScreen widget shows drop zone and no preview when idle`
   - `HomeScreen widget shows loading overlay when predicting`
   - `HomeScreen widget navigates to ResultScreen on successful prediction`
   - `ResultScreen widget shows image preview, prediction bars, model info, and classify another button`
   - `ResultScreen widget Classify Another button pops back to HomeScreen`
   Likely causes: unbounded `BarChart` height in tests, `Image.file` with nonexistent paths, missing providers, and the Row overflow.
10. **CI integrity is broken.** `.github/workflows/flutter-ci.yml:53` runs `flutter test || echo "Tests completed with failures (UI layout issues in CI env)"`. CI is green while tests fail. This must be restored to plain `flutter test`.

Current local state at planning time: working tree clean, branch `development`, `flutter analyze` clean, `dart format` clean, backend 159/159 tests pass.

## Rules that apply

- Mandatory `implement-review-ship` loop: branch from `origin/development`, TDD, FAST gate before every commit, FULL gate before PR, two brand-new context-free reviewers (`qa-post-implementation` + `general` running `test-coverage-analysis`), fix loop max 3 rounds, PR into `development`, **user merges manually** (branch protection blocks direct pushes).
- Commits via `emoji-commit`; stage explicit paths only; never `git add .`/`-A`.
- Scope: Flutter UI fix + test/CI integrity only. Do not touch backend logic.
- Out-of-scope findings become `docs/followups.md` + Plane work items, not in-branch fixes.

## Implementation steps

### Step 0 — Branch and baseline

```bash
git fetch origin --prune
git checkout -b feature/task-12-flutter-ui-fix origin/development
flutter pub get
flutter analyze                      # expect clean
flutter test                         # expect 75 pass / 7 fail
dart format --set-exit-if-changed lib test
```

### Step 1 — Wire the app root (`lib/main.dart`)

- Delete the duplicate placeholder `HomeScreen` (lines 42-52).
- Import the real screen: `package:image_classifier/screens/home_screen.dart`.
- Add `MultiProvider`:

```dart
MultiProvider(
  providers: [
    Provider<ImageService>(create: (_) => ImageService()),
    Provider<ApiClient>(
      create: (_) => ApiClient(client: http.Client()),
      dispose: (_, client) => client.client.close(),
    ),
    ChangeNotifierProvider<PredictionProvider>(
      create: (ctx) => PredictionProvider(apiClient: ctx.read<ApiClient>()),
    ),
  ],
  child: const HomeScreen(),
)
```

- `ApiClient` signature is `ApiClient({required http.Client client, String? baseUrl, Duration? timeout})` (`lib/services/api_client.dart:29`).
- `SystemChrome.setPreferredOrientations` is mobile-only; keep or guard with `defaultTargetPlatform` (not required for the fix).

### Step 2 — Implement `ImageService.pickImage()` (TDD)

- Make the picker injectable for tests, e.g.:

```dart
typedef PickFile = Future<XFile?> Function();

class ImageService {
  ImageService({PickFile? picker}) : _picker = picker ?? _defaultPicker;
  final PickFile _picker;
```

- Default behavior:
  - Linux/Windows/macOS: `FilePicker.platform.pickFiles(type: FileType.image, allowMultiple: false)`; return `XFile(result.files.single.path!)`; `null` on cancel.
  - Android/iOS: `ImagePicker().pickImage(source: ImageSource.gallery)` (returns `XFile?`).
- Tests in `test/image_service_test.dart`: cancel → `null`; selection → `XFile` with expected path; picker exception → rethrow/translate consistently.
- Red → implement → green → commit.

### Step 3 — Native drag & drop + `DropZone` refactor

- Add `desktop_drop` to `pubspec.yaml` (`flutter pub add desktop_drop`), verify `flutter build linux --debug` links it.
- Rewrite `lib/widgets/drop_zone.dart`:
  - Wrap content in `DropTarget(onDragEntered:, onDragExited:, onDragDone:)`; use `details.files` (`List<XFile>`).
  - Accept `ValueChanged<XFile> onImageSelected` (and optional `VoidCallback? onClassify` / `bool isBusy`).
  - Keep tap-to-pick; show picker progress and surface errors via `SnackBar`.
  - Constrain preview: fixed height (`SizedBox(height: 240)`) or `AspectRatio`; `Image.file(..., errorBuilder: ...)`.
  - Add an explicit **Classify** `FilledButton` when an image is selected (the user explicitly complained about missing buttons).
  - Extract file handling into a testable method (`handleFiles(List<XFile>)`) so drag-drop logic can be unit-tested without simulating OS drags.
- Tests: tap invokes picker; selected file shows preview + Classify button; `handleFiles` accepts first file and rejects empty list.

### Step 4 — `HomeScreen` orchestration

- Hold state: `XFile? _selectedImage`, `ModelInfo? _modelInfo`, `bool _navigated`.
- On image selected: store it; validate/compress via `ImageService.validateAndCompress(File(path))`; fetch/cache `ApiClient.getModelInfo()`.
- Classify button → `context.read<PredictionProvider>().predict(compressedFile)`.
- Observe provider state:
  - `PredictionStateLoading` → `LoadingOverlay`.
  - `PredictionStateResult` → push `ResultScreen` once with the **real** `XFile`, response, and `ModelInfo`; guard `_navigated`.
  - `PredictionStateError` → visible error (SnackBar/banner) with retry.
- Remove the placeholder `_ResultPreview` navigation (`XFile('')`), or rebuild it with real data.
- Fix the `Row` overflow: avoid `Expanded` + `Spacer` + button in one Row; use `Column`/`Wrap` on narrow widths.

### Step 5 — `ResultScreen` hardening

- Keep `ResultScreen({required XFile imageFile, required PredictionResponse response, required ModelInfo modelInfo})`.
- `_ImagePreview`: read bytes with `FutureBuilder(imageFile.readAsBytes())`, render `Image.memory`; `errorBuilder` → placeholder icon + "Preview unavailable". Never `Image.file(File(''))`.
- Bound the chart: `SizedBox(height: 280, child: PredictionBars(...))` (fl_chart needs finite height).

### Step 6 — Fix and extend tests

- Update `test/widget_test.dart`:
  - Provide `Provider<ImageService>` and `ChangeNotifierProvider<PredictionProvider>` (mock) in every `HomeScreen` test.
  - Use a real tiny PNG (bytes) or a temp file for previews; no nonexistent paths.
  - Wrap `PredictionBars` in a bounded `SizedBox`.
  - Fix the 7 failing tests listed above.
  - Add tests: tap → picker called; selection → Classify button; Classify → `predict()` called; result → navigation with real data; error → message shown; `handleFiles` path.
- Regenerate mocks: `dart run build_runner build --delete-conflicting-outputs`.
- Add `test/image_service_test.dart` for picker behavior.

### Step 7 — Restore CI integrity

- `.github/workflows/flutter-ci.yml:53`: `run: flutter test` (remove `|| echo ...`).
- Keep `flutter analyze --fatal-infos` and `dart format --set-exit-if-changed lib test`.
- Consider adding `flutter build linux --debug` to CI (verify runner dependencies; the plan's Task 11 asked for a release build — decide with reviewers).

### Step 8 — Verification (FULL gate)

```bash
dart format --set-exit-if-changed lib test
flutter analyze --fatal-infos
flutter test                       # all pass, 0 skips
flutter build linux --debug
cd backend && CNN_TEST_DATA_DIR=/mnt/ALL/LinuxDev/caches/.caches_mount/tmp/opencode/cifar10-data \
  /usr/bin/python3 -m pytest tests/test_api_predict.py tests/test_predictor.py tests/test_config.py -q   # 159 expected
git diff origin/development...HEAD --stat   # scope audit
```

### Step 9 — Review loop, PR, Plane

- Dispatch **both** fresh reviewers verbatim (repo path, branch `feature/task-12-flutter-ui-fix`, base `origin/development`, one-line description). Never reuse reviewer sessions.
- Apply findings; new reviewers each round; max 3 rounds → human checkpoint.
- `gh pr create --base development` with real scores/coverage/CI evidence; **HALT** for user merge.
- Plane: create/update Task 12 work item in project `IC` (state `In Progress` → `Review`); `Done` only after user merge + Gate 2 verification.

## Definition of Done

- [ ] App mounts the real `HomeScreen` with all providers; visible drop zone, pick action, and Classify button
- [ ] Tap opens the file picker; cancel is safe; errors are surfaced
- [ ] OS drag-and-drop works via `desktop_drop`
- [ ] Classify calls the backend: loading → result or error
- [ ] `ResultScreen` shows the actual image, bars, and model info; no placeholder `XFile('')`
- [ ] `flutter test` fully green; CI no longer masks failures
- [ ] `flutter analyze --fatal-infos` and `dart format` clean; Linux build succeeds
- [ ] `implement-review-ship` loop passed on current HEAD; PR open; user merged; Gate 2 verified
- [ ] Plane Task 12 work item `Done`
