# Contributing

Thanks for helping make Eye Tracker better. Reproducible bug reports, accuracy data from real
setups, and small focused pull requests are the most valuable contributions.

## Ground rules

- **Privacy is a hard constraint.** No network code under `src/`, no frames written to disk, no
  telemetry. `scripts/check_privacy.py` enforces this in CI; please don't work around it.
- **Degrade, never crash.** Platform features are best effort. An unsupported call returns
  `False`/`None` and the app keeps running.
- **Pure logic stays pure.** `gaze/*` and `engine/*` (except `controller.py`) import neither Qt nor
  OpenCV and take time as an argument, so they can be tested deterministically.
- **Measure performance claims.** If a change affects CPU or latency, include `eye-tracker bench`
  numbers before and after.

## Development setup

You need [uv](https://docs.astral.sh/uv/) (it installs the right Python automatically).

```bash
git clone https://github.com/bugraskl/eye-tracker.git
cd eye-tracker
uv sync
uv run eye-tracker            # start the tray app from source
uv run eye-tracker doctor     # environment and permission report
```

No webcam? Point the app at a video or photo instead:

```bash
uv run eye-tracker --camera path/to/face-video.mp4
```

## Checks

Run everything CI runs before opening a pull request:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python scripts/check_privacy.py
uv run pytest
```

After a local PyInstaller build ([docs/building.md](docs/building.md)), also run the bundle privacy
gate, which scans every bundled native library:

```bash
uv run python scripts/check_privacy.py --bundle dist/EyeTracker
```

Tests run Qt on the `offscreen` platform, so they work over SSH and on headless CI. Tests that need a
real face photo are skipped unless `EYE_TRACKER_TEST_FACE` points to one (face photos are never
committed).

## Project layout

```
src/eye_tracker/
  vision/     camera capture, motion gate, face backends (MediaPipe, OpenCV), worker thread
  gaze/       gaze regression model, smoothing, calibration, implicit learning
  engine/     switching decision, presence, shoulder guard, rate scheduler, controller
  platform/   OS integration (Windows, macOS, Linux), autostart, global hotkeys
  ui/         tray, settings, calibration, overlays, first-run wizard
  app.py      Qt application wiring
  cli.py      command-line interface
packaging/    PyInstaller spec, Windows installer, macOS DMG, Linux AppImage
scripts/      model fetch/verification, privacy check, icon generation
docs/         user and developer documentation
```

[docs/architecture.md](docs/architecture.md) explains how the pieces fit together.

GitHub Actions are pinned to full commit SHAs; Dependabot proposes updates.

## Pull requests

1. Open an issue first for anything larger than a bug fix, so we can agree on the approach.
2. Keep pull requests focused; one behaviour change per PR.
3. Add or update tests. Logic changes in `engine/` or `gaze/` need unit tests.
4. Update `CHANGELOG.md` under **Unreleased**.
5. Platform-specific changes: say which OS and desktop environment you tested on.

## Reporting bugs

Please use the bug report template and paste the output of `eye-tracker doctor`. It contains your
OS, camera and monitor layout and feature support, and **no images or personal data**.

## Code of conduct

Be kind and constructive. Harassment or personal attacks are not tolerated in issues, pull requests
or discussions; maintainers may remove such content and block repeat offenders.
