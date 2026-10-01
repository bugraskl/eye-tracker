# Building from source

Releases are built by [`.github/workflows/release.yml`](../.github/workflows/release.yml) on GitHub's
runners whenever a `v*` tag is pushed. You can run the same steps locally. Every build must be made
on the operating system it targets: PyInstaller does not cross-compile.

## Prerequisites

- [uv](https://docs.astral.sh/uv/)
- **Windows:** [Inno Setup 6](https://jrsoftware.org/isinfo.php) for the installer (optional)
- **macOS:** Xcode command line tools (`xcode-select --install`)
- **Linux:** `libxcb-cursor0` (Debian/Ubuntu) or `xcb-util-cursor` (Fedora, Arch), which is bundled
  into the AppImage

## Common steps

```bash
uv sync --locked
uv run python scripts/fetch_models.py --check     # verify the committed models and licence texts
uv run python scripts/make_icons.py               # .ico / .icns / .png from the app's own icon code
uv run pyinstaller packaging/pyinstaller/eye-tracker.spec --noconfirm --clean
```

The output lands in `dist/`:

| Platform | Output |
|---|---|
| Windows | `dist/EyeTracker/EyeTracker.exe` (tray app, no console) and `eye-tracker-cli.exe` (command line) |
| macOS | `dist/Eye Tracker.app` (menu-bar app) with `Contents/MacOS/eye-tracker-cli` |
| Linux | `dist/eye-tracker/eye-tracker` (one executable for the app and the command line) |

Then run the privacy gate on the bundle. It reads every native library, and with PyInstaller's own
archive readers every Python module inside the executables and inside ZIP files such as
`base_library.zip`. It fails on telemetry markers, networking packages, Qt's network plugins (TLS,
network information and access, VNC, WebGL, the TUIO listener) and networking code that is not on
its reviewed allow-list. The success line says how many native binaries and Python modules it
checked:

```bash
uv run python scripts/check_privacy.py --bundle dist/EyeTracker        # Windows
uv run python scripts/check_privacy.py --bundle "dist/Eye Tracker.app" # macOS
uv run python scripts/check_privacy.py --bundle dist/eye-tracker       # Linux
```

## Packages

**Windows installer and portable ZIP**

```powershell
iscc /DAppVersion=0.1.0 packaging\windows\installer.iss   # -> dist\EyeTracker-0.1.0-windows-x64-setup.exe
Compress-Archive dist\EyeTracker dist\EyeTracker-0.1.0-windows-x64-portable.zip
```

The installer is per user (no administrator rights), creates a Start menu shortcut, and optionally
registers start at sign-in. It also installs the command-line tool a second time as
`eye-tracker.exe` and, with the task *Add the "eye-tracker" command to PATH* (on by default;
`/MERGETASKS=!addtopath` turns it off in a silent install), adds the installation folder to the
user's PATH; uninstalling removes that entry. Upgrades keep your choices, and a silent upgrade
(winget, `/SILENT`, `/VERYSILENT`) restarts Eye Tracker in the background if it had to close it.

**macOS disk image**

```bash
bash packaging/macos/make_dmg.sh --app "dist/Eye Tracker.app" --version 0.1.0 \
  --out dist/EyeTracker-0.1.0-macos-arm64.dmg
```

The app is signed ad hoc unless you provide a certificate. With a stable identity (a Developer ID,
or a self-signed certificate as described at the top of `make_dmg.sh`), the Camera and Accessibility
permissions survive updates. In CI, set the repository secrets `MACOS_SIGNING_CERT_P12`,
`MACOS_SIGNING_CERT_PASSWORD` and optionally `MACOS_SIGNING_IDENTITY`.

**Linux AppImage and tarball**

```bash
bash packaging/linux/build_appimage.sh --version 0.1.0
```

`appimagetool` and the AppImage runtime are downloaded at pinned versions and verified against their
SHA-256. Where FUSE is unavailable (containers, CI), export `APPIMAGE_EXTRACT_AND_RUN=1`.

## Face models

The models live in [`src/eye_tracker/vision/models/`](../src/eye_tracker/vision/models/) together with
`NOTICE.md` (licences, upstream URLs and checksums) and, in its `licenses/` folder, the full licence
texts that must ship with them (Apache-2.0 and the YuNet MIT notice). To re-download and re-extract
the models from their upstream sources:

```bash
uv run python scripts/fetch_models.py
```

The licence texts are never downloaded: `fetch_models.py --check` fails when one is missing or
altered (restore it from git), and the PyInstaller spec refuses to build without them.

## Continuous integration

- [`ci.yml`](../.github/workflows/ci.yml) runs Ruff, mypy (also as on Windows and macOS), the
  source privacy check, the model check and the tests on Linux, Windows and macOS.
- [`bundle.yml`](../.github/workflows/bundle.yml) builds the three bundles like the release does
  whenever a pull request or a push to `main` changes `packaging/`, the privacy gate, the model or
  icon scripts, `pyproject.toml` or `uv.lock`. It runs the bundle privacy gate, checks that the
  Linux bundle contains no GTK or GIO libraries and that the macOS bundle's minimum version is the
  documented one, and smoke-tests the command-line tool. It publishes nothing.

## Releasing

1. Update `__version__` in `src/eye_tracker/__init__.py` and move the **Unreleased** notes in
   `CHANGELOG.md` under a new `## [x.y.z] - YYYY-MM-DD` heading.
2. Commit, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. The release workflow builds all platforms, smoke-tests each package, runs the privacy gate,
   writes `SHA256SUMS.txt`, attaches build-provenance attestations and publishes the GitHub release
   with the notes from `CHANGELOG.md`. The oldest macOS the DMG runs on follows from the bundled
   wheels (`MACOS_MINIMUM` in the workflows, currently 14.0); the smoke test fails when it changes,
   so that the README, these docs and the release notes are updated with it.
