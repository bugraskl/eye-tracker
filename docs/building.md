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
uv run python scripts/fetch_models.py --check     # the face models are committed; verify their SHA-256
uv run python scripts/make_icons.py               # .ico / .icns / .png from the app's own icon code
uv run pyinstaller packaging/pyinstaller/eye-tracker.spec --noconfirm --clean
```

The output lands in `dist/`:

| Platform | Output |
|---|---|
| Windows | `dist/EyeTracker/EyeTracker.exe` (tray app, no console) and `eye-tracker-cli.exe` (command line) |
| macOS | `dist/Eye Tracker.app` (menu-bar app) with `Contents/MacOS/eye-tracker-cli` |
| Linux | `dist/eye-tracker/eye-tracker` (one executable for the app and the command line) |

Then run the privacy gate on the bundle. It fails if any bundled library contains telemetry markers
or unexpected networking code:

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
registers start at sign-in. Upgrades keep your choice.

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
`NOTICE.md` (licences, upstream URLs and checksums). To re-download and re-extract them from their
upstream sources:

```bash
uv run python scripts/fetch_models.py
```

## Releasing

1. Update `__version__` in `src/eye_tracker/__init__.py` and move the **Unreleased** notes in
   `CHANGELOG.md` under a new `## [x.y.z] - YYYY-MM-DD` heading.
2. Commit, then tag and push: `git tag v0.1.0 && git push origin v0.1.0`.
3. The release workflow builds all platforms, smoke-tests each package, runs the privacy gate,
   writes `SHA256SUMS.txt`, attaches build-provenance attestations and publishes the GitHub release
   with the notes from `CHANGELOG.md`.
