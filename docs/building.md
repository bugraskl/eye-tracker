# Building from source

Releases are built on GitHub's runners whenever a `v*` tag is pushed, by the steps in
[`.github/workflows/build.yml`](../.github/workflows/build.yml) (see
[Continuous integration](#continuous-integration)). You can run the same steps locally. Every build
must be made on the operating system it targets: PyInstaller does not cross-compile.

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

The build fails if the output contains a symbolic link to a file that is not bundled: on macOS such
a link breaks signing. The spec drops the links to every library it leaves out (see below).

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

The allow-list (`BUNDLE_ALLOWLIST` in `scripts/check_privacy.py`) names each file that may contain
networking code, exactly what it may link or import, and why; the gate prints those as notices. A
library that the app does not need is left out by the spec instead of being allowed: Qt's network
plugins and what only they link, OpenSSL libraries that nothing imports, and on macOS every library
that no bundled binary binds a symbol to. OpenCV's macOS wheel ships FFmpeg as Homebrew builds it,
which links libraries it never uses (libX11, and libssl through libsrt); the spec makes those links
weak in the copies it bundles and leaves the libraries out, together with the symbolic links that
PyInstaller made to them.

Finally, smoke-test the command-line tool as CI does. The script checks the version, that the build
is frozen with every face model present and verified, that both backends analyse an image, that
the bundled FFmpeg decodes an AVI and an MP4 file (written on the spot with the build environment's
OpenCV), and that camera probing runs (AVFoundation on macOS). It writes only to a temporary folder:

```bash
uv run python scripts/smoke_test_bundle.py --version 0.2.0 dist/EyeTracker/eye-tracker-cli.exe
uv run python scripts/smoke_test_bundle.py --version 0.2.0 "dist/Eye Tracker.app/Contents/MacOS/eye-tracker-cli"
uv run python scripts/smoke_test_bundle.py --version 0.2.0 dist/eye-tracker/eye-tracker
```

## Packages

**Windows installer and portable ZIP**

```powershell
iscc /DAppVersion=0.2.0 packaging\windows\installer.iss   # -> dist\EyeTracker-0.2.0-windows-x64-setup.exe
Compress-Archive dist\EyeTracker dist\EyeTracker-0.2.0-windows-x64-portable.zip
```

The installer is per user (no administrator rights), creates a Start menu shortcut, and optionally
registers start at sign-in. It also installs the command-line tool a second time as
`eye-tracker.exe` and, with the task *Add the "eye-tracker" command to PATH* (on by default;
`/MERGETASKS=!addtopath` turns it off in a silent install), adds the installation folder to the
user's PATH; uninstalling removes that entry. A user PATH that is not stored as a string value
(some tools write it as `REG_MULTI_SZ`) cannot be read, and is left as it is. Upgrades keep your
choices, and a silent upgrade (winget, `/SILENT`, `/VERYSILENT`) restarts Eye Tracker in the
background if it had to close it.

Start at sign-in is never turned back on by an upgrade: it stays as you left it in the app or in
Task Manager (which can disable it without removing it), whatever the first installation chose. An
explicit choice wins, also in a silent upgrade: `/MERGETASKS=startup` turns it on (and clears Task
Manager's "disabled" flag), `/MERGETASKS=!startup` turns it off, and `/TASKS=...` turns it on only
when it lists `startup`.

**macOS disk image**

```bash
bash packaging/macos/make_dmg.sh --app "dist/Eye Tracker.app" --version 0.2.0 \
  --out dist/EyeTracker-0.2.0-macos-arm64.dmg
```

The script refuses an app that contains a symbolic link to a missing file, and names the links,
before it signs anything; rebuild the app with the spec. The app is signed ad hoc unless you
provide a certificate. With a stable identity (a Developer ID,
or a self-signed certificate as described at the top of `make_dmg.sh`), the Camera and Accessibility
permissions survive updates. In CI, set the repository secrets `MACOS_SIGNING_CERT_P12`,
`MACOS_SIGNING_CERT_PASSWORD` and optionally `MACOS_SIGNING_IDENTITY`.

**Linux AppImage and tarball**

```bash
bash packaging/linux/build_appimage.sh --version 0.2.0
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
- [`build.yml`](../.github/workflows/build.yml) is the one definition of how the packages are
  built and tested. It is a reusable workflow, never run on its own: the Bundle check and the
  release both call it, so a pull request checks exactly what a release ships. It reads the version
  and writes the release notes, then builds each platform on its own runner (`windows-latest`,
  `macos-14`, and `ubuntu-22.04` for the oldest supported glibc):
  - **every platform:** PyInstaller, the bundle privacy gate, the packages, and
    `scripts/smoke_test_bundle.py` on each package the way users get it;
  - **Windows:** the bundle (the portable ZIP's contents), then the installer, which a disposable
    runner installs, upgrades and uninstalls: start at sign-in after silent upgrades, with and
    without `/MERGETASKS=startup` or `!startup` and with Task Manager's "disabled" flag, the PATH
    entry (also with an unreadable PATH value), and the installed copy smoke-tested through the
    `eye-tracker` command;
  - **macOS:** the minimum macOS version, signing (with the optional certificate) and the disk
    image; the app is then checked inside the mounted image: no symbolic link without a target,
    `codesign --verify --deep --strict`, the Info.plist, and the smoke test of its
    `eye-tracker-cli`;
  - **Linux:** that `libxcb-cursor` is bundled and GTK and GIO are not, then the AppImage (smoke
    test) and the tarball (its links and its launcher).
- [`bundle.yml`](../.github/workflows/bundle.yml) runs `build.yml` whenever a pull request or a push
  to `main` changes the app (`src/`), `packaging/`, the privacy gate, the model, icon or smoke-test
  scripts, `pyproject.toml`, `uv.lock` or the release workflows. It passes no secret (the macOS app
  is signed ad hoc), keeps no artifact and publishes nothing. Start it by hand on any branch with
  `gh workflow run bundle.yml --ref <branch>`.
- [`release.yml`](../.github/workflows/release.yml) runs `build.yml` for a `v*` tag, or by hand to
  try a release without publishing it; it keeps the packages and the release notes as workflow
  artifacts. Only for a tag does its publishing job (the only one with write access) compute
  `SHA256SUMS.txt`, attach build-provenance attestations and publish the GitHub release.

Every workflow has read-only permissions unless a job needs more, uses actions pinned to commit
SHAs (kept current by Dependabot), and runs in a concurrency group: a newer push cancels an older
check, while release runs never cancel each other.

## Website

The project website, https://bugraskl.github.io/eye-tracker/, is built from [`site/`](../site/)
(an English and a Turkish page sharing one stylesheet and script). It loads nothing from other
sites: no web fonts, no analytics, no cookies. The download buttons are filled in from the latest
GitHub release at build time, so [`pages.yml`](../.github/workflows/pages.yml) rebuilds and
deploys it on every published release and on changes to the site.

To build and preview it locally:

```bash
gh api repos/bugraskl/eye-tracker/releases/latest > release.json   # optional
uv run python scripts/build_site.py --release release.json --out _site
uv run python -m http.server 8000 --directory _site
```

## Releasing

1. Update `__version__` in `src/eye_tracker/__init__.py` and move the **Unreleased** notes in
   `CHANGELOG.md` under a new `## [x.y.z] - YYYY-MM-DD` heading, leaving `## [Unreleased]` empty.
2. Commit, then tag and push: `git tag v0.2.0 && git push origin v0.2.0`.
3. The release workflow builds and tests every package as described above and publishes the GitHub
   release with the notes from `CHANGELOG.md`. It fails before building anything when the tag does
   not match `__version__`, when `CHANGELOG.md` has no section for the version, or when notes are
   still under **Unreleased** (they would be missing from the release notes). A run that is not
   for a tag only reports the two `CHANGELOG.md` problems. The oldest macOS the DMG runs on follows from the bundled wheels
   (`MACOS_MINIMUM` in `build.yml`, currently 14.0); the smoke test fails when it changes, so that
   the README, these docs and the release notes are updated with it.
