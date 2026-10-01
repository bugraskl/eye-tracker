# Security policy

Eye Tracker processes a live camera feed and can lock your session, so security and privacy
reports are taken seriously and handled before feature work.

## Supported versions

Only the latest release receives fixes. Please update before reporting.

| Version | Supported |
|---|---|
| 0.1.x (latest) | ✅ |
| older | ❌ |

## Reporting a vulnerability

**Do not open a public issue for security problems.**

Use GitHub's private reporting instead: **Security → Report a vulnerability** on
[bugraskl/eye-tracker](https://github.com/bugraskl/eye-tracker/security/advisories/new).

Please include:

- the affected version (`eye-tracker --version`) and operating system,
- the output of `eye-tracker doctor` (it contains no camera images or personal data),
- steps to reproduce and the impact you observed.

For `eye-tracker`, type the command of your package (see
[the command line](README.md#command-line)).

You can expect an acknowledgement within **72 hours** and a status update at least weekly until the
issue is resolved. Fixed issues are credited in the release notes unless you prefer otherwise.

## What counts as a security issue

- Any path by which camera frames, face data or calibration data leave the machine or are written
  to disk.
- Any network connection made by the application (it is designed to make none).
- Bypassing privacy mode (camera stays on or keeps being read while privacy mode is active).
- The local control socket (`eye-tracker ctl`) accepting commands from another user account.
- Walk-away lock failing *silently* while the tray shows it as active.
- Tampered or unverifiable release artifacts (every release ships `SHA256SUMS.txt`).

Accuracy problems (switching to the wrong monitor, false walk-away warnings) are regular bugs; please
use the issue tracker for those.

## Design safeguards

- **No network code, checked twice.** `scripts/check_privacy.py` fails CI if any networking API
  (sockets, HTTP clients, Qt network classes, asyncio connections, ctypes loads of network
  libraries) is used under `src/`, or if frame-writing APIs (`imwrite`, `VideoWriter`, `imencode`)
  appear. `check_privacy.py --bundle` then scans every native library and every bundled Python
  module of every release build (and of pull requests that change the packaging or the
  dependencies) for telemetry endpoints, networking packages and plugins, and networking imports,
  against a reviewed allow-list with a reason per entry. Telemetry markers are never allow-listed.
- **No MediaPipe runtime.** Its native library contains a usage-logging uploader, so the face model
  runs through OpenCV instead (see [docs/privacy.md](docs/privacy.md)).
- **Frames stay in memory.** Only numeric features (head angles, iris ratios) are stored in the
  calibration file; never images.
- **Least privilege.** The app runs as the current user, needs no administrator rights, and uses
  OS idle timers instead of keyboard hooks to detect typing.
- **A private control channel.** `eye-tracker ctl` talks to the running app over a per-user Unix
  socket or named pipe. On Windows the account of every connecting process is checked, and
  connections from other accounts are refused before anything is read.
