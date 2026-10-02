# Privacy

Eye Tracker watches your face all day, so it is built to be trustworthy by construction, and to be
checkable by anyone.

## The promises

| Promise | How it is kept | How you can check |
|---|---|---|
| **Nothing leaves your computer.** | Apart from the update check you can turn on (next section), there is no networking code in the application. The only socket it listens on is a local, per-user IPC channel (a named pipe on Windows, a Unix socket elsewhere) used by `eye-tracker ctl`, and it accepts only your own user account. On Linux the app also talks to your desktop over the local D-Bus and X11 sockets. None of them reaches the network. | `scripts/check_privacy.py` fails CI on any networking import or call, with one reviewed exception: `update/winhttp.py`, which it reports on every run. Every release build, and every pull request that changes the packaging or the dependencies, is scanned as well: every bundled native library and every bundled Python module. Run a firewall or `strace -f -e trace=connect` yourself. |
| **Frames are never stored.** | Camera frames live in memory for one analysis step. No image or video is written anywhere. | The privacy check fails CI on `imwrite`, `imencode` and `VideoWriter` under `src/`; other image APIs (such as Qt's `QImage.save`) are left to code review. |
| **Only numbers are kept.** | The calibration file holds head angles, iris ratios and the screen points they map to. | Open `calibration.json`; it is plain JSON. |
| **The camera is really off when it says so.** | Privacy mode, pause and the locked screen *release* the camera device (the webcam light goes out); they do not just ignore frames. Privacy mode stays on after Eye Tracker or the computer restarts, also after an update, until you turn it off, so the camera never comes back on by itself (**Keep privacy mode on after a restart** in Settings → Presence & privacy). | Watch the webcam LED. |
| **No telemetry and no accounts. Update checks only if you turn them on.** | There is nothing to phone home to. The update check is off by default and is the only thing that can use the network ([below](#the-update-check-opt-in)). | See the first row. |

## Why not the MediaPipe runtime

The face-landmark network comes from Google's MediaPipe Face Landmarker (Apache-2.0). The
MediaPipe *Python package* was used during development, until a review found that its native library
(`libmediapipe`) contains a usage-logging client that opens a connection to `play.googleapis.com`
whenever a landmarker is closed, with no way to switch it off.

Eye Tracker therefore runs the same network with OpenCV's DNN module, which it already needs for
camera capture. On a reference photo the landmark output matches MediaPipe's to within 0.2 % of the
image size (`tests/test_facemesh.py` compares them when that photo is supplied), and the MediaPipe
runtime is not installed, imported or shipped.
[`src/eye_tracker/vision/models/NOTICE.md`](../src/eye_tracker/vision/models/NOTICE.md) lists the
exact model files, their licences and checksums; the licence texts ship next to the models.

## The update check (opt in)

Eye Tracker can look for a newer version of itself. It is **off by default**, so the app never
touches the network unless you ask. Turn it on in **Settings → General → Updates** (or
`updates.check` in [`settings.json`](configuration.md#updates)); it is available on Windows only
for now. **Check for updates…** in the tray menu looks once, when you click it, with the setting off.

**What is sent.** One HTTPS `GET` to
`https://api.github.com/repos/bugraskl/eye-tracker/releases/latest`: once a day when the setting is
on (the first one about a minute and a half after the app starts), and when you click **Check for
updates…**. The request carries a `User-Agent` that names the app and its version
(`EyeTracker/0.2.1 (+https://github.com/bugraskl/eye-tracker)`) and an `Accept` header. Nothing
else: no identifier, no cookie, no settings, no calibration, no information about your computer.
GitHub sees your IP address, as any web server does.

**What it does with the answer.** It reads the version number and the names of the release's
files. If the version is newer, the tray menu says **Update to X.Y.Z…** and you get one notification
per version. Nothing is downloaded until you press **Install and restart**.

**Installing.** Only for a copy installed with the setup program (the portable ZIP, macOS and Linux
get the release page instead). It downloads that release's `…-windows-x64-setup.exe` and
`SHA256SUMS.txt` from `github.com` (GitHub redirects the file to its own download hosts), checks
the file's size and SHA-256, and only then starts the setup program silently, the way
`winget upgrade` would: it closes Eye Tracker, replaces its files in your profile (no administrator
rights) and starts it again. A file that fails a check is deleted. Downloads live in the `updates`
folder next to your data and are cleared at the next start.

**What it may talk to.** The check refuses every address that is not `https://` on `api.github.com`,
`github.com`, `objects.githubusercontent.com` or `release-assets.githubusercontent.com`, and checks
every redirect hop the same way. Each request has a size limit (1 MB for the answer, 64 KB for the
checksums, the announced size for the installer), and the installer's address must be exactly the
one GitHub derives from the repository, the release tag and the file name.

**How it connects.** Through Windows' own WinHTTP, from one module
([`update/winhttp.py`](../src/eye_tracker/update/winhttp.py)), which is the only exception in
`scripts/check_privacy.py` and is printed in its output on every run. Windows checks the
certificates against its root store and the proxy settings are the system's. No OpenSSL or Python
`ssl` is added to the app.

**What it cannot promise.** The checksum list sits on the same release as the installer, so it
guards against a damaged or truncated download, not against someone who can replace both files on
GitHub. Every release also carries a Sigstore build attestation that you can check yourself with
`gh attestation verify <file> --repo bugraskl/eye-tracker`; the app does not run that check. The
installer is not code-signed yet (see [platform notes](platform-support.md)).

**What it keeps.** `updates.json` in the data folder: the time of the last check and the version
you were told about, so that it asks at most once a day and tells you once.

**See it yourself.** Leave the setting off and run a firewall; or turn it on and watch that
`EyeTracker.exe` only ever contacts the hosts above.

## Networking code inside the libraries it ships

Apart from that update check, Eye Tracker's own code contains no networking code, but some
libraries it ships do, because they can do more than Eye Tracker asks of them:

- Qt's network module provides the local IPC channel; its TCP, TLS and HTTP parts are never used, and
  Qt's network plugins are not shipped.
- OpenCV reads video files with FFmpeg, which can also open network streams. Eye Tracker opens only
  local files: it refuses URLs and protocol prefixes for every video source, and FFmpeg lets a local
  file (a playlist, for example) refer only to local data. On macOS, OpenCV's package contains FFmpeg
  as Homebrew builds it, with libraries for SRT, RIST, SFTP, ZeroMQ and TLS streams and an OCR
  library that can download images; none of them is ever used. Libraries that nothing in the app
  uses at all, such as the X11 client library and OpenSSL's TLS library, are left out of the macOS
  app.
- Python's socket module is there because the standard library and psutil use it, and pyobjc can
  convert network addresses for macOS APIs that Eye Tracker does not call.

The privacy gate lists every such file, with what it links and why that is acceptable, in the log
of every release build, and fails the build when a library gains networking code that has not been
reviewed.

## What is stored, and where

| File | Contents | Location |
|---|---|---|
| `settings.json` | Your preferences | Windows `%LOCALAPPDATA%\bugraskl\eye-tracker\`, macOS `~/Library/Application Support/eye-tracker/`, Linux `~/.config/eye-tracker/` |
| `calibration.json` | Calibration profiles: numeric feature vectors, target points, the fitted model, and learned samples from mouse use | Windows `%LOCALAPPDATA%\bugraskl\eye-tracker\`, macOS `~/Library/Application Support/eye-tracker/`, Linux `~/.local/share/eye-tracker/` |
| `state.json` | Whether privacy mode is on, so that it stays on after a restart | Next to `calibration.json` |
| `eye-tracker.log` | Events such as "switched to monitor 2" or "camera opened". No window titles, no key contents, no images. Rotated at 1 MB × 3. | The platform log directory; `eye-tracker doctor` prints the exact paths |

The Windows portable ZIP uses the same folders. Started with `--config-dir DIR`, Eye Tracker keeps
everything in `DIR` instead (the calibration in `DIR/data`, the logs in `DIR/logs`).

`eye-tracker reset --all` deletes the settings and the calibration. On Windows the uninstaller
offers to delete the settings, calibration and logs (the default keeps them); elsewhere, delete the
folders yourself if you want them gone.

The optional `--trace FILE` switch writes numeric tracking data for tuning and bug reports: for
every analysed frame the face features, head angles, gaze estimate, mouse pointer position and
switching decision, with a timestamp (seconds since an arbitrary start, not the time of day). It is
off unless you pass it, and it never contains images, but it does record where your pointer was:
share a short one.

## What the app does *not* do

- It does not read what you type. Typing is detected from the operating system's idle timer: if the
  timer resets while the mouse did not move, you pressed something.
- It does not read window contents or titles. To restore focus it remembers opaque window handles in
  memory for the current session. To tell a terminal from an editor it reads the window's program
  name and window class (bundle id on macOS), never its title.
- With the experimental [split-pane focus](panes.md) on, it asks tmux and WezTerm where their panes
  are through their own command-line tools (`tmux`, `wezterm cli`). Those talk to the multiplexer
  over its local socket on your computer (for tmux in WSL, inside WSL); nothing reaches the network.
  Windows Terminal is asked through UI Automation (Windows' accessibility interface), on your
  computer as well, and only for the position, focus and id of its terminal panes, never their text.
  Only pane positions, sizes and ids are read: pane titles, commands, working directories and
  contents are never stored, logged, traced or shown in `eye-tracker ctl status`. Electron and
  Chromium apps are never asked anything, except the Claude and ChatGPT desktop apps when you turn
  on `panes.desktop_apps` (off by default): then their windows are asked through UI Automation for
  the type, class name and rectangle of the elements on the way to their side-by-side sessions, and
  the process of the element with the keyboard focus. Names and text of elements, and so your messages,
  are never read.
- It does not identify you. It detects *a* face and where it looks; it does not recognise whose face
  it is.

## Shoulder guard

The shoulder guard is off by default; turn it on under **Settings → Presence & privacy → React when
someone looks over my shoulder**. It counts faces in the camera image. It never stores or recognises
them. When a second face stays in view for two seconds, it shows a privacy curtain, sends a
notification or locks the screen, whichever you chose.

Because faces are only counted, never recognised:

- The curtain stays up while only the other person's face is left (you got up, they stayed), also
  when the camera misses their face for a moment. It lifts once your face is the one in view again,
  or when someone uses the keyboard or mouse. **Esc** or **Dismiss** takes it down at any time.
- Walk-away detection counts every face in view as you, so the guard never makes Eye Tracker believe
  that you left: only its own reaction (**Lock the computer**, when a second face appears) can lock
  the screen. The other way round, if you leave while another person stays in view, walk-away
  detection does not lock the computer; the curtain stays up instead.
- With **Lock the computer**, once you unlock a lock the guard caused, it covers the screens instead
  of locking again, until nobody has looked over your shoulder for 5 minutes.
- If the guard takes you for the person who left (you moved to another spot while a closer face hid
  yours), a new second face still gets its reaction: the curtain, notification or lock does not
  wait for your next keystroke.

Like walk-away detection, the guard needs the camera: it pauses while tracking is paused, in privacy
mode, during calibration and while another app has the camera.

## Reporting a privacy problem

Please report anything that contradicts this page privately, as described in
[SECURITY.md](../SECURITY.md).
