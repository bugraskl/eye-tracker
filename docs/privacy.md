# Privacy

Eye Tracker watches your face all day, so it is built to be trustworthy by construction, and to be
checkable by anyone.

## The promises

| Promise | How it is kept | How you can check |
|---|---|---|
| **Nothing leaves your computer.** | There is no networking code in the application. The only socket it listens on is a local, per-user IPC channel (a named pipe on Windows, a Unix socket elsewhere) used by `eye-tracker ctl`, and it accepts only your own user account. On Linux the app also talks to your desktop over the local D-Bus and X11 sockets. None of them reaches the network. | `scripts/check_privacy.py` fails CI on any networking import. Every release build, and every pull request that changes the packaging or the dependencies, is scanned as well: every bundled native library and every bundled Python module. Run a firewall or `strace -f -e trace=connect` yourself. |
| **Frames are never stored.** | Camera frames live in memory for one analysis step. No image or video is written anywhere. | The privacy check fails CI on `imwrite`, `imencode` and `VideoWriter` under `src/`; other image APIs (such as Qt's `QImage.save`) are left to code review. |
| **Only numbers are kept.** | The calibration file holds head angles, iris ratios and the screen points they map to. | Open `calibration.json`; it is plain JSON. |
| **The camera is really off when it says so.** | Privacy mode, pause and the locked screen *release* the camera device (the webcam light goes out); they do not just ignore frames. Privacy mode stays on after Eye Tracker or the computer restarts, also after an update, until you turn it off, so the camera never comes back on by itself (**Keep privacy mode on after a restart** in Settings → Presence & privacy). | Watch the webcam LED. |
| **No telemetry, no accounts, no update checks.** | There is nothing to phone home to. | See the first row. |

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

## Networking code inside the libraries it ships

Eye Tracker's own code contains no networking code, but some libraries it ships do, because they can
do more than Eye Tracker asks of them:

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
  memory for the current session.
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
