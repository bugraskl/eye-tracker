# Privacy

Eye Tracker watches your face all day, so it is built to be trustworthy by construction, and to be
checkable by anyone.

## The promises

| Promise | How it is kept | How you can check |
|---|---|---|
| **Nothing leaves your computer.** | There is no networking code in the application. The only socket it listens on is a local, per-user IPC channel (a named pipe on Windows, a Unix socket elsewhere) used by `eye-tracker ctl`, and it accepts only your own user account. On Linux the app also talks to your desktop over the local D-Bus and X11 sockets. None of them reaches the network. | `scripts/check_privacy.py` fails CI on any networking import. Every release build, and every pull request that changes the packaging or the dependencies, is scanned as well: every bundled native library and every bundled Python module. Run a firewall or `strace -f -e trace=connect` yourself. |
| **Frames are never stored.** | Camera frames live in memory for one analysis step. No image or video is written anywhere. | The privacy check fails CI on `imwrite`, `imencode` and `VideoWriter` under `src/`; other image APIs (such as Qt's `QImage.save`) are left to code review. |
| **Only numbers are kept.** | The calibration file holds head angles, iris ratios and the screen points they map to. | Open `calibration.json`; it is plain JSON. |
| **The camera is really off when it says so.** | Privacy mode, pause and the locked screen *release* the camera device (the webcam light goes out); they do not just ignore frames. | Watch the webcam LED. |
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

## What is stored, and where

| File | Contents | Location |
|---|---|---|
| `settings.json` | Your preferences | Windows `%LOCALAPPDATA%\bugraskl\eye-tracker\`, macOS `~/Library/Application Support/eye-tracker/`, Linux `~/.config/eye-tracker/` |
| `calibration.json` | Calibration profiles: numeric feature vectors, target points, the fitted model, and learned samples from mouse use | Windows `%LOCALAPPDATA%\bugraskl\eye-tracker\`, macOS `~/Library/Application Support/eye-tracker/`, Linux `~/.local/share/eye-tracker/` |
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
  of locking again for 5 minutes, or until the second face has left.

Like walk-away detection, the guard needs the camera: it pauses while tracking is paused, in privacy
mode, during calibration and while another app has the camera.

## Reporting a privacy problem

Please report anything that contradicts this page privately, as described in
[SECURITY.md](../SECURITY.md).
