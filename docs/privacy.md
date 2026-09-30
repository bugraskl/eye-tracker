# Privacy

Eye Tracker watches your face all day, so it is built to be trustworthy by construction, and to be
checkable by anyone.

## The promises

| Promise | How it is kept | How you can check |
|---|---|---|
| **Nothing leaves your computer.** | There is no networking code in the application. The only socket is a local, per-user IPC channel (a named pipe on Windows, a Unix socket elsewhere) used by `eye-tracker ctl`. | `scripts/check_privacy.py` fails CI on any networking import. The release pipeline also scans the bundled native libraries of every build. Run a firewall or `strace -f -e trace=connect` yourself. |
| **Frames are never stored.** | Camera frames live in memory for one analysis step. No image or video is written anywhere. | The privacy check fails CI on `imwrite`, `imencode` and `VideoWriter` under `src/`. |
| **Only numbers are kept.** | The calibration file holds head angles, iris ratios and the screen points they map to. | Open `calibration.json`; it is plain JSON. |
| **The camera is really off when it says so.** | Privacy mode, pause and the locked screen *release* the camera device (the webcam light goes out); they do not just ignore frames. | Watch the webcam LED. |
| **No telemetry, no accounts, no updates checks.** | There is nothing to phone home to. | See the first row. |

## Why not the MediaPipe runtime

The face-landmark network comes from Google's MediaPipe Face Landmarker (Apache-2.0). The
MediaPipe *Python package* was used during development, until a review found that its native library
(`libmediapipe`) contains a usage-logging client that opens a connection to `play.googleapis.com`
whenever a landmarker is closed, with no way to switch it off.

Eye Tracker therefore runs the same network with OpenCV's DNN module, which it already needs for
camera capture. The landmark output matches MediaPipe's to within 0.2 % of the image size, and
the MediaPipe runtime is not installed, imported or shipped. [`src/eye_tracker/vision/models/NOTICE.md`](../src/eye_tracker/vision/models/NOTICE.md)
lists the exact model files, their licences and checksums.

## What is stored, and where

| File | Contents | Location |
|---|---|---|
| `settings.json` | Your preferences | Windows `%LOCALAPPDATA%\bugraskl\eye-tracker\`, macOS `~/Library/Application Support/eye-tracker/`, Linux `~/.config/eye-tracker/` |
| `calibration.json` | Calibration profiles: numeric feature vectors, target points, the fitted model, and learned samples from mouse use | Windows `%LOCALAPPDATA%\bugraskl\eye-tracker\`, macOS `~/Library/Application Support/eye-tracker/`, Linux `~/.local/share/eye-tracker/` |
| `eye-tracker.log` | Events such as "switched to monitor 2" or "camera opened". No window titles, no key contents, no images. Rotated at 1 MB × 3. | The platform log directory; `eye-tracker doctor` prints the exact paths |

`eye-tracker reset --all` deletes the settings and calibration. Uninstalling does not touch these
files, so delete the folders if you want them gone.

The optional `--trace FILE` switch writes numeric tracking data (features, gaze estimates, decisions)
for tuning and bug reports. It is off unless you pass it, and it never contains images.

## What the app does *not* do

- It does not read what you type. Typing is detected from the operating system's idle timer: if the
  timer resets while the mouse did not move, you pressed something.
- It does not read window contents or titles. To restore focus it remembers opaque window handles in
  memory for the current session.
- It does not identify you. It detects *a* face and where it looks; it does not recognise whose face
  it is.

## Shoulder guard

When enabled, the shoulder guard counts faces in the camera image. It never stores or recognises
them. When a second face stays in view for two seconds, it shows a privacy curtain, sends a
notification or locks the screen, whichever you chose.

## Reporting a privacy problem

Please report anything that contradicts this page privately, as described in
[SECURITY.md](../SECURITY.md).
