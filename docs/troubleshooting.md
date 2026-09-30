# Troubleshooting

Start with the built-in report. It checks the camera, monitors, permissions, models and every
platform feature, and suggests fixes:

```bash
eye-tracker doctor
```

The same report is under **Settings → Diagnostics**, with a **Copy** button for bug reports. It
contains no images and no personal data.

## The camera

**"Camera could not be opened".**
Another app may be holding it, or the operating system blocks access:

- Windows: *Settings → Privacy & security → Camera* must allow desktop apps.
- macOS: *System Settings → Privacy & Security → Camera* must list Eye Tracker as allowed.
- Linux: your user needs access to `/dev/video*` (usually the `video` group).

With several cameras, pick the right one under **Settings → Camera & performance → Device**
(**Detect cameras** lists them). `eye-tracker doctor --probe-cameras` tests each one.

**The webcam light stays off.** That is expected while tracking is paused, in privacy mode, while
the screen is locked, and while another app uses the camera. The tray tooltip shows the state.

**"Camera appears covered".** Frames were almost completely dark. Open the shutter, or improve the
lighting. While the camera is covered, walk-away detection cannot tell whether you are there, so it
does not lock.

## Accuracy

**It switches to the wrong monitor, or too late.**

1. Open **Camera preview…** from the tray and check that the face box, eyes and irises are tracked
   steadily. A backlit face (window behind you) is the most common cause of poor tracking.
2. Recalibrate, sitting the way you normally work ([calibration guide](calibration.md)). Aim for
   *Good* or *Excellent*.
3. Keep the default **facemesh** backend (it uses iris position; **lite** uses head turns only).
4. Tune **Settings → Switching**:
   - *Look for at least* (dwell): longer means fewer accidental switches.
   - *Cross into a monitor by* (hysteresis): larger means you must look further into the other
     monitor.
   - *Smoothing*: steadier means less jitter but slightly slower reactions.

Turn on **Show a dot where I am looking** (Settings → General) to see the live estimate.

**It switches when I glance at my phone or keyboard.** Make sure the calibration grade is at least
*Good*: looking-away detection relies on knowing the range of your normal head and eye movement.
Increasing *Look for at least* also helps.

**Focus jumps to my reference document while I'm copying from it.** If you type while looking at
another monitor, Eye Tracker waits longer (*… while reading another monitor*, 6 s by default) after
your last keystroke before it moves focus there. Increase that value if you pause longer.

**The cursor moves but keyboard focus does not follow.**

- Windows: the target window runs as administrator. Windows does not let normal apps focus those.
- macOS: grant **Accessibility** access. After updating the app, remove and re-add it there
  ([details](platform-support.md#macos)).
- Linux Wayland: not possible by design ([details](platform-support.md#linux)).

## Walk-away lock

**It locked while I was sitting there.**
A 10-second countdown is always shown first, and any keyboard or mouse input cancels it. If it still
happens:

- Check the lighting and the camera angle in the preview. If your face is not detected while you
  read, walk-away detection sees an empty chair.
- Increase **After … s away** (Settings → Presence & privacy).
- Choose **Only show a notification** as the action while you find the cause.

**It does not lock when I leave.** The timeout starts when your face *and* keyboard/mouse activity
disappear, so the lock comes after the configured time plus the countdown. Also check that
**React when I leave the computer** is on. On Linux, `eye-tracker doctor` shows which lock method
your desktop supports; if none reacts, you get a notification instead of a silent failure.

## Hotkeys

**A hotkey does nothing.** `eye-tracker doctor` lists every hotkey and why it could not be
registered. The usual reasons are another app using the same combination, or a combination that
types a character on one of your keyboard layouts (AltGr). Pick another in **Settings → Hotkeys**.
On Linux Wayland, global hotkeys are not available; bind `eye-tracker ctl` commands instead
([details](platform-support.md#linux)).

## Start at login

**It does not start at login.** Run `eye-tracker autostart status`. *Stale* means the login entry
points to a copy of the app that no longer exists (for example an old AppImage, or the app run from
the macOS disk image). Start the app from its new location once and it repairs the entry, or toggle
**Start at login** in the tray menu.

## The tray icon

- Windows hides new tray icons in the overflow area (the **^** arrow). Drag the eye icon onto the
  taskbar to keep it visible.
- GNOME shows tray icons only with the *AppIndicator and KStatusNotifierItem Support* extension
  (preinstalled on Ubuntu).
- Without a tray, run `eye-tracker ctl settings` to open the settings.

## "Already running but does not respond"

Another instance is starting or is stuck. Wait a few seconds and try again. If it persists, end the
old process in your task manager. `eye-tracker ctl status` shows whether an instance answers.

## CPU use

Run `eye-tracker bench` to measure the analysis cost on your machine. To lower CPU use:

- choose the **Eco** profile (Settings → Camera & performance),
- keep the **Skip unchanged frames** motion gate on,
- close the camera preview window, which analyses every frame while open,
- or switch to the **lite** backend.

## Starting over

```bash
eye-tracker reset --calibration   # forget all calibrations
eye-tracker reset --settings      # restore default settings
eye-tracker reset --all           # both
```

## Reporting a bug

Open an issue with the bug report template and paste the output of `eye-tracker doctor`. For
accuracy problems, a short trace helps a lot. It contains numbers only, never images:

```bash
eye-tracker run --trace trace.jsonl
```

Logs are in the folder that `eye-tracker doctor` prints under *paths*.
