# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Project website at https://bugraskl.github.io/eye-tracker/ in English and Turkish, rebuilt
  with the download links of every release.
- Split-pane focus (experimental, off by default): looking at another pane of a tmux, WezTerm or
  Windows Terminal window on the monitor you work on gives that pane the keyboard focus, through the
  tools' own command-line interfaces (tmux also inside WSL) or, for Windows Terminal, UI Automation
  (while it is the foreground window). Only panes several times larger than your
  calibration's gaze error take part; the typing, reading and mouse graces apply, and Electron and
  Chromium apps are never inspected. Turn it on with **Follow split panes** in the tray menu or in
  **Settings → Switching**; see [split-pane focus](docs/panes.md).
- Split-pane focus for the Claude and ChatGPT desktop apps on Windows (opt-in with
  `panes.desktop_apps` or **Settings → Switching → Split panes**, off by default): with two Claude
  chat sessions side by side, or a ChatGPT or Codex conversation next to its side chat or side
  panel, the one you look at gets the keyboard focus in its message box. The app's accessibility tree is
  walked through UI Automation in a few dozen bounded steps (never more than 400), reading only
  element types, class names and rectangles. Asking for that tree makes the app build it, which can
  cost it some CPU and memory while the setting is on. Every other Chromium and Electron app stays
  on the deny-list.
- The calibration now measures the gaze error on each monitor across and down; calibrations made
  with earlier versions get it from their saved samples when it is first needed.
- `eye-tracker ctl status` reports split-pane focus in a `panes` block, and `--trace` records the
  pane decisions (pane ids only).

### Fixed

- Windows: started from a terminal inside a packaged (MSIX) app, Eye Tracker took its own use of the
  camera for another app's and released the camera every few seconds. It now also recognises its
  interpreter under the redirected path that Windows records.

## [0.1.0] - 2026-10-01

### Added

- Glance-to-switch: the cursor (and optionally keyboard focus) moves to the monitor you look at,
  using head pose and iris position from any webcam. MediaPipe's face-landmark model runs through
  OpenCV's DNN module; the MediaPipe runtime, which contains a usage logger, is not used.
- Per-monitor memory: the cursor returns to where you left it on each screen and the last-used
  window on that screen gets keyboard focus, without synthetic clicks.
- Guards against accidental switches: dwell time, hysteresis at the bezel, typing, mouse and
  reading grace periods, and a cooldown. Glances at your phone or desk are recognised from the
  calibration's estimate of where your head and eyes point, and ignored, also on desks that mix
  monitors of different pixel density (a 4K panel next to a 1080p one).
- Robust face tracking: posture shifts, large upward ones included, are caught up within the same
  frame, strongly tilted heads are found, and a poster or a colleague picked up while you were away
  gives way to you once you are back.
- Guided calibration for any number of monitors in any arrangement, with a quality grade and
  cross-validated accuracy, and one profile per monitor layout, camera and backend. A setup that
  is not calibrated yet is reminded of it, also when Eye Tracker starts at login; one monitor, or
  switching turned off, needs no calibration and gets no prompts.
- Adaptive learning from natural mouse use and drift alerts that suggest recalibration.
- Walk-away detection: lock the session and/or switch the displays off after a cancellable
  countdown, and wake the displays when you return. Until the setup assistant is finished, walking
  away only shows a notification.
- Privacy mode that fully releases the camera (hotkey and tray) and stays on across restarts and
  updates until you turn it off; automatic pause while the session is locked, while another app
  uses the camera (Windows and Linux) and while apps from a list run.
- An opt-in shoulder-surfer guard that covers the screens, notifies or locks when a second face
  appears behind you. Faces are counted, never recognised, and the guard never decides whether you
  are at the computer. After you unlock a lock it caused, it covers the screens instead of locking
  you out again while the colleague next to you is still around.
- Adaptive frame rate and a motion gate for very low CPU use; Eco, Balanced and Responsive profiles.
- Global hotkeys on Windows (Ctrl+Alt+Win+T/P/C), macOS (⌃⌥⌘T/P/C) and X11 (Ctrl+Alt+Super+T/P/C).
  A combination that would type a character, that a keyboard-layout option makes impossible to
  press, or that another app owns is reported instead of failing silently. The pause and privacy
  hotkeys confirm with a notification which way they switched. On Wayland, `eye-tracker ctl`
  commands can be bound to desktop shortcuts.
- Start at login on Windows, macOS and Linux.
- `eye-tracker doctor` diagnostics that are safe to paste into a public issue (the home folder is
  shown as `~`), including hotkey registration, the installed Linux lock tools and the command of
  your copy; `eye-tracker bench` performance measurement.
- Windows installer (per user; optionally puts an `eye-tracker` command on the PATH, which starts
  the app on its own rather than inside the terminal, and restarts the app after a silent upgrade)
  and portable ZIP, macOS DMG (macOS 14 or later, Apple silicon), Linux AppImage and tarball
  (without Qt's GTK theme and its VNC, WebGL and framebuffer platform plugins).
- A privacy gate that fails CI on networking or frame-writing code and scans every bundled native
  library and Python module of the release builds.

[Unreleased]: https://github.com/bugraskl/eye-tracker/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bugraskl/eye-tracker/releases/tag/v0.1.0
