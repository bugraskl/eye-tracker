# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-01

### Added

- Glance-to-switch: the cursor (and optionally keyboard focus) moves to the monitor you look at,
  using head pose and iris position from any webcam.
- Per-monitor memory: the cursor returns to where you left it on each screen and the last-used
  window on that screen gets keyboard focus, without synthetic clicks.
- Guards against accidental switches: dwell time, hysteresis at the bezel, typing and mouse grace
  periods, cooldown, and off-screen glances (phone, desk) are ignored.
- Guided calibration for any number of monitors in any arrangement, with a quality grade and
  cross-validated accuracy.
- Adaptive learning from natural mouse use and drift alerts that suggest recalibration.
- Walk-away detection: lock the session and/or switch displays off after a cancellable countdown,
  and wake the displays when you return.
- Privacy mode that fully releases the camera (hotkey and tray), automatic pause while the session
  is locked or another app uses the camera, and an app list that pauses tracking.
- Shoulder-surfer guard that reacts when a second face appears behind you.
- Adaptive frame rate and a motion gate for very low CPU use; Eco, Balanced and Responsive profiles.
- Global hotkeys on Windows, macOS and X11, plus `eye-tracker ctl` for binding shortcuts on Wayland.
- Start at login on Windows, macOS and Linux.
- `eye-tracker doctor` diagnostics and `eye-tracker bench` performance measurement.
- Windows installer and portable ZIP, macOS DMG (Apple silicon), Linux AppImage and tarball.

[Unreleased]: https://github.com/bugraskl/eye-tracker/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/bugraskl/eye-tracker/releases/tag/v0.1.0
