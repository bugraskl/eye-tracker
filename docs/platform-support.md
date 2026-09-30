# Platform support

Eye Tracker runs on Windows, macOS and Linux. Every feature is *best effort*: when an operating
system does not allow something (Wayland forbids moving the pointer, for example), that feature is
reported as unavailable and everything else keeps working. `eye-tracker doctor` shows the exact
feature matrix for your machine.

## Feature matrix

| Feature | Windows 10/11 | macOS 13+ | Linux X11 | Linux Wayland |
|---|:---:|:---:|:---:|:---:|
| Cursor follows your gaze | ✅ | ✅ | ✅ | ⚠️ sway / Hyprland exact; others via `ydotool` |
| Keyboard focus follows | ✅ | ✅ needs Accessibility | ✅ | ❌ not allowed by Wayland |
| Walk-away lock | ✅ | ✅ | ✅ | ✅ |
| Displays off / wake | ✅ | ✅ | ✅ DPMS | ✅ GNOME, KDE, sway, Hyprland |
| Typing detection (idle timer) | ✅ | ✅ keyboard-precise | ✅ | ✅ GNOME |
| Pause while the session is locked | ✅ | ✅ | ✅ | ✅ |
| Release the camera when another app needs it | ✅ automatic | ➖ use the app list | ✅ automatic¹ | ✅ automatic¹ |
| Global hotkeys | ✅ | ✅ | ✅ | ➖ bind `eye-tracker ctl` in your desktop |
| Start at login | ✅ | ✅ | ✅ | ✅ |
| Tray icon | ✅ | ✅ menu bar | ✅ | ✅ (StatusNotifier) |

¹ Apps that capture through PipeWire's camera portal (some Flatpak apps) cannot be seen. Add them
to **Settings → Presence & privacy → Pause while these apps run**.

## Downloads

| Platform | Package | Notes |
|---|---|---|
| Windows 10/11 x64 | `EyeTracker-<version>-windows-x64-setup.exe` | Per-user install, no administrator rights. A portable ZIP is also published. |
| macOS 13+ (Apple silicon) | `EyeTracker-<version>-macos-arm64.dmg` | Intel Macs: [run from source](../README.md#run-from-source). |
| Linux x86_64 | `EyeTracker-<version>-linux-x86_64.AppImage` | glibc 2.35+ (Ubuntu 22.04, Debian 12, Fedora 36 or newer). A tarball is also published. |

## Windows

- **Keyboard focus.** Windows restricts which process may bring a window to the front. Eye Tracker
  uses the documented techniques (thread input attachment, then an empty input event) and never
  sends a click. Windows running *as administrator* cannot be focused by a normal process; the
  cursor still moves.
- **Hotkeys.** On many keyboard layouts AltGr is reported as Ctrl+Alt, so Ctrl+Alt+letter would
  swallow characters such as `₺` (Turkish Q) or `ć` (Polish). The defaults are therefore
  **Ctrl+Alt+Win+T / P / C**, and any hotkey that types a character on one of your installed layouts
  is refused with an explanation.
- **Camera sharing.** When another app starts using the camera (Teams, Zoom, the Camera app),
  Windows records it; Eye Tracker notices within three seconds, releases the camera and resumes
  when the other app is done.
- **SmartScreen.** Releases are not code-signed yet, so SmartScreen may warn on first launch. Choose
  *More info → Run anyway*, and verify the download against `SHA256SUMS.txt` from the release.

## macOS

- **Permissions.** On first launch macOS asks for **Camera** access. For keyboard focus to follow
  your gaze, also allow **Accessibility** in *System Settings → Privacy & Security → Accessibility*.
  Without it, the cursor still moves.
- **Updates and Accessibility.** Release builds are signed ad hoc, so macOS treats every update as a
  new app and silently ignores the old Accessibility grant. After updating, remove Eye Tracker from
  the Accessibility list with **−** and add it again. The app detects this situation and tells you.
- **Gatekeeper.** The app is not notarised. The first time, right-click the app and choose **Open**,
  or run `xattr -dr com.apple.quarantine "/Applications/Eye Tracker.app"`.
- **Move it to Applications first.** Start at login is refused while the app runs from the disk
  image or a quarantined location, because that path disappears after ejecting.
- **Hotkeys** are Carbon hotkeys (**⌃⌥T / ⌃⌥P / ⌃⌥C**), which need no extra permission.
- **App Nap** is disabled while tracking so timers are not throttled, and re-enabled while paused.

## Linux

**X11** gets the full feature set: pointer warping, window focus through EWMH, idle time from the
XScreenSaver extension, DPMS display power, and global hotkeys through key grabs. The default
hotkeys are **Ctrl+Alt+Shift+T / P / C** (Ctrl+Alt+T opens a terminal on most desktops).

**Wayland** deliberately forbids apps from moving the pointer, reading other windows or grabbing
keys. On Wayland:

- The cursor moves exactly through the compositor's own IPC on **sway** (`swaymsg`) and
  **Hyprland** (`hyprctl`). Elsewhere it needs **ydotool 1.x** with `ydotoold` running. Because
  ydotool emulates absolute moves with relative motion, set a *flat* pointer-acceleration profile for
  its virtual device, or the pointer lands short.
- Keyboard focus cannot follow your gaze.
- Global hotkeys are not available. Bind these commands in your desktop's keyboard settings instead:

  ```bash
  eye-tracker ctl toggle            # pause / resume tracking
  eye-tracker ctl privacy-toggle    # privacy mode
  eye-tracker ctl calibrate         # start calibration
  ```

- Presence, walk-away lock, privacy mode and the shoulder guard work normally.

**Locking.** Eye Tracker asks logind to lock the session and confirms that a locker reacted. On
desktops without a lock handler (a bare tiling window manager without `xss-lock`) it tries
`xdg-screensaver`, `dm-tool` and other tools in turn, and tells you if nothing locked the screen.

**AppImage dependencies.** The AppImage bundles its libraries, including `libxcb-cursor`. If Qt still
cannot find a platform plugin, install your distribution's `libxcb-cursor0` (Debian/Ubuntu) or
`xcb-util-cursor` (Fedora, Arch) package.
