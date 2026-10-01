# Platform support

Eye Tracker runs on Windows, macOS and Linux. Every feature is *best effort*: when an operating
system does not allow something (Wayland forbids moving the pointer, for example), that feature is
reported as unavailable and everything else keeps working. `eye-tracker doctor` shows the exact
feature matrix for your machine. Commands on this page are written as `eye-tracker`; see
[the command line](../README.md#command-line) for what to type with your package.

## Feature matrix

| Feature | Windows 10/11 | macOS 14+ | Linux X11 | Linux Wayland |
|---|:---:|:---:|:---:|:---:|
| Cursor follows your gaze | ✅ | ✅ | ✅ | ⚠️ sway / Hyprland exact; others via `ydotool` |
| Cursor returns to where you left it on a monitor | ✅ | ✅ | ✅ | ❌ lands in the middle² |
| Keyboard focus follows | ✅ | ✅ needs Accessibility | ✅ | ❌ not allowed by Wayland |
| Learns from your mouse use | ✅ | ✅ | ✅ | ❌² |
| Walk-away lock | ✅ | ✅ | ✅ | ✅ |
| Displays off / wake | ✅ | ✅ | ✅ DPMS | ✅ GNOME, KDE, sway, Hyprland |
| Typing detection (idle timer) | ✅ | ✅ keyboard-precise | ✅ | ✅ GNOME only³ |
| Pause while the session is locked | ✅ | ✅ | ✅ | ✅ |
| Release the camera when another app needs it | ✅ automatic | ➖ use the app list | ✅ automatic¹ | ✅ automatic¹ |
| Global hotkeys | ✅ | ✅ | ✅ | ➖ bind `eye-tracker ctl` in your desktop |
| Start at login | ✅ | ✅ | ✅ | ✅ |
| Tray icon | ✅ | ✅ menu bar | ✅ | ✅ (StatusNotifier) |

¹ Apps that capture through PipeWire's camera portal (some Flatpak apps) cannot be seen. Add them
to **Settings → Presence & privacy → Pause while these apps run**.

² Wayland does not let apps read the pointer position; see [Wayland](#wayland).

³ Elsewhere on Wayland, keyboard and mouse use are not reported, so only the camera tells whether
you are there; see [Wayland](#wayland).

## Downloads

| Platform | Package | Notes |
|---|---|---|
| Windows 10/11 x64 | `EyeTracker-<version>-windows-x64-setup.exe` | Per-user install, no administrator rights. A portable ZIP is also published. |
| macOS 14+ (Apple silicon) | `EyeTracker-<version>-macos-arm64.dmg` | Intel Macs: [run from source](../README.md#run-from-source). |
| Linux x86_64 | `EyeTracker-<version>-linux-x86_64.AppImage` | glibc 2.35+ (Ubuntu 22.04, Debian 12, Fedora 36 or newer). A tarball is also published. |

## Windows

- **Keyboard focus.** Windows restricts which process may bring a window to the front. Eye Tracker
  uses the documented techniques (thread input attachment, then an empty mouse input event, and as
  a last resort a tap of the Alt key, masked so that it opens no menu) and never sends a click.
  Windows running *as administrator* cannot be focused by a normal process; the cursor still moves.
- **Hotkeys.** On many keyboard layouts AltGr is reported as Ctrl+Alt, so Ctrl+Alt+letter would
  swallow characters such as `₺` (Turkish Q) or `ć` (Polish). The defaults are therefore
  **Ctrl+Alt+Win+T / P / C**, and any hotkey that types a character on one of your installed layouts
  is refused with an explanation. A combination that another app has already registered is
  reported as "… is already in use by another application".
- **Command line.** The installer adds `eye-tracker.exe`, the command-line tool, next to the app.
  With the task *Add the "eye-tracker" command to PATH* (ticked by default) it also puts the
  installation folder on your user PATH, so `eye-tracker doctor` works in any terminal you open
  afterwards; uninstalling removes that entry again. For silent installs, `/MERGETASKS=!addtopath`
  installs without the PATH entry (and removes one that an earlier installation added). The
  portable ZIP has `eye-tracker-cli.exe` instead.
- **Your data.** Settings, calibration and logs live in `%LOCALAPPDATA%\bugraskl\eye-tracker`, also
  for the portable ZIP; start it with `--config-dir FOLDER` to keep everything in that folder. The
  uninstaller asks whether to delete them (the default keeps them).
- **Upgrades.** A silent upgrade (winget, `/SILENT`, `/VERYSILENT`) restarts Eye Tracker in the
  background if the installer had to close it. An interactive upgrade offers the *Launch* checkbox
  instead.
- **Camera sharing.** When another app starts using the camera (Teams, Zoom, the Camera app),
  Windows records it; Eye Tracker notices within three seconds, releases the camera and resumes
  when the other app is done.
- **Shared PCs.** The command channel that `eye-tracker ctl` talks to accepts only your own user
  account. If another account created a channel with the same name first, Eye Tracker still runs,
  walk-away lock included, but `eye-tracker ctl` and desktop shortcuts cannot reach it; the log
  says so.
- **SmartScreen.** Releases are not code-signed yet, so SmartScreen may warn on first launch. Choose
  *More info → Run anyway*, and verify the download against `SHA256SUMS.txt` from the release.

## macOS

- **Permissions.** On first launch macOS asks for **Camera** access. For keyboard focus to follow
  your gaze, also allow **Accessibility** in *System Settings → Privacy & Security → Accessibility*.
  Without it, the cursor still moves.
- **Updates and Accessibility.** Unless the release notes say otherwise, release builds are
  signed ad hoc, so macOS treats every update as a new app and silently ignores the old
  Accessibility grant. After updating, remove Eye Tracker from the Accessibility list with **−** and
  add it again. The app detects this situation and tells you.
- **Accessibility and the command line.** For `eye-tracker-cli`, macOS judges the Accessibility
  permission of the program that started it (your terminal), not the app's. `eye-tracker-cli doctor`
  therefore reports it as unknown; **Settings → Diagnostics** in the app shows the app's own state.
- **Gatekeeper.** The app is not notarised, so macOS blocks its first launch. Open
  *System Settings → Privacy & Security*, scroll to the message about Eye Tracker and click
  **Open Anyway** (on macOS 14, right-clicking the app and choosing **Open** also works), or run
  `xattr -dr com.apple.quarantine "/Applications/Eye Tracker.app"`.
- **Move it to Applications first.** Start at login is refused while the app runs from the disk
  image or a quarantined location, because that path disappears after ejecting.
- **Start at login** is a launch agent in `~/Library/LaunchAgents`. If launchd has it switched off
  (`launchctl disable`; the switch under *System Settings → General → Login Items* may do the
  same), `eye-tracker autostart status` and the *Start at login* checkbox show it as off. Turning it
  on again clears that switch, or, if macOS still blocks it, asks you to allow Eye Tracker under
  *Login Items* ("Allow in the Background").
- **Hotkeys** are Carbon hotkeys (**⌃⌥⌘T / ⌃⌥⌘P / ⌃⌥⌘C**), which need no extra permission. They are
  registered exclusively: a combination that another app (Rectangle, Magnet, …) already holds is
  reported as "Hotkey unavailable: … is already in use by another application" instead of firing in
  both apps.
- **App Nap** is disabled while tracking so timers are not throttled, and re-enabled while paused.

## Linux

**X11** gets the full feature set: pointer warping, window focus through EWMH, idle time from the
XScreenSaver extension, DPMS display power, and global hotkeys through key grabs. The default
hotkeys are **Ctrl+Alt+Super+T / P / C**: Ctrl+Alt+T opens a terminal on most desktops, and
Ctrl+Alt+Shift cannot be pressed at all where Alt+Shift or Ctrl+Shift switches the keyboard layout
(it is also a JetBrains IDE shortcut). A hotkey that one of your keyboard options makes impossible
to press is reported as unavailable with the option's name instead of silently never firing, for
example "Ctrl+Alt+Shift+T includes Alt+Shift, which switches the keyboard layout (XKB option
grp:alt_shift_toggle)". The defaults themselves collide only with rare options such as
`grp:ctrl_alt_toggle`, `grp:lwin_toggle`, `grp:win_menu_select`, `lv3:win_switch` or
`altwin:ctrl_win`. When the options change while Eye Tracker runs, the hotkeys are checked again: one
that stops working is named in the log and by `eye-tracker doctor`, and one that could not be
registered starts working as soon as the option is removed, without a restart.

Desktops that open a menu with the Super key on its own (Xubuntu's Xfce binds Super to the Whisker
menu) take the whole keyboard while Super is held, so **press Ctrl and Alt before Super** there;
with Super pressed first the shortcut does nothing. Eye Tracker detects such a binding and says so
in **Settings → Hotkeys** and in `eye-tracker doctor`.

### Wayland

Wayland deliberately forbids apps from moving the pointer, reading its position, reading other
windows or grabbing keys. Eye Tracker runs through XWayland where it can. On Wayland:

- The cursor moves exactly through the compositor's own IPC on **sway** (`swaymsg`) and
  **Hyprland** (`hyprctl`). Elsewhere it needs **ydotool 1.x** with `ydotoold` running. Because
  ydotool emulates absolute moves with relative motion, set a *flat* pointer-acceleration profile for
  its virtual device, or the pointer lands short.
- The pointer position cannot be read: the cursor does not return to where you left it on a
  monitor (it lands in the middle), and mouse use does not refine the calibration.
- Keyboard focus cannot follow your gaze.
- Outside GNOME, Wayland reports no keyboard or mouse activity (only pointer moves over X11
  windows are seen). Typing therefore does not hold switching back, and walk-away detection relies
  on the camera alone: only looking at the camera cancels the countdown, and the countdown says so.
- Global hotkeys are not available. Bind these commands in your desktop's keyboard settings
  instead, written the way your package is run (an AppImage by its file name, see
  [the command line](../README.md#command-line); **Settings → Hotkeys** shows them for your copy):

  ```bash
  eye-tracker ctl toggle            # pause / resume tracking
  eye-tracker ctl privacy-toggle    # privacy mode
  eye-tracker ctl calibrate         # start calibration
  ```

- Presence, walk-away lock, privacy mode and the shoulder guard work, within the limits above.

### Locking

Eye Tracker asks logind to lock the session (`loginctl lock-session`) and checks that the screen
really locked. On GNOME and KDE every lock is confirmed through logind's *LockedHint*: GNOME ignores
lock requests while locking is disabled by policy, and that is reported instead of taken for a
lock. Cinnamon, MATE, Xfce and Budgie (which lists GNOME only as a fallback) lock on logind's
request too, but their lock screens do not all report the hint, so `loginctl` is trusted there.
Elsewhere `loginctl` only counts once a screen locker (i3lock, swaylock, hyprlock, slock, …) is
running, because nothing may listen to logind there (a bare tiling window manager without
`xss-lock` or a swayidle `lock` hook); then `xdg-screensaver`, `dm-tool` and other tools are tried
in turn. `dm-tool` (the LightDM greeter) is never put on top of a desktop's own lock screen. Each
check waits up to 2 seconds. If nothing locked the screen, you get a "Could not lock the screen"
notification instead of a silent failure. sway, Hyprland and similar sessions that set
`XDG_CURRENT_DESKTOP=GNOME` or `KDE` are recognised by their own sockets.

`eye-tracker doctor` lists the lock tools that are installed, in the order they are tried
(*lock_methods*). Whether one locks your session is only known when it is tried, so test it once
with a short **After … s away**.

### Camera hand-off

Apps that open `/dev/video*` are noticed (fanotify on Linux 5.13 and newer also sees refused
attempts; otherwise inotify), and the cameras are polled every 3 seconds if those notifications
stop working. Apps that use the PipeWire camera portal are not seen: add them to **Pause while
these apps run** (for example `zoom`, `teams-for-linux`, `skypeforlinux`, `obs`).

### Packages

The AppImage bundles its libraries, including `libxcb-cursor`. If Qt still cannot find a platform
plugin, install your distribution's `libxcb-cursor0` (Debian/Ubuntu) or `xcb-util-cursor` (Fedora,
Arch) package. The packages leave out Qt's GTK platform theme (Qt then uses its own GNOME and KDE
styles or the desktop portal) and the VNC, WebGL and framebuffer platform plugins, so
`QT_QPA_PLATFORM=vnc`, `eglfs` or `linuxfb` does not work with them.
