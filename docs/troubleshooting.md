# Troubleshooting

Start with the built-in report. It checks the camera, monitors, permissions, models and every
platform feature, and suggests fixes:

```bash
eye-tracker doctor
```

Commands on this page are written as `eye-tracker`. With a release package, type the command of
your package instead (for example the `.AppImage` file, or `eye-tracker-cli` inside the macOS
app): [the command line](../README.md#command-line) lists them, and the report shows yours under
*command_line*.

The same report is under **Settings → Diagnostics**, with a **Copy report** button for bug reports.
It contains no images and no personal data (your home folder is shown as `~`).

## The camera

**"Camera unavailable" or "The camera could not be opened".**
Another app may be holding it, or the operating system blocks access:

- Windows: *Settings → Privacy & security → Camera* must allow desktop apps.
- macOS: *System Settings → Privacy & Security → Camera* must list Eye Tracker as allowed.
- Linux: your user needs access to `/dev/video*` (usually the `video` group).

With several cameras, pick the right one under **Settings → Camera & performance → Device**
(**Detect cameras** lists them). `eye-tracker doctor --probe-cameras` tests cameras 0 to 3 (their
lights may flash).

**The webcam light stays off.** That is expected while tracking is paused, in privacy mode, while
the screen is locked, while another app uses the camera, and while an app from **Pause while these
apps run** is running. The tray tooltip shows the state. Walk-away detection and the shoulder guard
pause in these states too. Privacy mode stays on after a restart until you turn it off (a
notification says so when Eye Tracker starts in it); to have it forgotten at every restart, untick
**Keep privacy mode on after a restart** under Settings → Presence & privacy.

**"Camera appears covered".** Frames were almost completely dark. Open the shutter, or improve the
lighting. If you cover the camera while you are at the computer, walk-away detection pauses, but
only until 5 minutes pass without keyboard or mouse input; after that a covered camera counts as
"nobody here" and the walk-away timer runs as usual. To switch the camera off for longer, use
privacy mode. A camera that goes dark after you left (the light switched off) does not pause
anything.

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
*Good*: looking-away detection relies on the calibration's estimate of where your head and eyes
point together. A calibration made by an earlier version uses an older, less reliable test until
you recalibrate once (or adaptive learning refines it). Increasing *Look for at least* also helps.

**Focus jumps to my reference document while I'm copying from it.** If you type while looking at
another monitor, Eye Tracker waits longer (*… while reading another monitor*, 6 s by default) after
your last keystroke before it moves focus there. Increase that value if you pause longer.

**The cursor moves but keyboard focus does not follow.**

- Windows: the target window runs as administrator. Windows does not let normal apps focus those.
- macOS: grant **Accessibility** access. After updating the app, remove and re-add it there
  ([details](platform-support.md#macos)).
- Linux Wayland: not possible by design ([details](platform-support.md#wayland)).

**The cursor always lands in the middle of the monitor.** On Wayland, apps cannot read the pointer
position, so Eye Tracker cannot remember where you left it ([details](platform-support.md#wayland)).
Elsewhere, check **Cursor lands** under Settings → Switching.

## Walk-away lock

**It locked while I was sitting there.**
A countdown is shown first (10 s by default; a *Countdown* of 0 turns it off). Looking at the
camera cancels it, and so does keyboard or mouse use while **Keyboard or mouse use counts as being
here** is on (on Wayland outside GNOME only the camera can cancel it). If it still happens:

- Check the lighting and the camera angle in the preview. If your face is not detected while you
  read, walk-away detection sees an empty chair.
- Increase **After … s away** (Settings → Presence & privacy).
- Choose **Only show a notification** as the action while you find the cause.

**It does not lock when I leave.**

- Until the setup assistant is finished, walking away only shows a notification ("Are you still
  there?"), and the countdown says "Marking you as away". Finish it with **Run setup assistant…**
  under Settings → General.
- The time counts from when your face *and* your keyboard and mouse activity disappear. The action
  runs when **After … s away** has passed (45 s by default); the countdown takes its last seconds.
- Walk-away detection needs the camera, so it pauses while tracking is paused, in privacy mode,
  during calibration, while another app uses the camera (a call, for example) and while an app from
  **Pause while these apps run** is running.
- Any face in view counts as you, also a colleague, a photo or a face on a TV: the camera cannot
  tell faces apart.
- Check that **React when I leave the computer** is on and what **Then** is set to.
- On Linux, `eye-tracker doctor` lists the lock tools that are installed (*lock_methods*). Whether
  one locks your session is only known when it is tried; if nothing locks, you get a "Could not
  lock the screen" notification. Test it once with a short **After … s away**
  ([details](platform-support.md#locking)).

## Shoulder guard

**It never reacts.** The guard is off by default: turn on **React when someone looks over my
shoulder** (Settings → Presence & privacy). A second face must stay in view for **After … s**
(2 s by default); a face far behind you, or seen from the side, may not be detected. Like
walk-away detection, the guard pauses whenever the camera is released.

**The screens stay covered.** The curtain stays up while a second face is in view, and also while
only the other person's face is left (you got up, they stayed). It lifts once the face that remains
is yours again, or when someone uses the keyboard or mouse a little after you were judged gone.
**Esc** or **Dismiss** takes it down at any time, and so do pause and privacy mode.

**"Lock the computer" covered the screens instead.** After you unlock a lock that the guard caused,
it covers the screens instead of locking again until nobody has looked over your shoulder for 5
minutes; otherwise a colleague next to you would lock you out again every time they turn back to
the screen. If locking fails, the guard covers the screens as well.

## Hotkeys

**A hotkey does nothing.** When Eye Tracker cannot register a hotkey, it shows a "Hotkey
unavailable" notification with the reason, and writes it to the log. The usual reasons are another
app using the same combination, or a combination that types a character on one of your keyboard
layouts (AltGr); on Linux X11 also a keyboard option that switches the layout with keys of the
hotkey (Alt+Shift, Ctrl+Alt, the Win key, …). While Eye Tracker runs, `eye-tracker doctor` shows
each hotkey under *registration* ("registered", or "not registered:" and the reason) and flags
combinations that are invalid or type a character. Pick another in **Settings → Hotkeys**. On Linux
Wayland, global hotkeys are not available; bind `eye-tracker ctl` commands instead
([details](platform-support.md#wayland)). On Xubuntu (and other desktops that open a menu with the
Super key alone), press Ctrl and Alt before Super: Super pressed first goes to the menu.

**My hotkeys changed after an update.** Hotkeys that were still at an earlier version's defaults
(Ctrl+Alt+T/P/C, or Ctrl+Alt+Shift+T/P/C on Linux) move to today's defaults once. Hotkeys you chose
yourself are kept.

## Start at login

**It does not start at login.** Run `eye-tracker autostart status`.

- *stale*: the login entry points to a copy of the app that no longer exists (for example an old
  AppImage, or the app run from the macOS disk image). Start the app from its new location once and
  it repairs the entry, or toggle **Start at login** in the tray menu.
- *disabled* although you turned it on: Windows Task Manager (*Startup apps*), your desktop's
  autostart settings or macOS (*Login Items*, `launchctl disable`) switched it off. Turn it on
  there, or with **Start at login** in the tray menu.
- *other-profile*: the entry starts Eye Tracker with another `--config-dir`.

A start at login is quiet: no window opens (the setup assistant is offered as a notification), and
if Eye Tracker is already running, the new start exits without popping anything up.

## The tray icon

- Windows hides new tray icons in the overflow area (the **^** arrow). Drag the eye icon onto the
  taskbar to keep it visible.
- GNOME shows tray icons only with the *AppIndicator and KStatusNotifierItem Support* extension
  (preinstalled on Ubuntu).
- Without a tray, run `eye-tracker ctl settings` to open the settings.

## "Already running but does not respond"

Another instance is starting, stuck, or still shutting down. A new launch waits up to about 10
seconds for an instance that is shutting down and then takes its place, so starting Eye Tracker
again right after quitting works. If the message persists, end the old process in your task
manager. `eye-tracker ctl status` shows whether an instance answers.

**`eye-tracker ctl` says Eye Tracker is not running, but it is.** Each `--config-dir` profile is a
separate instance: give `ctl` the same `--config-dir`. On a shared Windows PC, another user account
may have taken the name of the command channel first; the log then says that `eye-tracker ctl`
cannot reach this instance.

## CPU use

Run `eye-tracker bench` to measure the analysis cost on your machine. To lower CPU use:

- choose the **Eco** profile (Settings → Camera & performance),
- keep **Skip analysis while the picture is still** on,
- close the camera preview window, which analyses many more frames while it is open (24 per second
  with the Balanced profile),
- or switch to the **lite** backend.

## Starting over

Quit Eye Tracker first; `reset` refuses to run while it is running.

```bash
eye-tracker reset --calibration   # forget all calibrations
eye-tracker reset --settings      # restore default settings
eye-tracker reset --all           # both
```

## Reporting a bug

Open an issue with the bug report template and paste the output of `eye-tracker doctor`. For
accuracy problems, a trace of a few minutes helps a lot. It contains numbers only, never images:
face features, head angles, gaze estimates, the mouse pointer position and the switching decisions,
with timestamps (seconds since an arbitrary start, not the time of day). `--trace` only applies when
Eye Tracker starts, so quit the running app first (tray menu → **Quit Eye Tracker**, or
`eye-tracker ctl quit`); otherwise the command says that it was not applied.

```bash
eye-tracker run --trace trace.jsonl
```

Logs are in the folder that `eye-tracker doctor` prints under *paths*. For more detail, set
**Settings → General → Log level** to *Debug* before you reproduce the problem.
