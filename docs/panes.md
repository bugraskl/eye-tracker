# Split-pane focus (experimental)

Eye Tracker moves the cursor and the keyboard focus to the monitor you look at. With split-pane
focus on, it goes one step further: when you look at another pane of the focused terminal window,
on the monitor you are already working on, that pane gets the keyboard focus. Nothing is typed or
clicked for you: Eye Tracker asks the terminal itself to focus the pane, through its own
command-line tool or, for Windows Terminal, through UI Automation. The cursor stays where it is.

It is **off by default**. Turn it on with **Follow split panes** in the tray menu, under
**Settings → Switching → Split panes (experimental)**, or with `panes.enabled` in
[`settings.json`](configuration.md#split-panes-experimental).

## Supported tools

| Tool | How | Notes |
|---|---|---|
| tmux | `tmux list-clients`, `list-panes`, `select-pane` | In Windows Terminal, WezTerm, Alacritty, kitty, GNOME Terminal, Konsole, xterm, iTerm2, Terminal, foot, Ghostty, Tilix and the Xfce terminal. The tmux client running in the focused window is found through the window's processes. On Windows, tmux inside WSL is used when the window runs WSL and exactly one tmux client is attached there. Only the default tmux server is asked. |
| WezTerm | `wezterm cli --no-auto-start list`, `list-clients`, `activate-pane` | WezTerm's own split panes, in the tab on screen. Never starts a WezTerm server. |
| Windows Terminal | UI Automation: the `TermControl` elements of the window, `SetFocus` | Windows only. Windows Terminal's own split panes, in the tab on screen. Which pane is active can only be seen while Windows Terminal is the foreground window, so its panes are asked for (and focused) only then. With a single Windows Terminal pane, tmux running in it is followed instead; tmux panes inside one of several Windows Terminal panes are not followed yet. |
| Claude desktop app | UI Automation: the session panes of the window, `SetFocus` on a session's message box | Windows only, **opt-in** (`panes.desktop_apps`, see [below](#claude-and-chatgpt-desktop-app-sessions)). Two chat sessions side by side. |
| ChatGPT desktop app (also Codex) | UI Automation: the main conversation and the side chat or side panel, `SetFocus` on its message box | Windows only, **opt-in** (`panes.desktop_apps`, same section). |
| VS Code, Cursor, JetBrains IDEs | planned | Editors need purpose-built support (see below). |

Supported platforms: Windows, macOS and Linux on X11. Wayland does not tell applications which
window has the focus, so split-pane focus is not available there. On macOS the window frame stands
in for its content area, so panes are placed a title bar's height too high; on all platforms a tab
bar the terminal draws inside its window shifts tmux panes by its height. Both are small compared
with the panes split-pane focus takes part in (next section).

Electron and Chromium applications (VS Code, Cursor, the Claude and ChatGPT apps, Chrome, Edge,
Firefox, Slack, Teams, Discord, Obsidian, and on Windows every window of the class
`Chrome_WidgetWin_1`) are **never inspected**, not even to ask whether they have panes: asking them
about their contents can switch them into a slower screen-reader mode. The exceptions are the
Claude and ChatGPT desktop apps, and only when you turn them on (next section). A zoomed pane (tmux `Ctrl+B z`,
WezTerm's and Windows Terminal's zoom) fills the window, so nothing happens while one is zoomed.

### Claude and ChatGPT desktop app sessions

The Claude desktop app for Windows can show two chat sessions side by side. The ChatGPT desktop app
for Windows, which also hosts Codex, can show the main conversation next to a side chat or side
panel. With `panes.desktop_apps` on (**Settings → Switching → Split panes → Also switch between
side-by-side sessions in the Claude and ChatGPT apps**), they are followed like terminal panes:
look at the other session and its message box gets the keyboard focus, so you can type there at
once. It is **off by default**, also when split-pane focus is on, because of what it costs the app:

- Eye Tracker reads the app's accessibility tree through UI Automation. Chromium builds that tree
  only once someone asks for it and then keeps it up to date, which can cost the app some CPU and
  memory while the setting is on (the app may keep the tree until it restarts). Right after you turn it
  on, the first look may find no sessions yet; they are found a second later.
- Only these two apps are opened up this way, each matched by its program name (`claude.exe`,
  `ChatGPT.exe`) *and* its window class. VS Code, Cursor, Antigravity, Windsurf, browsers and every
  other Chromium or Electron app stay on the deny-list whatever the setting.
- The tree is never walked as a whole. From the window, Eye Tracker steps down through the native
  views to the web content, then only through groups (at most 8 levels), and stops at the session
  panes without looking inside: in Claude the groups marked as panes, in ChatGPT the parts of the
  main content area (parts smaller than 200 × 200 px, such as pop-ups, do not count). The sidebar is
  skipped. On trees like the apps' that is 15 to 25 steps, and never more than 400. To focus a
  session, it looks for the message box inside that one pane, from the bottom up, again at most 400
  steps. Element names are not used (they depend on the app's language).
- Of the elements on the way only the control type is read, plus the class name of groups and
  editors and the automation id of the web-content document; of the session panes their rectangle,
  visibility and runtime id; of an editor whether it can take the focus. Names, text and messages
  are never read. The session that has the focus is the one containing the element with the
  keyboard focus (its process and rectangle are read).
- As with Windows Terminal, sessions are looked at and focused only while the app's window is the
  foreground window.

## How it decides

Panes are much smaller than monitors, and the gaze estimate is the same, so split-pane focus is
deliberately cautious. Everything is measured against how accurate *your* calibration is: when you
calibrate, Eye Tracker measures its gaze error on each monitor, separately across (x) and down (y),
on dots it has not learned from (the 75th percentile of the error, see
[calibration](calibration.md)). Calibrations made with earlier versions get the same measurement
from their saved samples the first time it is needed.

1. **Same monitor only.** Pane focus is considered only while the gaze is on the monitor the cursor
   is on, and not until `panes.after_monitor_switch_ms` (1.5 s) after a monitor switch.
2. **Large enough panes only.** For two panes side by side the new pane must be at least
   `panes.precision` (2.5) times your horizontal gaze error wide; for stacked panes, that many times
   your vertical error tall; and never less than `panes.min_pane_px` (240 px). With a horizontal
   error of 100 px, panes side by side take part from 250 px wide. Smaller panes are simply left
   alone: a pane the gaze cannot reliably be placed in never gets the focus. Raise the precision
   for fewer surprises, lower it to include smaller panes.
3. **Past the divider.** The gaze must be `panes.hysteresis` (half of your gaze error) past the
   divider between the two panes.
4. **A steady look.** For `panes.dwell_ms` (0.4 s), at least 80 % of the gaze samples must be on the
   new pane. A stray sample does not restart the wait.
5. **Not while you work.** No pane switch for 3 s after you type (`panes.typing_grace_ms`), for the
   mouse grace of the switching settings after you use the mouse, for 1 s after the previous pane
   switch. Switching panes yourself (a shortcut, a click) counts like typing: no pane switch
   for the same `panes.typing_grace_ms` after it, so your choice is not undone.
6. **Reading is not switching.** If you keep typing in one pane while looking at another (reading
   a log or a man page next to your editor), that other pane becomes a *reading pane*: reading pauses
   do not move the focus there until 8 s after your last keystroke (`panes.reading_grace_ms`). This
   is the same rule as for [reading another monitor](../README.md#features).

The panes of the focused window are asked for when it gets the focus and then about every second
while you look at it. A tool that keeps failing (three times in a row) is left alone until Eye
Tracker restarts. `eye-tracker ctl status` shows what is going on in its `panes` block: whether
the feature is on and supported, the tool in use, how many panes there are and how many are large
enough, the gaze error in use and how many pane switches happened.

## Privacy

tmux and WezTerm are asked through their command-line tools, which talk to the multiplexer over
its local socket on your computer (inside WSL for tmux there); nothing reaches the network. Windows
Terminal is asked through UI Automation, the Windows accessibility interface, inside your session;
only the bounding rectangle, keyboard focus, visibility and runtime id of its terminal controls are
read, never their text or names. For the Claude and ChatGPT desktop apps (opt-in), only the control type, class
name, rectangle, visibility and runtime id of the elements on the way to the session panes are
read, plus the process of the focused element; never names, text or messages. Only pane positions,
sizes and ids are read: pane titles,
commands, working directories and contents are never stored, logged or shown. See
[privacy](privacy.md#what-the-app-does-not-do).
