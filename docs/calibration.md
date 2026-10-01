# Calibration guide

Calibration teaches Eye Tracker how *your* head and eyes move when you look at each of *your*
monitors. It takes about 15 seconds per monitor and you normally do it once per desk setup.

## Before you start

- **Sit the way you normally work.** Same chair height, same distance. The model learns your usual
  posture; leaning back later is fine, but calibrating while hunched forward is not.
- **Light your face from the front or the side.** A bright window behind you turns your face into
  a silhouette. Room lighting is enough; you don't need a ring light.
- **Put the webcam where it stays.** On top of a monitor, centred on your body if you can. Moving the
  camera later changes every angle it measures.
- **Glasses are fine.** Strong reflections on the lenses can hide the irises; if calibration grades
  poorly, tilt the camera or the lamp slightly.

## Running it

Start it from the tray menu (**Calibrate…**), with the hotkey, from **Settings → General →
Recalibrate…**, or from a terminal ([which command](../README.md#command-line)):

```bash
eye-tracker calibrate
```

1. Every monitor shows the instructions. Press **Space** to begin (**Esc** cancels at any time).
2. A dot appears. Look at it **the way you naturally would**. Turn your head as much as you
   normally do when you look at that spot. Don't hold your head artificially still, and don't
   exaggerate either.
3. The ring shrinks while your eyes settle, then fills while samples are collected. The next dot
   follows automatically; the dots walk across each monitor in turn. **Space** pauses and resumes
   the dots, **R** starts over.
4. At the end you get a grade and the accuracy per monitor. Press **Enter** to save, **R** to retry.

If *"Can't see your face — check the camera"* appears, the dots pause until your face is visible
again: check the camera direction and the lighting. After 60 seconds without a face the calibration
closes. A dot that gets too few usable samples (*"Keep looking at the dot…"*) is tried once more,
then skipped. While the window waits for a key (the instructions, the result, or paused with
**Space**), it closes after 2 minutes without one. Walk-away detection is off while the calibration
is open, which is why it never stays open unattended.

## Understanding the grade

The grade is honest: every dot is predicted by a model that was trained **without** that dot
(leave-one-point-out cross-validation), so it reflects how well your gaze is recognised at places
the model has not memorised.

| Grade | Monitor accuracy | What it means |
|---|---|---|
| Excellent | ≥ 97 % | Switching will feel instant and reliable. |
| Good | ≥ 90 % | Reliable. The dwell time filters out the occasional wrong frame. |
| Fair | ≥ 75 % | Works, with an occasional wrong or late switch. Improve the lighting and recalibrate. |
| Poor | < 75 % | Recalibrate: the camera probably cannot see your eyes well. |

## It keeps getting better

After calibration, Eye Tracker quietly learns from how you use the mouse: when you move the pointer
somewhere and stop, you are almost always looking there. These samples refine the calibration over
time (turn this off under **Settings → Switching → Learn from how I use the mouse**). **Keep at
most … samples** limits how many are kept; lowering it (0 forgets them all) applies to every saved
calibration the next time it is used. If the app notices that you often correct it by hand, it
suggests a recalibration. On Wayland nothing is learned: apps cannot read the pointer position
there.

## When to recalibrate

- You moved the webcam or a monitor, or changed your chair height noticeably.
- Your monitor arrangement changed. Eye Tracker keeps separate calibrations per arrangement, so
  switching between a home and an office dock does not need a recalibration once each has been
  calibrated.
- You switched to a different camera or to the other vision backend.
- The app suggests it.
- You calibrated with an earlier version and glances at your phone or keyboard still switch
  monitors: a new calibration enables the current looking-away detection.

## Tips for tricky setups

- **Laptop screen below an external monitor.** Works, but the vertical head movement is small. Make
  sure the camera sees your eyes when you look down at the laptop, and use the *facemesh* backend
  (default), which also uses iris position.
- **Three or more monitors.** Supported in any arrangement. Calibrate all of them together.
- **Very large or curved single monitors.** Switching needs at least two monitors, but walk-away
  lock, privacy mode and the shoulder guard work with one.
