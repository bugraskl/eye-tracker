"""Decision engine: turns observations and input activity into actions.

Everything in this package except :mod:`~eye_tracker.engine.controller` is pure,
deterministic logic (standard library and numpy only, no Qt, no OpenCV). Time
is always injected as ``now`` (``time.monotonic()`` seconds), which keeps the
modules trivially unit-testable:

* :mod:`.decision` - which monitor to switch to, and when (dwell, hysteresis,
  off-screen rejection, mouse/typing/cooldown guards).
* :mod:`.presence` - walk-away detection with a cancellable countdown.
* :mod:`.guard` - shoulder-surfer detection (a second face that persists).
* :mod:`.scheduler` - adaptive camera frame rate (the main CPU lever).
* :mod:`.input_state` - mouse vs keyboard activity without input hooks.
* :mod:`.window_memory` - last cursor position and window per monitor.
* :mod:`.camera_yield` - when to release the camera for other apps.
* :mod:`.controller` - the Qt orchestrator wiring all of the above together.
"""
