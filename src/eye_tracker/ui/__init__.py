"""Qt user interface (PySide6 Widgets).

The UI holds no business logic. Every widget talks to the
:class:`~eye_tracker.engine.controller.Controller` through its public methods and
signals, so each piece can be exercised in tests with a small fake controller:

* :mod:`.tray` - system-tray icon and menu, tooltip and notifications.
* :mod:`.icons` - every icon, drawn with ``QPainter`` at any size (no image files).
* :mod:`.overlay` - optional click-through gaze dot (a testing aid).
* :mod:`.countdown` - walk-away countdown toast.
* :mod:`.curtain` - shoulder-surfer privacy curtain.
* :mod:`.preview` - live camera preview (frames are shown there only, never saved).
* :mod:`.about` - about box with licences and the privacy statement.
* :mod:`.settings_dialog`, :mod:`.calibration_window`, :mod:`.wizard` - settings,
  calibration and first-run flows.
* :mod:`.util` - DPI scaling, screen lookup, palette and small formatting helpers.

Sizes that are painted by hand are multiplied by :func:`.util.ui_scale` so they
look the same on every display; font sizes are given in points, which Qt already
scales with the screen's DPI.
"""
