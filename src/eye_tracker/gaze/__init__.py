"""Gaze estimation: from vision features to a point on the virtual desktop.

Pure numpy (no Qt, no OpenCV), so everything here is deterministic and unit-testable.

* :mod:`.model` - ridge regression with polynomial terms, cross-validated model selection.
* :mod:`.filters` - One Euro smoothing of the gaze point.
* :mod:`.calibration` - target plan, sample collection state machine, quality report.
* :mod:`.store` - calibration file format (numbers only, never images).
* :mod:`.learning` - refinement from natural mouse use and drift detection.
"""
