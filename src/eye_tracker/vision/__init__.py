"""Camera capture and face analysis.

``camera`` provides frame sources (webcams and files), ``motion`` a cheap gate that
skips analysis while the picture is unchanged, ``backends`` turns frames into
:class:`~eye_tracker.types.Observation` objects and ``worker`` runs the capture and
analysis loop on a background thread. Frames only ever live in memory.
"""
