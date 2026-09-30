"""OpenCV thread-pool policy for the vision pipeline.

OpenCV parallelises most calls (colour conversion, resizing, DNN layers) over a
process-wide worker pool with one thread per core. For the small images this app
handles the extra threads barely shorten a call but spin while they wait for
work, which multiplies the CPU time. Measured on a 16-thread desktop (OpenCV
5.0, single frames paced like the worker), the face landmark network took
12.5 ms wall / 10.9 ms CPU with one thread and 9.6 ms wall / 84 ms CPU with 16;
YuNet detection at 320 px took 5.8 ms / 3.9 ms against 3.3 ms / 24.5 ms.

So the pool is capped at a single thread: a few milliseconds of latency buy an
up to eightfold CPU saving, which is what matters for an app that runs all day.
"""

from __future__ import annotations

import logging

import cv2

log = logging.getLogger(__name__)

#: Upper bound for OpenCV's worker pool while the vision pipeline runs.
MAX_OPENCV_THREADS = 1


def limit_opencv_threads(max_threads: int = MAX_OPENCV_THREADS) -> int:
    """Cap OpenCV's process-wide thread pool at ``max_threads``; return the new size.

    Never raises the limit (the host may have lowered it on purpose) and is
    cheap enough to call whenever a vision component starts.
    """
    limit = max(1, int(max_threads))
    try:
        current = cv2.getNumThreads()
        if current > limit:
            cv2.setNumThreads(limit)
            log.debug("OpenCV thread pool capped at %d (was %d)", limit, current)
        return int(cv2.getNumThreads())
    except cv2.error:  # a build without a parallel framework: nothing to cap
        log.debug("Could not cap the OpenCV thread pool", exc_info=True)
        return 1
