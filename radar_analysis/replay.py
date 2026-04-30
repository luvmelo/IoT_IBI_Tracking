"""Thread-safe ring buffer + replay engine driving the dashboard.

`ReplayEngine` consumes a `FrameSource` on a background thread and pushes
each frame into `FrameRing`. UI / pipeline code reads via `snapshot(n)`,
which copies out the last N frames in chronological order — never sharing
the internal buffer.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from radar_analysis.streams import FrameSource


class FrameRing:
    """Pre-allocated complex64 ring buffer.

    `append` is O(1); `snapshot(n)` is O(n) (one copy). All access is
    serialized through a `threading.Lock`, so the dashboard's recompute
    thread and the replay thread can interleave safely.
    """

    def __init__(self, capacity: int, n_chirps: int, n_range_bins: int, n_rx: int):
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = int(capacity)
        self._buf = np.zeros(
            (self._capacity, n_chirps, n_range_bins, n_rx), dtype=np.complex64
        )
        self._head = 0          # next write index
        self._n_appended = 0    # monotonic; not modulo
        self._lock = threading.Lock()

    def append(self, frame: np.ndarray) -> None:
        if frame.shape != self._buf.shape[1:]:
            raise ValueError(
                f"frame shape {frame.shape} does not match ring slot shape {self._buf.shape[1:]}"
            )
        with self._lock:
            self._buf[self._head] = frame
            self._head = (self._head + 1) % self._capacity
            self._n_appended += 1

    def snapshot(self, n_recent: int) -> np.ndarray:
        """Return up to `n_recent` most recent frames, oldest-first. Always a copy."""
        if n_recent < 0:
            raise ValueError(f"n_recent must be >= 0, got {n_recent}")
        with self._lock:
            available = min(self._n_appended, self._capacity)
            n = min(n_recent, available)
            if n == 0:
                return np.zeros((0,) + self._buf.shape[1:], dtype=self._buf.dtype)
            end = self._head
            start = (end - n) % self._capacity
            if start < end:
                return self._buf[start:end].copy()
            return np.concatenate([self._buf[start:], self._buf[:end]]).copy()

    @property
    def n_appended(self) -> int:
        with self._lock:
            return self._n_appended

    @property
    def capacity(self) -> int:
        return self._capacity

    def reset(self) -> None:
        with self._lock:
            self._head = 0
            self._n_appended = 0
            self._buf[:] = 0


class ReplayEngine:
    """Drains a `FrameSource` into a `FrameRing` on a daemon thread."""

    def __init__(self, source: FrameSource, ring: FrameRing) -> None:
        self._source = source
        self._ring = ring
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._pause = threading.Event()
        self._exhausted = threading.Event()
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._pause.clear()
            self._exhausted.clear()
            self._thread = threading.Thread(target=self._run, daemon=True, name="ReplayEngine")
            self._thread.start()

    def _run(self) -> None:
        try:
            for frame in self._source:
                if self._stop.is_set():
                    return
                # Cooperative pause: hold the frame until resumed or stopped. We
                # sleep on `_stop.wait` so stop() returns within ~50 ms.
                while self._pause.is_set():
                    if self._stop.wait(0.05):
                        return
                if self._stop.is_set():
                    return
                self._ring.append(frame)
        finally:
            self._exhausted.set()

    def pause(self) -> None:
        self._pause.set()

    def resume(self) -> None:
        self._pause.clear()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._source.close()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    @property
    def is_paused(self) -> bool:
        return self._pause.is_set()

    @property
    def is_exhausted(self) -> bool:
        return self._exhausted.is_set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()
