"""Server-side state singleton for the dashboard.

Owns the `FrameSource`, the ring buffer, the replay engine, and a
background recompute thread that re-runs `run_pipeline` over the most
recent `window_s` of frames every `recompute_period_s`. Dash callbacks
read `state.latest` (a `PipelineResult` or `None`) and never call the
pipeline themselves — keeping the UI loop snappy regardless of pipeline
runtime.

The recompute loop is single-threaded and synchronous: if the pipeline
takes longer than the budget, the next iteration just resyncs the clock
without piling up.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from radar_analysis.pipeline import PipelineResult, run_pipeline
from radar_analysis.replay import FrameRing, ReplayEngine
from radar_analysis.streams import FrameSource


@dataclass
class _Snapshot:
    """A point-in-time view the UI can read without holding the state lock."""

    result: PipelineResult | None
    n_appended: int
    is_paused: bool
    is_exhausted: bool
    elapsed_s: float


class DashboardState:
    """Single instance per dashboard process."""

    def __init__(
        self,
        source: FrameSource,
        *,
        window_s: float = 10.0,
        recompute_period_s: float = 0.25,
        plot_window_s: float = 10.0,
    ) -> None:
        self.fs_slow_hz = float(source.fs_slow_hz)
        self.range_res_m = float(source.range_res_m)
        self.window_frames = max(2, int(round(window_s * self.fs_slow_hz)))
        self._recompute_period_s = float(recompute_period_s)
        self.plot_window_s = float(plot_window_s)

        capacity = max(int(round(30.0 * self.fs_slow_hz)), self.window_frames * 2)
        self._ring = FrameRing(
            capacity=capacity,
            n_chirps=source.n_chirps,
            n_range_bins=source.n_range_bins,
            n_rx=source.n_rx,
        )
        self._engine = ReplayEngine(source, self._ring)

        self._latest: PipelineResult | None = None
        self._latest_lock = threading.Lock()
        self._recompute_stop = threading.Event()
        self._recompute_thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._lifecycle_lock = threading.Lock()
        self._recompute_failures = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._start_time is None:
                self._start_time = time.perf_counter()
            self._engine.start()
            self._recompute_stop.clear()
            if self._recompute_thread is None or not self._recompute_thread.is_alive():
                self._recompute_thread = threading.Thread(
                    target=self._recompute_loop, daemon=True, name="DashRecompute"
                )
                self._recompute_thread.start()

    def pause(self) -> None:
        self._engine.pause()

    def resume(self) -> None:
        self._engine.resume()

    def stop(self) -> None:
        with self._lifecycle_lock:
            self._recompute_stop.set()
            self._engine.stop()
            if self._recompute_thread is not None:
                self._recompute_thread.join(timeout=2.0)

    # ── recompute loop ────────────────────────────────────────────────────────

    def _recompute_loop(self) -> None:
        period = self._recompute_period_s
        next_fire = time.perf_counter()
        log = logging.getLogger(__name__)
        while not self._recompute_stop.is_set():
            next_fire += period
            gap = next_fire - time.perf_counter()
            if gap > 0:
                # Wait on the stop event so stop() returns within ~one tick.
                if self._recompute_stop.wait(gap):
                    return
            else:
                # Recompute took longer than the budget; resync without piling up.
                next_fire = time.perf_counter()
            try:
                self.recompute()
            except Exception:
                # Tolerate transients (e.g. very short window after restart) but
                # surface the first occurrence so silent freezes are noticed.
                self._recompute_failures += 1
                if self._recompute_failures == 1:
                    log.exception("recompute failed (first occurrence)")

    def recompute(self) -> PipelineResult | None:
        snap = self._ring.snapshot(self.window_frames)
        # Wait for at least half a window before producing metrics.
        if snap.shape[0] < self.window_frames // 2:
            return None
        result = run_pipeline(
            snap,
            range_res_m=self.range_res_m,
            fs_slow_hz=self.fs_slow_hz,
            save_plots=False,
        )
        with self._latest_lock:
            self._latest = result
        return result

    # ── read-side ─────────────────────────────────────────────────────────────

    @property
    def latest(self) -> PipelineResult | None:
        with self._latest_lock:
            return self._latest

    def snapshot(self) -> _Snapshot:
        elapsed = (time.perf_counter() - self._start_time) if self._start_time else 0.0
        with self._latest_lock:
            result = self._latest
        return _Snapshot(
            result=result,
            n_appended=self._ring.n_appended,
            is_paused=self._engine.is_paused,
            is_exhausted=self._engine.is_exhausted,
            elapsed_s=float(elapsed),
        )

