import time

import numpy as np
import pytest

from dashboard.state import DashboardState
from radar_analysis.streams import SyntheticReplaySource


def _make_state(duration_s=15.0, fs=25.0, window_s=8.0):
    src = SyntheticReplaySource(
        duration_s=duration_s, fs_slow_hz=fs,
        n_chirps=8, n_range_bins=64, n_rx=4,
        target_motion_hz=1.2, seed=0, realtime=False,
    )
    return DashboardState(src, window_s=window_s, recompute_period_s=0.05)


def test_state_recompute_returns_none_before_half_window_filled():
    state = _make_state(window_s=4.0)
    # Manually drop in a couple of zero frames; pipeline shouldn't be called yet.
    src = SyntheticReplaySource(
        duration_s=0.4, fs_slow_hz=25.0,
        n_chirps=8, n_range_bins=64, n_rx=4,
        seed=0, realtime=False,
    )
    for i, frame in enumerate(iter(src)):
        if i >= 3:
            break
        state._ring.append(frame)
    assert state.recompute() is None


def test_state_recompute_returns_result_after_window_fills():
    state = _make_state()
    # Fill the ring synchronously by draining the source ourselves.
    for frame in iter(state._engine._source):
        state._ring.append(frame)
        if state._ring.n_appended >= state.window_frames:
            break
    result = state.recompute()
    assert result is not None
    # 1.2 Hz simulated → ~72 BPM, but with only the window of data the estimate may
    # spread; just assert it's in a plausible cardiac range.
    assert 40.0 <= result.metrics["mean_hr_bpm"] <= 120.0


def test_state_start_then_stop_drives_pipeline_and_emits_latest():
    src = SyntheticReplaySource(
        duration_s=12.0, fs_slow_hz=50.0,
        n_chirps=4, n_range_bins=64, n_rx=4,
        target_motion_hz=1.2, seed=0, realtime=True,
    )
    state = DashboardState(src, window_s=6.0, recompute_period_s=0.10)
    state.start()
    try:
        # Wait up to 9 seconds for at least one valid PipelineResult to land.
        deadline = time.perf_counter() + 9.0
        while state.latest is None and time.perf_counter() < deadline:
            time.sleep(0.1)
        assert state.latest is not None, "no pipeline result produced within 9 s"
        # HR should be in the right ballpark for a 1.2 Hz source.
        assert 50.0 < state.latest.metrics["mean_hr_bpm"] < 100.0
    finally:
        state.stop()


def test_state_snapshot_exposes_lifecycle_flags():
    state = _make_state()
    snap = state.snapshot()
    assert snap.result is None
    assert snap.is_paused is False
    assert snap.is_exhausted is False
    assert snap.n_appended == 0
    assert snap.elapsed_s == pytest.approx(0.0, abs=0.01)


def test_recompute_loop_does_not_pile_up_on_slow_pipeline(monkeypatch):
    state = _make_state()
    counter = {"calls": 0}

    def slow_recompute():
        counter["calls"] += 1
        time.sleep(0.20)  # Way over the 50ms recompute_period_s.
        return None

    monkeypatch.setattr(state, "recompute", slow_recompute)
    state.start()
    try:
        time.sleep(0.6)
    finally:
        state.stop()
    # If we piled up, calls would explode (>=12 in 0.6s @ 50ms). Guard at 8.
    assert counter["calls"] <= 8, f"recompute piled up: {counter['calls']} calls"


def test_recompute_loop_survives_one_exception_and_keeps_running(monkeypatch):
    """A transient pipeline failure must not freeze the dashboard forever."""
    state = _make_state()
    state_holder = {"calls": 0}
    real_recompute = state.recompute

    def patched():
        state_holder["calls"] += 1
        if state_holder["calls"] == 1:
            raise RuntimeError("transient failure")
        return real_recompute()

    # Pre-fill the ring so the second call has something to chew on.
    src = SyntheticReplaySource(
        duration_s=2.0, fs_slow_hz=25.0,
        n_chirps=8, n_range_bins=64, n_rx=4,
        seed=0, realtime=False,
    )
    for frame in iter(src):
        state._ring.append(frame)
        if state._ring.n_appended >= state.window_frames:
            break

    monkeypatch.setattr(state, "recompute", patched)
    state.start()
    try:
        deadline = time.perf_counter() + 4.0
        while state.latest is None and time.perf_counter() < deadline:
            time.sleep(0.05)
    finally:
        state.stop()
    assert state_holder["calls"] >= 2, "loop did not retry after exception"
    assert state.latest is not None, "latest never updated after first exception"


def test_state_stop_returns_promptly_even_during_idle_sleep():
    """Regression guard: stop() must not block on a sleep timer."""
    src = SyntheticReplaySource(
        duration_s=30.0, fs_slow_hz=25.0,
        n_chirps=8, n_range_bins=64, n_rx=4,
        seed=0, realtime=True,
    )
    state = DashboardState(src, window_s=8.0, recompute_period_s=1.0)  # long sleep
    state.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    state.stop()
    elapsed = time.perf_counter() - t0
    # With an Event.wait-based loop, stop should return well under the period.
    assert elapsed < 0.5, f"stop() took {elapsed:.3f}s; expected < 0.5s"
