import threading
import time

import numpy as np
import pytest

from radar_analysis.replay import FrameRing, ReplayEngine
from radar_analysis.streams import SyntheticReplaySource


def _make_frame(value: int, n_chirps=4, n_bins=8, n_rx=2):
    return np.full((n_chirps, n_bins, n_rx), value, dtype=np.complex64)


def test_ring_snapshot_returns_chronological_last_n():
    ring = FrameRing(capacity=10, n_chirps=4, n_range_bins=8, n_rx=2)
    for v in range(15):
        ring.append(_make_frame(v))
    snap = ring.snapshot(5)
    assert snap.shape == (5, 4, 8, 2)
    # Last 5 appended values are 10, 11, 12, 13, 14
    for i, v in enumerate([10, 11, 12, 13, 14]):
        assert snap[i, 0, 0, 0].real == pytest.approx(v)


def test_ring_snapshot_handles_underfill():
    ring = FrameRing(capacity=10, n_chirps=4, n_range_bins=8, n_rx=2)
    ring.append(_make_frame(0))
    ring.append(_make_frame(1))
    snap = ring.snapshot(5)
    assert snap.shape == (2, 4, 8, 2)


def test_ring_snapshot_returns_empty_when_no_appends():
    ring = FrameRing(capacity=10, n_chirps=4, n_range_bins=8, n_rx=2)
    snap = ring.snapshot(5)
    assert snap.shape == (0, 4, 8, 2)


def test_ring_snapshot_is_a_copy_not_a_view():
    ring = FrameRing(capacity=4, n_chirps=2, n_range_bins=2, n_rx=2)
    ring.append(_make_frame(7, 2, 2, 2))
    snap = ring.snapshot(1)
    snap[:] = 999
    snap2 = ring.snapshot(1)
    assert snap2[0, 0, 0, 0].real == pytest.approx(7)


def test_ring_rejects_wrong_shape():
    ring = FrameRing(capacity=4, n_chirps=2, n_range_bins=2, n_rx=2)
    with pytest.raises(ValueError):
        ring.append(np.zeros((3, 3, 3), dtype=np.complex64))


def test_ring_concurrent_append_and_snapshot():
    ring = FrameRing(capacity=200, n_chirps=2, n_range_bins=2, n_rx=2)

    def writer():
        for v in range(500):
            ring.append(_make_frame(v, 2, 2, 2))

    def reader():
        for _ in range(50):
            snap = ring.snapshot(20)
            # Either empty or chronologically increasing
            if snap.shape[0] >= 2:
                seq = snap[:, 0, 0, 0].real
                assert np.all(np.diff(seq) >= 0), seq

    w = threading.Thread(target=writer)
    r = threading.Thread(target=reader)
    w.start(); r.start()
    w.join(); r.join()


def test_replay_engine_fills_ring_from_synthetic_source():
    src = SyntheticReplaySource(
        duration_s=1.0, fs_slow_hz=50.0, n_chirps=4, n_range_bins=64, n_rx=4,
        realtime=False,
    )
    ring = FrameRing(capacity=200, n_chirps=4, n_range_bins=64, n_rx=4)
    eng = ReplayEngine(src, ring)
    eng.start()
    deadline = time.perf_counter() + 2.0
    while not eng.is_exhausted and time.perf_counter() < deadline:
        time.sleep(0.02)
    eng.stop()
    assert ring.n_appended == 50  # 1.0s × 50 Hz


def test_replay_engine_stop_while_paused_terminates_promptly():
    src = SyntheticReplaySource(
        duration_s=30.0, fs_slow_hz=100.0,
        n_chirps=4, n_range_bins=64, n_rx=4,
        seed=0, realtime=True,
    )
    ring = FrameRing(capacity=500, n_chirps=src.n_chirps,
                     n_range_bins=src.n_range_bins, n_rx=src.n_rx)
    eng = ReplayEngine(src, ring)
    eng.start()
    time.sleep(0.05)
    eng.pause()
    time.sleep(0.05)
    t0 = time.perf_counter()
    eng.stop()
    elapsed = time.perf_counter() - t0
    assert not eng.is_running, "thread still alive after stop() while paused"
    # Must come back well within the 2s join timeout.
    assert elapsed < 0.5, f"stop() took {elapsed:.3f}s while paused"


def test_replay_engine_pause_and_resume():
    src = SyntheticReplaySource(duration_s=10.0, fs_slow_hz=100.0, realtime=True)
    ring = FrameRing(capacity=2000, n_chirps=src.n_chirps,
                     n_range_bins=src.n_range_bins, n_rx=src.n_rx)
    eng = ReplayEngine(src, ring)
    eng.start()
    time.sleep(0.15)
    eng.pause()
    n_paused = ring.n_appended
    time.sleep(0.20)
    n_after_pause = ring.n_appended
    # While paused, the ring shouldn't grow more than a small handful (one in-flight frame).
    assert n_after_pause - n_paused <= 3
    eng.resume()
    time.sleep(0.20)
    assert ring.n_appended > n_after_pause
    eng.stop()
