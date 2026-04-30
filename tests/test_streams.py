import time

import numpy as np
import pytest

from radar_analysis.streams import FrameSource, SyntheticReplaySource


def test_synthetic_source_emits_correct_shape_and_dtype():
    src = SyntheticReplaySource(
        duration_s=2.0, fs_slow_hz=25.0, n_chirps=8, n_range_bins=32, n_rx=4,
        seed=0, realtime=False,
    )
    frames = list(iter(src))
    assert len(frames) == int(2.0 * 25.0)
    f0 = frames[0]
    assert f0.shape == (8, 32, 4)
    assert f0.dtype == np.complex64


def test_synthetic_source_satisfies_protocol():
    src = SyntheticReplaySource(duration_s=0.5, fs_slow_hz=10.0, realtime=False)
    assert isinstance(src, FrameSource)


def test_synthetic_source_realtime_pacing_within_tolerance():
    fs = 50.0
    src = SyntheticReplaySource(duration_s=0.5, fs_slow_hz=fs, realtime=True)
    t0 = time.perf_counter()
    frames = list(iter(src))
    elapsed = time.perf_counter() - t0
    expected = (len(frames) - 1) / fs
    # Allow 30% slack for OS scheduling on a busy CI host.
    assert expected * 0.85 <= elapsed <= expected * 1.5 + 0.05, (
        f"elapsed={elapsed:.3f}s vs expected≈{expected:.3f}s"
    )


def test_synthetic_source_close_stops_iteration_early():
    src = SyntheticReplaySource(duration_s=10.0, fs_slow_hz=100.0, realtime=False)
    it = iter(src)
    first = next(it)
    assert first.shape == (src.n_chirps, src.n_range_bins, src.n_rx)
    src.close()
    # After close, iterator must terminate; consume the rest with a guard.
    remaining = []
    for f in it:
        remaining.append(f)
        if len(remaining) > 5:
            pytest.fail("close() did not stop iteration")
    # Any remaining frames are fine; the contract is "must terminate", not "exactly 0 more".
    assert len(remaining) <= 5
