import math

import numpy as np
import pytest

from radar_analysis.pipeline import PipelineResult
from dashboard.figures import heartbeat_figure, ibi_figure, phase_figure


def _stub_result(*, n=200, fs=25.0, peak_times_s=None):
    rng = np.random.default_rng(0)
    phi = 0.1 * np.sin(2 * np.pi * 1.2 * np.arange(n) / fs)
    hb = phi.copy()
    return PipelineResult(
        chest_bin=10,
        chest_range_m=0.4,
        chest_bin_score=1.0,
        fs_slow_hz=fs,
        duration_s=n / fs,
        n_frames=n,
        phi_clean=phi,
        motion_mask=np.ones(n, dtype=bool),
        heartbeat=hb,
        peak_times_s=np.asarray([] if peak_times_s is None else peak_times_s, dtype=np.float64),
        ibi_ms=np.array([], dtype=np.float64),
        nn_ms=np.array([], dtype=np.float64),
        metrics={
            "mean_ibi_ms": float("nan"),
            "mean_hr_bpm": float("nan"),
            "sdnn_ms": float("nan"),
            "rmssd_ms": float("nan"),
            "pnn50_pct": float("nan"),
        },
    )


def test_phase_figure_with_none_returns_empty():
    fig = phase_figure(None, plot_window_s=10.0)
    assert len(fig.data) == 0


def test_heartbeat_figure_excludes_peaks_outside_window():
    # 8 s of signal; window 4 s → only peaks at t >= 4.0 s should appear.
    fs = 25.0
    n = int(8 * fs)
    result = _stub_result(n=n, fs=fs, peak_times_s=[1.0, 5.0, 7.5])
    fig = heartbeat_figure(result, plot_window_s=4.0)
    # Trace 0 = line, trace 1 = markers (peaks)
    assert len(fig.data) == 2
    marker_xs = list(fig.data[1].x)
    assert 1.0 not in marker_xs
    assert 5.0 in marker_xs and 7.5 in marker_xs


def test_phase_figure_yaxis_is_pinned_not_autoranged():
    # Two slightly different signals must produce *the same* y-range bucket
    # so the axis doesn't twitch tick-to-tick.
    rng = np.random.default_rng(0)
    n, fs = 200, 25.0
    phi_a = 0.5 * np.sin(2 * np.pi * 1.2 * np.arange(n) / fs)
    phi_b = phi_a + 0.05 * rng.standard_normal(n)
    a = _stub_result(n=n, fs=fs); a.phi_clean = phi_a; a.heartbeat = phi_a
    b = _stub_result(n=n, fs=fs); b.phi_clean = phi_b; b.heartbeat = phi_b
    fig_a = phase_figure(a, plot_window_s=8.0)
    fig_b = phase_figure(b, plot_window_s=8.0)
    range_a = fig_a.layout.yaxis.range
    range_b = fig_b.layout.yaxis.range
    assert range_a is not None and range_b is not None
    assert tuple(range_a) == tuple(range_b), (range_a, range_b)


def test_ibi_figure_with_no_data_is_empty():
    fig = ibi_figure(_stub_result(n=10))
    assert len(fig.data) == 0
