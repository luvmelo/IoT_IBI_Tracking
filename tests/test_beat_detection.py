import numpy as np
import pytest

from radar_analysis.beat_detection import (
    detect_beats,
    parabolic_refine,
    peaks_to_ibi_ms,
    clean_ibi,
)


def _heartbeat_pulses(fs, duration_s, bpm, jitter_ms=0.0, seed=0):
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    n_beats = int(duration_s / period)
    beat_times = np.arange(n_beats) * period + 0.05  # offset away from sample 0
    if jitter_ms:
        beat_times += 1e-3 * jitter_ms * rng.standard_normal(n_beats)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    signal = np.zeros(n)
    # Each beat: a half-sine pulse 100 ms wide.
    pulse_w_s = 0.10
    for bt in beat_times:
        idx_lo = int(round((bt - pulse_w_s / 2) * fs))
        idx_hi = idx_lo + int(round(pulse_w_s * fs))
        if 0 <= idx_lo and idx_hi < n:
            tt = np.linspace(0, np.pi, idx_hi - idx_lo)
            signal[idx_lo:idx_hi] += np.sin(tt)
    return t, signal, beat_times


def test_detect_beats_finds_known_peaks():
    fs = 25.0
    _, h, beats_true = _heartbeat_pulses(fs, 30.0, bpm=72)
    peaks_s = detect_beats(h, fs=fs)
    # Tolerance: ±2 samples (80 ms) — wider than refinement error
    matched = []
    for bt in beats_true:
        if peaks_s.size:
            err = np.min(np.abs(peaks_s - bt))
            matched.append(err < 2 / fs)
    assert sum(matched) >= int(0.95 * len(beats_true))


def test_parabolic_refine_improves_timing():
    fs = 25.0
    period = 1.0
    n = int(20 * fs)
    t = np.arange(n) / fs
    # Pulses centered between samples to give the integer detector a 1/2-sample bias
    signal = np.zeros(n)
    true_times = np.arange(1.5, 20, period) + 0.5 / fs  # offset 1/2 sample
    for bt in true_times:
        idx_lo = int(round((bt - 0.05) * fs))
        idx_hi = idx_lo + int(round(0.10 * fs))
        if 0 <= idx_lo and idx_hi < n:
            tt = np.linspace(0, np.pi, idx_hi - idx_lo)
            signal[idx_lo:idx_hi] += np.sin(tt)

    peaks_int = detect_beats(signal, fs=fs, refine=False)
    peaks_ref = detect_beats(signal, fs=fs, refine=True)
    err_int = np.median([np.min(np.abs(peaks_int - bt)) for bt in true_times])
    err_ref = np.median([np.min(np.abs(peaks_ref - bt)) for bt in true_times])
    assert err_ref <= err_int  # refinement should not make things worse
    # Refined error should typically be < integer step
    assert err_ref < 1.0 / fs


def test_peaks_to_ibi_ms_basic():
    peaks = np.array([0.0, 1.0, 2.0, 3.2])
    ibi = peaks_to_ibi_ms(peaks)
    np.testing.assert_allclose(ibi, [1000.0, 1000.0, 1200.0])


def test_peaks_to_ibi_ms_handles_empty():
    assert peaks_to_ibi_ms(np.array([])).size == 0
    assert peaks_to_ibi_ms(np.array([1.0])).size == 0


def test_clean_ibi_rejects_out_of_range():
    ibi = np.array([1000.0, 1000.0, 200.0, 1000.0, 1800.0, 1000.0])
    mask = clean_ibi(ibi, low_ms=300, high_ms=1500, rel_tol=1.0)
    assert not mask[2] and not mask[4]
    assert mask[0] and mask[1] and mask[3] and mask[5]


def test_clean_ibi_rejects_relative_outlier():
    ibi = np.array([1000.0, 1000.0, 1000.0, 1300.0, 1000.0, 1000.0])
    mask = clean_ibi(ibi, rel_tol=0.20)
    assert not mask[3]


def test_clean_ibi_handles_short_series():
    ibi = np.array([1000.0, 1010.0])
    mask = clean_ibi(ibi)
    assert mask.all()


def test_clean_ibi_empty():
    assert clean_ibi(np.array([])).size == 0


def test_detect_beats_validates_input():
    with pytest.raises(ValueError):
        detect_beats(np.zeros((4, 4)), fs=25.0)
    with pytest.raises(ValueError):
        detect_beats(np.zeros(100), fs=0)


def test_detect_beats_returns_empty_on_flat_signal():
    # std == 0 → no real peaks; must NOT spuriously detect numerical-noise maxima.
    out = detect_beats(np.zeros(500), fs=25.0)
    assert out.size == 0


def test_clean_ibi_all_out_of_range_yields_empty_mask():
    # Realistic failure mode: a vibration burst produces a stretch of <250 ms IBI.
    # The function must reject all of them rather than silently passing them through.
    ibi = np.array([200.0, 250.0, 180.0, 210.0])
    mask = clean_ibi(ibi, low_ms=300.0, high_ms=1500.0)
    assert mask.dtype == bool
    assert mask.shape == ibi.shape
    assert not mask.any()
