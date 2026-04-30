import numpy as np
import pytest

from radar_analysis.phase_processing import (
    remove_dc,
    extract_phase,
    detrend_median,
    despike_hampel,
    motion_mask,
)


def test_extract_phase_recovers_linear_phase():
    n = 256
    t = np.arange(n)
    true_phase = 0.05 * t  # ramp, no wrap issues
    z = np.exp(1j * true_phase).astype(np.complex64)
    out = extract_phase(z)
    np.testing.assert_allclose(out, true_phase, atol=1e-5)


def test_extract_phase_unwraps_past_pi():
    # Slope > π/sample → atan2 alone wraps; np.unwrap should fix it
    n = 64
    true_phase = 0.5 * np.arange(n)  # 0.5 rad/sample, clearly wraps
    z = np.exp(1j * true_phase)
    out = extract_phase(z)
    # unwrapped phase should be monotonically increasing
    assert np.all(np.diff(out) > 0)
    np.testing.assert_allclose(out - out[0], true_phase - true_phase[0], atol=1e-6)


def test_extract_phase_rejects_real_input():
    with pytest.raises(TypeError):
        extract_phase(np.zeros(16, dtype=np.float64))


def test_detrend_median_removes_linear_drift_keeps_oscillation():
    fs = 25.0
    n = int(30 * fs)
    t = np.arange(n) / fs
    drift = 5.0 * (t / t[-1])
    osc = 0.5 * np.sin(2 * np.pi * 1.2 * t)
    phi = drift + osc
    detrended = detrend_median(phi, fs=fs, window_s=2.0)
    # Drift removed: detrended should have near-zero linear slope
    slope = np.polyfit(t, detrended, 1)[0]
    assert abs(slope) < 0.05
    # Oscillation broadly preserved by the median detrend; bandpass tightens it
    # downstream. A 2-s median window does soften a 1.2 Hz sine to ~70-80 % std.
    assert detrended.std() > 0.7 * osc.std()


def test_despike_hampel_replaces_spike_keeps_signal():
    rng = np.random.default_rng(0)
    n = 500
    base = 0.1 * np.sin(2 * np.pi * np.arange(n) / 50) + 0.005 * rng.standard_normal(n)
    spiked = base.copy()
    spike_idx = [50, 200, 400]
    for i in spike_idx:
        spiked[i] += 10.0  # huge spike

    cleaned = despike_hampel(spiked, k_w=3, n_sigma=3.0)
    # Spikes flattened
    for i in spike_idx:
        assert abs(cleaned[i] - base[i]) < 0.5
    # Bulk signal preserved
    untouched = np.setdiff1d(np.arange(n), spike_idx)
    np.testing.assert_allclose(cleaned[untouched], base[untouched], atol=0.05)


def test_motion_mask_flags_burst_only():
    rng = np.random.default_rng(1)
    fs = 25.0
    n = int(20 * fs)
    phi = 0.1 * np.sin(2 * np.pi * 1.2 * np.arange(n) / fs) + 0.001 * rng.standard_normal(n)
    # Inject a high-energy burst in the middle 1 s
    burst_start, burst_end = int(10 * fs), int(11 * fs)
    phi[burst_start:burst_end] += 5.0 * rng.standard_normal(burst_end - burst_start)

    mask = motion_mask(phi, fs=fs, window_s=1.0, energy_factor=5.0)
    # Outside the burst → mostly clean
    outside = np.r_[mask[: burst_start - int(fs)], mask[burst_end + int(fs) :]]
    assert outside.mean() > 0.9
    # Inside the burst → flagged
    inside = mask[burst_start:burst_end]
    assert inside.mean() < 0.5


def test_motion_mask_all_clean_when_signal_is_flat():
    fs = 25.0
    phi = 0.001 * np.sin(2 * np.pi * 1.0 * np.arange(int(5 * fs)) / fs)
    mask = motion_mask(phi, fs=fs)
    # Everything below 5× median energy → all True
    assert mask.all()


def test_motion_mask_constant_input_returns_all_clean():
    # A literally flat phase has zero energy everywhere; threshold must not collapse.
    phi = np.full(200, 0.5)
    mask = motion_mask(phi, fs=25.0)
    assert mask.all()


def test_remove_dc_subtracts_mean():
    z = np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], dtype=np.complex64)
    centered = remove_dc(z)
    assert np.isclose(centered.mean(), 0)
    # Mean of [1+1j..4+4j] = 2.5+2.5j; centered should be the original minus that
    np.testing.assert_allclose(centered, z - (2.5 + 2.5j))


def test_remove_dc_recovers_heartbeat_swing_after_clutter_added():
    # Real-world failure mode: a strong static phasor compresses the atan2
    # operating point. Without remove_dc, the unwrapped phase swing collapses.
    fs = 25.0
    n = int(20 * fs)
    t = np.arange(n) / fs
    chest_swing_rad = 1.0 * np.sin(2 * np.pi * 1.2 * t)  # ~1 rad chest oscillation
    chest_phasor = np.exp(1j * chest_swing_rad).astype(np.complex64)
    static_clutter = np.complex64(50.0 + 0j)             # 50× the chest amplitude
    z_with_clutter = chest_phasor + static_clutter

    swing_no_remove = np.ptp(extract_phase(z_with_clutter))
    swing_remove = np.ptp(extract_phase(remove_dc(z_with_clutter)))
    # DC removal should preserve the full ~2 rad peak-to-peak swing;
    # without it the swing is squashed by a large factor (depends on clutter mag).
    assert swing_remove > 1.5
    assert swing_no_remove < 0.5 * swing_remove


def test_remove_dc_rejects_real_input():
    with pytest.raises(TypeError):
        remove_dc(np.zeros(16, dtype=np.float64))
