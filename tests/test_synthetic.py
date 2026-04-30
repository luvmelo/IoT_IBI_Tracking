import numpy as np

from radar_analysis.synthetic import radar_phase_signal, synthetic_range_cube


def test_radar_phase_signal_recovers_known_frequencies():
    fs = 25.0
    t, phi = radar_phase_signal(
        duration_s=120.0,
        fs=fs,
        hr_hz=1.2,
        hr_amp_mm=0.5,
        resp_hz=0.25,
        resp_amp_mm=2.0,
        drift_mm=0.0,
        noise_std_mm=0.0,
        seed=0,
    )
    assert phi.shape == t.shape
    # FFT should have peaks near 0.25 Hz and 1.2 Hz
    spec = np.abs(np.fft.rfft(phi - phi.mean()))
    freqs = np.fft.rfftfreq(phi.size, d=1 / fs)
    # Find local peaks near expected frequencies
    resp_idx = int(np.argmin(np.abs(freqs - 0.25)))
    hr_idx = int(np.argmin(np.abs(freqs - 1.2)))
    # Both bins should be far above the median bin magnitude
    median_mag = float(np.median(spec))
    assert spec[resp_idx] > 50 * median_mag
    assert spec[hr_idx] > 50 * median_mag


def test_radar_phase_signal_includes_drift():
    fs = 25.0
    _, phi_drift = radar_phase_signal(
        duration_s=10.0, fs=fs, hr_amp_mm=0, resp_amp_mm=0, drift_mm=5.0,
        noise_std_mm=0, seed=0,
    )
    # Linear ramp → ends should be ordered
    assert phi_drift[-1] > phi_drift[0]


def test_radar_phase_signal_spike_injection():
    _, phi = radar_phase_signal(
        duration_s=5.0, fs=25.0, hr_amp_mm=0, resp_amp_mm=0, drift_mm=0,
        noise_std_mm=0, n_spikes=3, spike_amp_rad=10.0, seed=1,
    )
    # 3 samples should be ≥ 8 rad (spike_amp_rad - some), rest ≈ 0
    big = np.sum(np.abs(phi) > 5.0)
    assert big == 3


def test_synthetic_range_cube_shape_and_target_location():
    range_res = 0.04
    cube = synthetic_range_cube(
        n_frames=20, n_chirps=16, n_range_bins=64, n_rx=4,
        range_res_m=range_res, target_range_m=0.6, target_amp=2000.0,
        noise_std=5.0, seed=0,
    )
    assert cube.shape == (20, 16, 64, 4)
    assert cube.dtype == np.complex64
    # Target bin should have the highest mean power
    power = (np.abs(cube) ** 2).mean(axis=(0, 1, 3))
    assert int(np.argmax(power)) == int(round(0.6 / range_res))


def test_synthetic_range_cube_clutter_optional():
    range_res = 0.04
    cube = synthetic_range_cube(
        range_res_m=range_res,
        target_range_m=0.5, target_amp=500.0,
        clutter_range_m=1.0, clutter_amp=2000.0,
        noise_std=5.0, seed=0,
    )
    power = (np.abs(cube) ** 2).mean(axis=(0, 1, 3))
    # Clutter (higher amplitude) should dominate static power
    assert int(np.argmax(power)) == int(round(1.0 / range_res))
