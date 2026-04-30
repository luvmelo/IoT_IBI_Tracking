"""Synthetic generators for unit tests of the IBI/HRV pipeline.

We model the slow-time chest signal directly in phase space (radians) and
also provide a post-range-FFT cube for chest-bin-selection tests, so the
downstream modules can be exercised end-to-end without real captures.

Carrier defaults to 77 GHz (IWR1443 BOOST). Phase-to-displacement mapping
is the standard `Δd = λ/(4π) · Δφ`, so a 1 mm chest motion at 77 GHz
produces |Δφ| ≈ 3.22 rad — well above the noise floor and easy to detect.
"""

from __future__ import annotations

import numpy as np

C_LIGHT_M_S = 3e8


def radar_phase_signal(
    duration_s: float,
    fs: float,
    *,
    carrier_hz: float = 77e9,
    hr_hz: float = 1.2,
    hr_amp_mm: float = 0.5,
    resp_hz: float = 0.25,
    resp_amp_mm: float = 2.0,
    drift_mm: float = 1.0,
    noise_std_mm: float = 0.02,
    n_spikes: int = 0,
    spike_amp_rad: float = 5.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic chest-displacement phase signal.

    Returns `(t_s, phi_rad)` where `phi_rad` is the unwrapped phase a real
    radar would report after `extract_phase`. Useful for downstream tests
    that should *not* depend on the FFT/unwrap path itself.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs

    wavelength = C_LIGHT_M_S / carrier_hz
    phase_per_m = 4 * np.pi / wavelength

    disp_m = (
        1e-3 * resp_amp_mm * np.sin(2 * np.pi * resp_hz * t)
        + 1e-3 * hr_amp_mm * np.sin(2 * np.pi * hr_hz * t)
        + 1e-3 * drift_mm * (t / max(duration_s, 1e-9))
    )
    phi = phase_per_m * disp_m
    phi += rng.normal(0.0, phase_per_m * 1e-3 * noise_std_mm, size=n)

    if n_spikes > 0 and n > 2:
        idx = rng.choice(np.arange(1, n - 1), size=min(n_spikes, n - 2), replace=False)
        signs = rng.choice([-1.0, 1.0], size=idx.size)
        phi[idx] += signs * spike_amp_rad

    return t, phi


def synthetic_range_cube(
    *,
    n_frames: int = 60,
    n_chirps: int = 32,
    n_range_bins: int = 64,
    n_rx: int = 4,
    range_res_m: float = 0.04,
    target_range_m: float = 0.7,
    target_amp: float = 1000.0,
    target_motion_amp_mm: float = 0.5,
    target_motion_hz: float = 1.2,
    clutter_range_m: float | None = None,
    clutter_amp: float = 0.0,
    noise_std: float = 10.0,
    fs_frame: float = 25.0,
    carrier_hz: float = 77e9,
    seed: int | None = None,
) -> np.ndarray:
    """Synthetic post-range-FFT cube: noise + a moving point target + optional static clutter.

    Output shape `(n_frames, n_chirps, n_range_bins, n_rx)` complex64 — the
    same shape `np.fft.fft(load_capture(...), axis=2)[..., :n_range_bins, :]`
    would produce on real data.
    """
    rng = np.random.default_rng(seed)
    shape = (n_frames, n_chirps, n_range_bins, n_rx)
    cube = (
        rng.standard_normal(shape, dtype=np.float32)
        + 1j * rng.standard_normal(shape, dtype=np.float32)
    ) * np.float32(noise_std)

    wavelength = C_LIGHT_M_S / carrier_hz
    phase_per_m = 4 * np.pi / wavelength
    target_bin = int(round(target_range_m / range_res_m))
    if not 0 <= target_bin < n_range_bins:
        raise ValueError(f"target_range_m {target_range_m} → bin {target_bin} out of [0, {n_range_bins})")

    t_frame = np.arange(n_frames) / fs_frame
    delta_m = 1e-3 * target_motion_amp_mm * np.sin(2 * np.pi * target_motion_hz * t_frame)
    target_phase = phase_per_m * delta_m  # (n_frames,)
    cube[:, :, target_bin, :] += np.float32(target_amp) * np.exp(
        1j * target_phase[:, None, None]
    ).astype(np.complex64)

    if clutter_range_m is not None and clutter_amp > 0:
        clutter_bin = int(round(clutter_range_m / range_res_m))
        if 0 <= clutter_bin < n_range_bins:
            cube[:, :, clutter_bin, :] += np.complex64(clutter_amp)

    return cube
