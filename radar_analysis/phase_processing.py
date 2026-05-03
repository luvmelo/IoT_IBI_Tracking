"""Phase extraction, unwrapping, detrending, despiking, and motion gating.

Inputs are 1-D slow-time signals (one phase value per frame, optionally
per chirp). Operations follow the TI vital-signs lab + Hampel-filter
recipe from `mmwave_ibi_hrv_research_note.md`:

    z          → remove_dc (optional)    → z_centered     # static clutter
    z          → atan2 + np.unwrap        → φ_unwrapped
    φ          → median-filter detrend    → φ_detrended
    φ          → Hampel despike           → φ_clean
    φ          → first-diff motion gate   → bool mask (True = clean)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter


def remove_dc(z: np.ndarray) -> np.ndarray:
    """Subtract the mean of a complex slow-time signal before atan2.

    Real captures have a static clutter component (wall, table, the radar's
    own near-field echo) at the chest range bin that adds a fixed phasor to
    every frame. That biases the atan2 operating point away from the origin
    and compresses the effective phase swing from heartbeat motion. The TI
    vital-signs developer guide §2.3 calls this "DC offset correction".

    Plain mean subtraction is biased when the chest motion is not zero-mean
    over the window — short captures where respiration's slow component
    isn't centered. For a more robust DC estimate use `circle_fit_dc` /
    `coherent_combine_rx` which fits the IQ cloud's rotation center.
    """
    if not np.iscomplexobj(z):
        raise TypeError("remove_dc expects a complex array")
    return z - z.mean(axis=-1, keepdims=True)


def _kasa_circle_center(z: np.ndarray) -> complex:
    """Algebraic Kasa fit: minimize Σ (|z_i - c|^2 - r^2)^2 in linear form.

    Falls back to the temporal mean when (a) there are <3 samples, (b) the
    linear system is singular, or (c) the fitted center sits implausibly
    far from the data (the failure mode for a short IQ arc).
    """
    if z.size < 3:
        return complex(z.mean())
    z_mean = complex(z.mean())
    x, y = z.real, z.imag
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    try:
        D, E, _ = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return z_mean
    c = complex(-D / 2.0, -E / 2.0)
    spread = float(np.std(x) + np.std(y))
    if not np.isfinite(c.real + c.imag) or abs(c - z_mean) > 10.0 * spread:
        return z_mean
    return c


def circle_fit_dc(z: np.ndarray) -> np.ndarray:
    """Per-trace circle-fit DC removal along the last axis.

    More robust than `remove_dc` when the chest motion arc is asymmetric
    around its rotation center — the failure mode for short captures where
    respiration leaves a non-zero net drift in the temporal mean.
    """
    if not np.iscomplexobj(z):
        raise TypeError("circle_fit_dc expects a complex array")
    if z.ndim == 1:
        return z - _kasa_circle_center(z)
    flat = z.reshape(-1, z.shape[-1])
    centers = np.array([_kasa_circle_center(row) for row in flat])
    return (flat - centers[:, None]).reshape(z.shape)


def coherent_combine_rx(z_per_rx: np.ndarray) -> np.ndarray:
    """Per-RX circle-fit DC removal + clutter-phase alignment + coherent sum.

    Each RX has an unknown phase offset from RF path length differences
    (cable / antenna / receive chain). Naive complex averaging across RX
    causes partial cancellation — up to 6 dB SNR loss for 4 RX worst case.
    For a near-broadside chest target the clutter and target phasors
    share that RF path offset, so:

    1. estimate each RX's static-clutter phasor by Kasa circle-fit on its
       IQ cloud (TI Vital Signs Developer Guide §2.3 — unbiased to
       small-arc motion, unlike a plain temporal mean);
    2. subtract the clutter (DC removal);
    3. rotate the residual by `conj(c)/|c|` so each RX's clutter direction
       maps to the positive real axis — common phase frame across RX;
    4. coherent average across RX.

    Input:  ``z_per_rx`` (F, R) complex, the chirp-averaged signal at one
            range bin per RX.
    Output: (F,) complex.
    """
    if z_per_rx.ndim != 2:
        raise ValueError(f"expected (F, R), got {z_per_rx.shape}")
    n_frames, n_rx = z_per_rx.shape
    out = np.zeros(n_frames, dtype=np.complex128)
    for r in range(n_rx):
        z_r = z_per_rx[:, r]
        c = _kasa_circle_center(z_r)
        if abs(c) < 1e-12:
            out += z_r - c
        else:
            out += (z_r - c) * (np.conj(c) / abs(c))
    return out / n_rx


def extract_phase(z: np.ndarray) -> np.ndarray:
    """Atan2 + unwrap along the last axis. Input complex, output float64 radians.

    For best heartbeat SNR, call `remove_dc(z)` before this on real captures.
    """
    if not np.iscomplexobj(z):
        raise TypeError("extract_phase expects a complex array (range-FFT bin slice).")
    return np.unwrap(np.angle(z), axis=-1)


def detrend_median(phi: np.ndarray, fs: float, window_s: float = 2.0) -> np.ndarray:
    """Subtract a sliding-median baseline. `window_s` of ~2 s passes respiration."""
    win = max(3, int(round(window_s * fs)))
    if win % 2 == 0:
        win += 1  # median_filter prefers odd
    baseline = median_filter(phi, size=win, mode="nearest")
    return phi - baseline


def despike_hampel(
    phi: np.ndarray,
    k_w: int = 12,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Hampel filter: replace |x - median| > n_sigma·1.4826·MAD with the local median.

    `k_w` is the half-window size in samples (so the full window is 2·k_w+1).
    Default `k_w=12` is ~1 s at fs_slow_hz=25 Hz, longer than one cardiac
    period (500–1000 ms). A shorter window (e.g. k_w=3 = 280 ms) misclassifies
    real systolic peaks as spikes when the surrounding window happens to sit
    on the diastolic baseline.

    Returns a copy; input is not modified.
    """
    if k_w < 1:
        raise ValueError("k_w must be >= 1")
    out = phi.astype(np.float64, copy=True)
    win = 2 * k_w + 1
    local_median = median_filter(out, size=win, mode="nearest")
    abs_dev = np.abs(out - local_median)
    mad = median_filter(abs_dev, size=win, mode="nearest")
    threshold = n_sigma * 1.4826 * mad
    spikes = abs_dev > np.maximum(threshold, 1e-12)
    out[spikes] = local_median[spikes]
    return out


def motion_mask(
    phi: np.ndarray,
    fs: float,
    window_s: float = 1.0,
    energy_factor: float = 5.0,
) -> np.ndarray:
    """Return bool mask, True where phase is "clean" (low motion energy).

    Energy proxy: sliding-window mean of squared first differences. Threshold
    is `energy_factor × median(energy)` — robust to long quiet stretches but
    flags abrupt motion bursts, which is the failure mode TI's lab calls out.
    """
    if phi.size == 0:
        return np.zeros(0, dtype=bool)
    win = max(3, int(round(window_s * fs)))
    diff = np.diff(phi, prepend=phi[0])
    energy = np.convolve(diff ** 2, np.ones(win) / win, mode="same")
    median_energy = float(np.median(energy))
    mean_energy = float(np.mean(energy))
    # Truly flat input → no motion anywhere → keep everything.
    if median_energy <= 0 and mean_energy <= 0:
        return np.ones_like(energy, dtype=bool)
    if median_energy <= 0:
        median_energy = mean_energy
    threshold = energy_factor * median_energy
    return energy < threshold
