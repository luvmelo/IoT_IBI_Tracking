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
    """
    if not np.iscomplexobj(z):
        raise TypeError("remove_dc expects a complex array")
    return z - z.mean(axis=-1, keepdims=True)


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
    k_w: int = 3,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """Hampel filter: replace |x - median| > n_sigma·1.4826·MAD with the local median.

    `k_w` is the half-window size in samples (so the full window is 2·k_w+1).
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
