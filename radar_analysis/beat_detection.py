"""Beat detection and IBI quality control on a heartbeat-band slow-time signal.

Pipeline:
    h(t) → find_peaks   → integer peak indices
         → parabolic    → sub-sample peak times (better than ±1/(2 fs))
         → diffs        → IBI series (ms)
         → clean_ibi    → bool mask (True = keep this interval)

The `find_peaks` `distance` constraint enforces the cardiac refractory
period (default 200 BPM = 300 ms), preventing harmonic peaks from being
mis-detected as beats.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def detect_beats(
    h: np.ndarray,
    fs: float,
    *,
    max_bpm: float = 200.0,
    prominence_factor: float = 0.5,
    refine: bool = True,
) -> np.ndarray:
    """Return peak times in seconds (sub-sample if `refine=True`)."""
    if h.ndim != 1:
        raise ValueError(f"h must be 1-D, got shape {h.shape}")
    if fs <= 0:
        raise ValueError("fs must be positive")
    distance = max(1, int(round(60.0 / max_bpm * fs)))
    # MAD-based scale (robust to outlier amplitude bursts) instead of std.
    # Without this, motion-burst residuals inflate std and the prominence
    # threshold rises above weak true beats during quiet periods.
    mad = float(np.median(np.abs(h - np.median(h))))
    if not np.isfinite(mad) or mad <= 0:
        return np.array([], dtype=np.float64)
    prominence = prominence_factor * 1.4826 * mad  # 1.4826 → sigma-equivalent for Gaussian
    peaks, _ = find_peaks(h, distance=distance, prominence=prominence)
    if not refine or peaks.size == 0:
        return peaks.astype(np.float64) / fs
    return parabolic_refine(h, peaks) / fs


def parabolic_refine(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak refinement via 3-point parabolic interpolation."""
    out = peaks.astype(np.float64).copy()
    for k, idx in enumerate(peaks):
        if 0 < idx < signal.size - 1:
            a, b, c = signal[idx - 1], signal[idx], signal[idx + 1]
            denom = a - 2 * b + c
            if denom != 0:
                offset = 0.5 * (a - c) / denom
                # parabolic fit valid only for offsets in (-1, 1)
                if -1.0 < offset < 1.0:
                    out[k] = idx + offset
    return out


def peaks_to_ibi_ms(peak_times_s: np.ndarray) -> np.ndarray:
    """Successive differences in milliseconds."""
    if peak_times_s.size < 2:
        return np.array([], dtype=np.float64)
    return np.diff(peak_times_s) * 1000.0


def clean_ibi(
    ibi_ms: np.ndarray,
    *,
    low_ms: float = 300.0,
    high_ms: float = 1500.0,
    rel_tol: float = 0.30,
    median_window: int = 5,
) -> np.ndarray:
    """Bool mask: True where the interval is physiologic and within `rel_tol`
    of the local median.

    Local median uses a centered window of `median_window` intervals; if the
    series is shorter than the window we fall back to the global median.

    `rel_tol = 0.30` matches Kubios "medium" filter and preserves real RSA
    (respiratory sinus arrhythmia) variability, which can exceed 20% per
    breath cycle. Tighter tolerances bias SDNN/RMSSD toward zero by
    rejecting the very HRV variability they are meant to measure.
    """
    if ibi_ms.size == 0:
        return np.array([], dtype=bool)
    in_range = (ibi_ms >= low_ms) & (ibi_ms <= high_ms)

    if ibi_ms.size < median_window:
        local_med = np.full_like(ibi_ms, fill_value=float(np.median(ibi_ms)))
    else:
        # Manual centered rolling median that always uses k neighbors when possible.
        k = median_window
        half = k // 2
        local_med = np.empty_like(ibi_ms)
        for i in range(ibi_ms.size):
            lo = max(0, i - half)
            hi = min(ibi_ms.size, i + half + 1)
            local_med[i] = float(np.median(ibi_ms[lo:hi]))

    rel = np.abs(ibi_ms - local_med) / np.maximum(local_med, 1e-9)
    within_tol = rel <= rel_tol
    return in_range & within_tol
