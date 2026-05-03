"""Pick the chest range-bin from a post-range-FFT cube.

Input shape: `(n_frames, n_chirps, n_range_bins, n_rx)` complex (output of
`np.fft.fft(load_capture(...), axis=2)[..., :n_range_bins, :]`).

Score = mean per-bin power within the physical search window, optionally
multiplied by a normalized slow-time phase-variance term to prefer bins
where something is actually moving (chest) over static high-power clutter
(walls, table edges). This is the TI vital-signs lab criterion plus the
motion-variance enhancement noted in the local research note.
"""

from __future__ import annotations

import numpy as np


def select_chest_bin(
    rfft: np.ndarray,
    range_res_m: float,
    *,
    search_window_m: tuple[float, float] = (0.3, 2.5),
    fs_slow_hz: float | None = None,
    motion_band_hz: tuple[float, float] = (0.8, 3.0),
    use_motion_variance: bool = True,
    motion_weight: float = 1.0,
) -> tuple[int, float]:
    """Return `(bin_idx, score)` for the strongest moving bin in the search window.

    When `fs_slow_hz` is provided, the motion-variance term is computed on
    phase bandpassed to `motion_band_hz` (defaults to the heart-rate band).
    Without this, raw phase variance is dominated by respiration / body sway
    and the picker happily selects bins where the chest *isn't* — which is
    what happened on real captures with default settings.
    """
    if rfft.ndim != 4:
        raise ValueError(f"rfft must be 4-D (F, C, S, R); got shape {rfft.shape}")
    n_range_bins = rfft.shape[2]

    bin_lo, bin_hi = search_window_m
    if bin_lo < 0 or bin_hi <= bin_lo:
        raise ValueError(f"search_window_m must be (lo, hi) with hi > lo >= 0, got {search_window_m}")

    # Bin 0 is the DC bin — always dominated by static clutter / radar self-echo;
    # we never want to pick it for the chest, so the lower bound is clamped to 1
    # even if the user passes search_window_m=(0.0, …).
    bin_min = max(1, int(np.ceil(bin_lo / range_res_m)))
    bin_max = min(n_range_bins, int(np.floor(bin_hi / range_res_m)) + 1)
    if bin_max <= bin_min:
        raise ValueError(
            f"search window {search_window_m} m yields no valid bins at "
            f"range_res_m={range_res_m} (bin_min={bin_min}, bin_max={bin_max})"
        )

    # Coherent integration across RX, then power per (frame, chirp, bin)
    coh = rfft.mean(axis=3)                       # (F, C, S)
    power_per_bin = (np.abs(coh) ** 2).mean(axis=(0, 1))   # (S,)

    score = power_per_bin.copy()

    if use_motion_variance:
        # Average chirps within each frame to one complex value per (frame, bin)
        per_frame = coh.mean(axis=1)              # (F, S)
        phase = np.unwrap(np.angle(per_frame), axis=0)
        if fs_slow_hz is not None:
            from radar_analysis.heartbeat_extractors import bandpass
            try:
                # filtfilt operates on the last axis; transpose so per-bin
                # phase becomes the inner dim, then transpose back.
                phase_band = bandpass(
                    phase.T, fs_slow_hz, motion_band_hz[0], motion_band_hz[1]
                ).T
                phase_var = phase_band.var(axis=0)
            except ValueError:
                # Capture too short for filtfilt's padlen — fall back to the
                # full-band variance with a clear name for debugging.
                phase_var = phase.var(axis=0)
        else:
            phase_var = phase.var(axis=0)
        denom = float(phase_var.max())
        phase_var_norm = phase_var / denom if denom > 0 else np.zeros_like(phase_var)
        score = score * (1.0 + motion_weight * phase_var_norm)

    masked = np.full_like(score, -np.inf)
    masked[bin_min:bin_max] = score[bin_min:bin_max]
    best = int(np.argmax(masked))
    return best, float(masked[best])
