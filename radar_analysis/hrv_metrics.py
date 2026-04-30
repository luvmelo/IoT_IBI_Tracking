"""Time-domain HRV metrics following the Task Force 1996 standard.

All functions take an array of NN intervals in milliseconds (already
artifact-corrected; pass `ibi_ms[clean_ibi(ibi_ms)]`). All formulas use
the sample-stat denominator `N-1` for SDNN and the diff-count denominator
`N-1` for RMSSD, matching the Task Force convention. pNN50 is reported
in percent (0–100).

Refs: Eur. Heart J. 1996;17:354–381 §IV. Standard formulas summarized in
Shaffer & Ginsberg, Front. Public Health 2017.
"""

from __future__ import annotations

import numpy as np


def _check(nn: np.ndarray, min_n: int = 2) -> np.ndarray:
    arr = np.asarray(nn, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"NN intervals must be 1-D, got shape {arr.shape}")
    if arr.size < min_n:
        raise ValueError(f"need >= {min_n} NN intervals, got {arr.size}")
    return arr


def mean_ibi_ms(nn: np.ndarray) -> float:
    arr = _check(nn, min_n=1)
    return float(arr.mean())


def mean_hr_bpm(nn: np.ndarray) -> float:
    return 60_000.0 / mean_ibi_ms(nn)


def sdnn_ms(nn: np.ndarray) -> float:
    """Standard deviation of NN intervals (sample, N-1 denominator)."""
    arr = _check(nn, min_n=2)
    return float(arr.std(ddof=1))


def rmssd_ms(nn: np.ndarray) -> float:
    """Root-mean-square of successive differences. Uses M-1 = (N-2)?

    Task Force 1996 defines RMSSD = sqrt( (1/(N-1)) Σ (NN_{i+1} - NN_i)^2 ),
    where the sum runs over the N-1 successive differences. We follow that.
    """
    arr = _check(nn, min_n=2)
    diffs = np.diff(arr)
    return float(np.sqrt(np.sum(diffs ** 2) / diffs.size))


def pnn50(nn: np.ndarray) -> float:
    """Percent of successive differences strictly greater than 50 ms."""
    arr = _check(nn, min_n=2)
    diffs = np.abs(np.diff(arr))
    return 100.0 * float(np.mean(diffs > 50.0))
