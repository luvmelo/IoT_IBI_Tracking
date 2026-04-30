"""Bandpass extractors for respiration & heartbeat from a slow-time phase signal.

Defaults follow the TI vital-signs lab guide and the parameter cheat-sheet
in `mmwave_ibi_hrv_research_note.md`:

    Respiration : 0.10 – 0.60 Hz   (6–36 breaths/min)
    Heartbeat   : 0.80 – 4.00 Hz   (48–240 BPM)

Implementation: Butterworth order-4 IIR + zero-phase `filtfilt`, so beat
timing isn't biased by group delay. VMD-based extractor (Stage 4 of the
plan) is intentionally deferred — wire it in once the bandpass baseline
is validated end-to-end.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    *,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass. Raises if cutoffs are outside (0, fs/2)."""
    nyq = 0.5 * fs
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(
            f"need 0 < low_hz < high_hz < fs/2; got low={low_hz}, high={high_hz}, fs={fs}"
        )
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype="band")
    # filtfilt's default padlen is 3*max(len(a), len(b)); for a band-pass order=4
    # that's ~27 samples. Short motion-gated segments would crash with a
    # cryptic message — fail loudly with the actual numbers instead.
    min_len = 3 * max(len(a), len(b))
    if x.shape[-1] < min_len:
        raise ValueError(
            f"bandpass needs at least {min_len} samples for order={order} "
            f"(filtfilt padlen); got {x.shape[-1]}. Either lower the order "
            f"or splice longer segments."
        )
    return filtfilt(b, a, x)


def extract_respiration(
    phi: np.ndarray,
    fs: float,
    *,
    low_hz: float = 0.10,
    high_hz: float = 0.60,
) -> np.ndarray:
    """Default 0.10–0.60 Hz band."""
    return bandpass(phi, fs, low_hz, high_hz)


def extract_heartbeat(
    phi: np.ndarray,
    fs: float,
    *,
    low_hz: float = 0.80,
    high_hz: float = 4.00,
) -> np.ndarray:
    """Default 0.80–4.00 Hz band (48–240 BPM)."""
    return bandpass(phi, fs, low_hz, high_hz)
