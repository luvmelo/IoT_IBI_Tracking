"""Offline analysis utilities for IWR1443 + DCA1000 captures.

Designed to run on macOS / Linux *without* the radar hardware. The capture
side (mmWave Studio + Lua + DCA1000) lives on a Windows machine; this
package only consumes the artifacts it produces (`.lua` config + raw
`.bin` or reshaped `.npz`).
"""

from radar_analysis.radar_config import RadarConfig
from radar_analysis.reader import (
    load_capture,
    load_bin,
    load_npz,
    reshape_iwr1443_frame,
)
from radar_analysis.synthetic import radar_phase_signal, synthetic_range_cube
from radar_analysis.phase_processing import (
    remove_dc,
    circle_fit_dc,
    coherent_combine_rx,
    extract_phase,
    detrend_median,
    despike_hampel,
    motion_mask,
)
from radar_analysis.chest_bin_selection import select_chest_bin
from radar_analysis.heartbeat_extractors import (
    bandpass,
    extract_respiration,
    extract_heartbeat,
)
from radar_analysis.beat_detection import (
    detect_beats,
    parabolic_refine,
    peaks_to_ibi_ms,
    clean_ibi,
)
from radar_analysis.hrv_metrics import (
    mean_ibi_ms,
    mean_hr_bpm,
    sdnn_ms,
    rmssd_ms,
    pnn50,
)
from radar_analysis.pipeline import PipelineResult, run_pipeline

__all__ = [
    "RadarConfig",
    "load_capture",
    "load_bin",
    "load_npz",
    "reshape_iwr1443_frame",
    "radar_phase_signal",
    "synthetic_range_cube",
    "remove_dc",
    "circle_fit_dc",
    "coherent_combine_rx",
    "extract_phase",
    "detrend_median",
    "despike_hampel",
    "motion_mask",
    "select_chest_bin",
    "bandpass",
    "extract_respiration",
    "extract_heartbeat",
    "detect_beats",
    "parabolic_refine",
    "peaks_to_ibi_ms",
    "clean_ibi",
    "mean_ibi_ms",
    "mean_hr_bpm",
    "sdnn_ms",
    "rmssd_ms",
    "pnn50",
    "PipelineResult",
    "run_pipeline",
]
