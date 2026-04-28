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

__all__ = [
    "RadarConfig",
    "load_capture",
    "load_bin",
    "load_npz",
    "reshape_iwr1443_frame",
]
