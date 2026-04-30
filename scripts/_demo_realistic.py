"""Generate a realistic-ish synthetic capture so you can see what the
inspect tool actually shows. Two stationary targets at 2 m and 5 m.

Run from project root: `uv run python scripts/_demo_realistic.py`
Then:                  `uv run python scripts/inspect_capture.py
                            --cfg data/_demo.lua --data data/_demo.bin`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from radar_analysis import RadarConfig

CFG_LUA = """\
COM_PORT = 3
NUM_TX = 3
NUM_RX = 4
START_FREQ = 77
IDLE_TIME = 138
RAMP_END_TIME = 62
ADC_START_TIME = 6
FREQ_SLOPE = 60.012
ADC_SAMPLES = 512
SAMPLE_RATE = 10000
RX_GAIN = 30
START_CHIRP_TX = 0
END_CHIRP_TX = 0
NUM_FRAMES = 0
CHIRP_LOOPS = 64
PERIODICITY = 100
ar1.SelectChipVersion("XWR1443")
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
"""


def make_target_chirp(n_samples: int, range_m: float, range_res_m: float,
                      amplitude: float) -> np.ndarray:
    """Single chirp's IF signal for a stationary point target at `range_m`."""
    bin_idx = range_m / range_res_m
    t = np.arange(n_samples)
    return amplitude * np.exp(1j * 2 * np.pi * bin_idx * t / n_samples)


def main() -> None:
    cfg_path = ROOT / "data" / "_demo.lua"
    bin_path = ROOT / "data" / "_demo.bin"
    cfg_path.write_text(CFG_LUA)

    cfg = RadarConfig(cfg_path)
    p = cfg.get_params()
    n_frames = 5
    n_chirps, n_samples, n_rx = p["n_chirps"], p["n_samples"], p["n_rx"]
    range_res = float(p["range_res"])

    rng = np.random.default_rng(0)
    chirp_2m = make_target_chirp(n_samples, 2.0, range_res, amplitude=2500)
    chirp_5m = make_target_chirp(n_samples, 5.0, range_res, amplitude=1500)
    target = chirp_2m + chirp_5m

    n_int16 = n_frames * n_chirps * n_samples * n_rx * 2
    flat = np.empty(n_int16, dtype=np.int16).reshape(-1, 8)
    iq = np.empty((n_frames * n_chirps * n_samples, n_rx), dtype=np.complex64)
    for rx in range(n_rx):
        per_rx = np.tile(target, n_chirps * n_frames)
        per_rx = per_rx + rng.normal(0, 50, per_rx.size) \
                       + 1j * rng.normal(0, 50, per_rx.size)
        iq[:, rx] = per_rx.astype(np.complex64)

    flat[:, :n_rx] = np.real(iq).astype(np.int16)
    flat[:, n_rx:] = np.imag(iq).astype(np.int16)
    bin_path.write_bytes(flat.reshape(-1).tobytes())

    print(f"Wrote {bin_path.name} ({bin_path.stat().st_size / 1024:.1f} KB), "
          f"{n_frames} frames, targets at 2 m and 5 m")


if __name__ == "__main__":
    main()
