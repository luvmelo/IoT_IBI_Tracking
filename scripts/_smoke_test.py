"""Synthetic round-trip test: build a fake capture matching the MakeyMakey
default config, write it to .bin, and verify the reader recovers the original
shape and a clean tone in the range FFT.

Run from project root: `uv run python scripts/_smoke_test.py`
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from radar_analysis import RadarConfig, load_bin, reshape_iwr1443_frame


CFG_LUA = """\
COM_PORT = 3
RADARSS_PATH = "C:\\\\fake.bin"
MASTERSS_PATH = "C:\\\\fake.bin"
NUM_TX = 3
NUM_RX = 4
START_FREQ = 77
IDLE_TIME = 138
RAMP_END_TIME = 62
ADC_START_TIME = 6
FREQ_SLOPE = 60.012
ADC_SAMPLES = 64
SAMPLE_RATE = 10000
RX_GAIN = 30
START_CHIRP_TX = 0
END_CHIRP_TX = 0
NUM_FRAMES = 0
CHIRP_LOOPS = 8
PERIODICITY = 100
ar1.SelectChipVersion("XWR1443")
ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
"""


def build_synthetic_frame(n_chirps: int, n_samples: int, n_rx: int,
                          tone_bin: int) -> np.ndarray:
    """Build a frame with a clean tone at fast-time bin `tone_bin`, then
    pack it into the IWR1443 LVDS layout (groups of 8 int16: I0..I3, Q0..Q3).
    """
    t = np.arange(n_samples)
    base = np.exp(1j * 2 * np.pi * tone_bin * t / n_samples) * 1000
    frame = np.broadcast_to(base[None, :, None],
                            (n_chirps, n_samples, n_rx)).astype(np.complex64)
    flat = np.empty(n_chirps * n_samples * n_rx * 2, dtype=np.int16)
    flat = flat.reshape(-1, 8)
    iq = frame.reshape(-1, n_rx)
    flat[:, :n_rx] = np.real(iq).astype(np.int16)
    flat[:, n_rx:] = np.imag(iq).astype(np.int16)
    return flat.reshape(-1)


def main() -> None:
    cfg_path = ROOT / "data" / "_smoke.lua"
    bin_path = ROOT / "data" / "_smoke.bin"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(CFG_LUA)

    cfg = RadarConfig(cfg_path)
    params = cfg.get_params()
    print(cfg)
    print()

    n_frames = 3
    tone_bin = 7
    frames = np.concatenate([
        build_synthetic_frame(
            params["n_chirps"], params["n_samples"], params["n_rx"], tone_bin
        )
        for _ in range(n_frames)
    ])
    bin_path.write_bytes(frames.tobytes())

    data = load_bin(
        bin_path,
        n_chirps_per_frame=params["n_chirps"],
        samples_per_chirp=params["n_samples"],
        n_receivers=params["n_rx"],
    )
    expected_shape = (n_frames, params["n_chirps"], params["n_samples"],
                      params["n_rx"])
    assert data.shape == expected_shape, (data.shape, expected_shape)

    rfft = np.fft.fft(data[0], axis=1)
    peak_bin = int(np.argmax(np.abs(rfft[0, :, 0])))
    assert peak_bin == tone_bin, (peak_bin, tone_bin)

    direct = reshape_iwr1443_frame(
        frames[: params["n_chirps"] * params["n_samples"] * params["n_rx"] * 2],
        params["n_chirps"], params["n_samples"], params["n_rx"],
    )
    assert np.allclose(direct, data[0])

    print(f"OK: shape {data.shape}, tone recovered at bin {peak_bin}")


if __name__ == "__main__":
    main()
