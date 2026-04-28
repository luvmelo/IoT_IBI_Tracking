"""Read DCA1000 captures into shape (n_frames, n_chirps, n_samples, n_rx) complex.

Two input formats supported:

1. **`.npz`** produced by MakeyMakey's `record.py`. The Python listener
   already reshaped frames per `dsp.reshape_frame` (IWR1443: 4 RX, 1 TX,
   complex 1x). The file has a single `data` key with shape
   `(n_frames, n_chirps, n_samples, n_rx)` complex.

2. **Raw `.bin`** produced by mmWave Studio itself
   (`ar1.CaptureCardConfig_StartRecord(SAVE_DATA_PATH, 1)`). This is the
   DCA1000 byte stream with header bytes stripped — interleaved int16 IQ.
   We reshape it here using the same logic the live listener uses, so the
   output is identical.

The reshape rule for the IWR1443 BOOST (4 RX, complex 1x, non-interleaved
LVDS) is documented in PDF section 24.6: each LVDS group of 8 int16 words
is `[I_rx0, I_rx1, I_rx2, I_rx3, Q_rx0, Q_rx1, Q_rx2, Q_rx3]`. With a
single TX (the MakeyMakey default), no TDM deinterleave is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def reshape_iwr1443_frame(
    raw_int16: np.ndarray,
    n_chirps_per_frame: int,
    samples_per_chirp: int,
    n_receivers: int = 4,
) -> np.ndarray:
    """Reshape a flat int16 frame buffer into (n_chirps, n_samples, n_rx) complex.

    `raw_int16` is the int16 view of the DCA1000 stream for one frame.
    """
    if n_receivers != 4:
        raise NotImplementedError(
            "IWR1443 reshape assumes 4 RX antennas; got n_receivers="
            f"{n_receivers}"
        )

    expected = n_chirps_per_frame * samples_per_chirp * n_receivers * 2
    if raw_int16.size != expected:
        raise ValueError(
            f"Frame size mismatch: got {raw_int16.size} int16 samples, "
            f"expected {expected} (n_chirps={n_chirps_per_frame}, "
            f"n_samples={samples_per_chirp}, n_rx={n_receivers}, IQ)"
        )

    grouped = raw_int16.reshape(-1, 8).astype(np.float32)
    iq = grouped[:, :4] + 1j * grouped[:, 4:]
    return iq.reshape(n_chirps_per_frame, samples_per_chirp, n_receivers)


def load_bin(
    path: str | Path,
    n_chirps_per_frame: int,
    samples_per_chirp: int,
    n_receivers: int = 4,
    *,
    n_frames: int | None = None,
    drop_partial: bool = True,
) -> np.ndarray:
    """Load a raw DCA1000 `.bin` file into (n_frames, n_chirps, n_samples, n_rx).

    Args:
        path: path to e.g. `adc_data.bin`.
        n_chirps_per_frame, samples_per_chirp, n_receivers: from the Lua config.
        n_frames: clamp to this many frames if set; else infer from file size.
        drop_partial: if True, silently drop a trailing partial frame.
    """
    path = Path(path)
    raw = np.fromfile(path, dtype=np.int16)
    samples_per_frame = n_chirps_per_frame * samples_per_chirp * n_receivers * 2

    total_frames = raw.size // samples_per_frame
    leftover = raw.size - total_frames * samples_per_frame
    if leftover and not drop_partial:
        raise ValueError(
            f"{path.name} has {leftover} trailing int16 samples that don't "
            "fill a frame; pass drop_partial=True to ignore."
        )

    if n_frames is not None:
        total_frames = min(total_frames, n_frames)

    raw = raw[: total_frames * samples_per_frame]

    out = np.empty(
        (total_frames, n_chirps_per_frame, samples_per_chirp, n_receivers),
        dtype=np.complex64,
    )
    for i in range(total_frames):
        chunk = raw[i * samples_per_frame : (i + 1) * samples_per_frame]
        out[i] = reshape_iwr1443_frame(
            chunk, n_chirps_per_frame, samples_per_chirp, n_receivers
        )
    return out


def load_npz(path: str | Path) -> np.ndarray:
    """Load `.npz` produced by MakeyMakey's `record.py`.

    The file stores one array under the `data` key with shape already in
    `(n_frames, n_chirps, n_samples, n_rx)` complex form.
    """
    with np.load(Path(path), allow_pickle=False) as f:
        if "data" not in f.files:
            raise KeyError(
                f"{path}: expected a 'data' key (got {f.files}); is this a "
                "MakeyMakey record.py output?"
            )
        return f["data"]


def load_capture(
    data_path: str | Path,
    params: Mapping[str, Any],
) -> np.ndarray:
    """Auto-dispatch on file extension. Pass in `RadarConfig.get_params()`."""
    data_path = Path(data_path)
    suffix = data_path.suffix.lower()
    if suffix == ".npz":
        return load_npz(data_path)
    if suffix == ".bin":
        return load_bin(
            data_path,
            n_chirps_per_frame=int(params["n_chirps"]),
            samples_per_chirp=int(params["n_samples"]),
            n_receivers=int(params["n_rx"]),
        )
    raise ValueError(
        f"Unsupported capture extension '{suffix}'. Expected .bin or .npz."
    )
