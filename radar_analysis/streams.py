"""Frame sources for the live/replay dashboard.

Step-1 ships two replay sources (synthetic + .npz). Step-2 will add a
`UdpDca1000Source`; the dashboard never imports concrete sources, only
the `FrameSource` protocol below — so the live socket lands as a 1-file
swap.

A "frame" here is a single slow-time sample after range-FFT, shape
`(n_chirps, n_range_bins, n_rx)` complex64. The dashboard accumulates
frames in a ring buffer and feeds windows to `run_pipeline`.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from radar_analysis.radar_config import RadarConfig
from radar_analysis.reader import load_capture
from radar_analysis.synthetic import synthetic_range_cube


@runtime_checkable
class FrameSource(Protocol):
    """Yield `(n_chirps, n_range_bins, n_rx)` complex64 frames at `fs_slow_hz`."""

    fs_slow_hz: float
    range_res_m: float
    n_chirps: int
    n_range_bins: int
    n_rx: int

    def __iter__(self) -> Iterator[np.ndarray]: ...
    def close(self) -> None: ...


class SyntheticReplaySource:
    """Synthetic 5-min chest signal for dashboard development & demos."""

    def __init__(
        self,
        *,
        duration_s: float = 300.0,
        fs_slow_hz: float = 25.0,
        range_res_m: float = 0.04,
        n_chirps: int = 16,
        n_range_bins: int = 64,
        n_rx: int = 4,
        target_range_m: float = 0.7,
        target_motion_hz: float = 1.2,
        target_motion_amp_mm: float = 1.0,
        seed: int | None = 0,
        realtime: bool = True,
    ) -> None:
        self.fs_slow_hz = float(fs_slow_hz)
        self.range_res_m = float(range_res_m)
        self.n_chirps = int(n_chirps)
        self.n_range_bins = int(n_range_bins)
        self.n_rx = int(n_rx)
        self._realtime = bool(realtime)
        self._cube = synthetic_range_cube(
            n_frames=int(duration_s * fs_slow_hz),
            n_chirps=n_chirps,
            n_range_bins=n_range_bins,
            n_rx=n_rx,
            range_res_m=range_res_m,
            target_range_m=target_range_m,
            target_amp=2000.0,
            target_motion_amp_mm=target_motion_amp_mm,
            target_motion_hz=target_motion_hz,
            clutter_range_m=1.2,
            clutter_amp=300.0,
            noise_std=10.0,
            fs_frame=fs_slow_hz,
            seed=seed,
        )
        self._closed = False

    def __iter__(self) -> Iterator[np.ndarray]:
        period = 1.0 / self.fs_slow_hz
        start = time.perf_counter()
        for i in range(self._cube.shape[0]):
            if self._closed:
                return
            if self._realtime:
                target = start + i * period
                gap = target - time.perf_counter()
                if gap > 0:
                    time.sleep(gap)
            yield self._cube[i]

    def close(self) -> None:
        self._closed = True


class NpzReplaySource:
    """Replays a `.npz` capture (or `.bin` with a paired `.lua`) at its native rate."""

    def __init__(
        self,
        cfg_path: str | Path,
        data_path: str | Path,
        *,
        realtime: bool = True,
    ) -> None:
        cfg = RadarConfig(cfg_path)
        params = cfg.get_params()
        raw = load_capture(data_path, params)             # (F, C, S, R) complex
        # Range FFT along fast time. Keep all bins for complex 1x ADC.
        rfft = np.fft.fft(raw, axis=2)
        if int(params.get("adc_output_fmt", 1)) == 0:
            rfft = rfft[..., : int(params["n_samples"]) // 2, :]
        self._cube = rfft.astype(np.complex64, copy=False)

        self.fs_slow_hz = 1000.0 / float(params["frame_time"])
        self.range_res_m = float(params["range_res"])
        self.n_chirps = int(self._cube.shape[1])
        self.n_range_bins = int(self._cube.shape[2])
        self.n_rx = int(self._cube.shape[3])
        self._realtime = bool(realtime)
        self._closed = False

    def __iter__(self) -> Iterator[np.ndarray]:
        period = 1.0 / self.fs_slow_hz
        start = time.perf_counter()
        for i in range(self._cube.shape[0]):
            if self._closed:
                return
            if self._realtime:
                target = start + i * period
                gap = target - time.perf_counter()
                if gap > 0:
                    time.sleep(gap)
            yield self._cube[i]

    def close(self) -> None:
        self._closed = True
