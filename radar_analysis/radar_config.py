"""Parser for the .lua scripts mmWave Studio uses to configure the IWR1443.

We don't execute Lua — we just regex-extract the variable assignments and
the `ar1.*` function calls, then derive frame shape and physical limits.

Adapted from r-bt/MakeyMakey/src/xwr/radar_config.py, which itself adapted
ConnectedSystemsLab/xwr_raw_ros. Kept stdlib-only so it runs on the Mac
without numba.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path
from typing import Any


class RadarConfig(OrderedDict):
    platforms = {"XWR1443": "xWR14xx"}

    def __init__(self, cfg: str | Path):
        super().__init__()
        self._path = Path(cfg)
        self._parse(self._path)

    def _parse(self, path: Path) -> None:
        with path.open("r") as f:
            for line in f:
                m = re.match(r"(\w+)\s*=\s*([^\n]*?)(?=\s*--|$)", line)
                if m:
                    k, v = m.groups()
                    self[k] = self._coerce(v)
                    continue
                m = re.match(r"([\w\.]+)\((.+?)\)", line)
                if m:
                    fn, args = m.groups()
                    self[fn] = [a.strip() for a in args.split(",")]

    @staticmethod
    def _coerce(value: str) -> Any:
        v = value.strip().strip('"').strip("'")
        try:
            return float(v) if "." in v else int(v)
        except ValueError:
            return v

    def get_params(self) -> "OrderedDict[str, Any]":
        chip = self["ar1.SelectChipVersion"][0].strip('"').strip("'")
        platform = self.platforms.get(chip, "Unknown")

        # ar1.ChanNAdcConfig args: TX1,TX2,TX3, RX1,RX2,RX3,RX4, _, adc_fmt, _, _
        adc_output_fmt = int(self["ar1.ChanNAdcConfig"][8])  # 0=real, 1=cplx1x, 2=cplx2x

        start_chirp = int(self["START_CHIRP_TX"])
        end_chirp = int(self["END_CHIRP_TX"])
        chirp_loops = int(self["CHIRP_LOOPS"])
        n_chirps = (end_chirp - start_chirp + 1) * chirp_loops

        rx = [int(x) for x in self["ar1.ChanNAdcConfig"][3:7]]
        n_rx = sum(rx)
        tx = [int(x) for x in self["ar1.ChanNAdcConfig"][0:3]]
        n_tx = sum(tx)

        n_samples = int(self["ADC_SAMPLES"])

        # 2 bytes per int16 sample, 2x for I+Q if complex
        bytes_per_sample = 2 * (2 if adc_output_fmt > 0 else 1)
        frame_size = n_samples * n_rx * n_chirps * bytes_per_sample

        frame_time = self["PERIODICITY"]
        chirp_time = self["IDLE_TIME"] + self["RAMP_END_TIME"]
        chirp_slope = self["FREQ_SLOPE"] * 1e12
        sample_rate = self["SAMPLE_RATE"] * 1e3
        t_sweep = n_samples / sample_rate
        chirp_sampling_rate = 1.0 / (chirp_time * 1e-6)

        operating_freq = self["START_FREQ"] * 1e9
        wavelength = 3e8 / operating_freq
        velocity_max = wavelength / (4 * chirp_time * 1e-6)
        velocity_res = velocity_max / n_chirps

        range_max = (sample_rate * 3e8) / (2 * chirp_slope)
        range_res = range_max / n_samples

        return OrderedDict(
            platform=platform,
            adc_output_fmt=adc_output_fmt,
            n_chirps=n_chirps,
            rx=rx,
            n_rx=n_rx,
            tx=tx,
            n_tx=n_tx,
            n_samples=n_samples,
            frame_size=frame_size,
            frame_time=frame_time,
            chirp_time=chirp_time,
            chirp_slope=chirp_slope,
            sample_rate=sample_rate,
            chirp_sampling_rate=chirp_sampling_rate,
            velocity_max=velocity_max,
            velocity_res=velocity_res,
            range_max=range_max,
            range_res=range_res,
            t_sweep=t_sweep,
        )

    def __str__(self) -> str:
        try:
            params = self.get_params()
        except Exception as e:
            return f"Error generating config parameters: {e}"

        units = {
            "frame_time": "ms",
            "chirp_time": "us",
            "chirp_slope": "Hz/s",
            "sample_rate": "samples/s",
            "chirp_sampling_rate": "Hz",
            "velocity_max": "m/s",
            "velocity_res": "m/s",
            "range_max": "m",
            "range_res": "m",
            "t_sweep": "s",
            "frame_size": "bytes",
        }
        lines = []
        for k, v in params.items():
            if isinstance(v, list):
                s = ", ".join(str(x) for x in v)
            elif isinstance(v, float):
                s = f"{v:.4f}"
            else:
                s = str(v)
            unit = units.get(k, "")
            lines.append(f"{k:25}: {s} {unit}".rstrip())
        return "\n".join(lines)
