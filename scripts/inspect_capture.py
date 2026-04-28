"""Inspect a DCA1000 capture: parse the Lua, load the data, plot a range-FFT.

Usage:
    uv run python scripts/inspect_capture.py \
        --cfg path/to/1443_capture.lua \
        --data path/to/adc_data.bin
        # or --data path/to/radar_data_*.npz

Outputs:
    - prints derived radar params (range res, max range, frame size, ...)
    - prints capture stats (n_frames, mean magnitude, NaN count)
    - saves data/inspect_range_fft.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from radar_analysis import RadarConfig, load_capture  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cfg", required=True, type=Path, help="Path to .lua config")
    p.add_argument("--data", required=True, type=Path, help="Path to .bin or .npz")
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "inspect_range_fft.png",
        help="Where to save the range-FFT plot",
    )
    p.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Which frame to plot (default: 0)",
    )
    args = p.parse_args()

    cfg = RadarConfig(args.cfg)
    params = cfg.get_params()
    print("=== Radar params (from Lua) ===")
    print(cfg)

    print("\n=== Loading capture ===")
    data = load_capture(args.data, params)
    print(f"shape  : {data.shape}  (n_frames, n_chirps, n_samples, n_rx)")
    print(f"dtype  : {data.dtype}")
    mag = np.abs(data)
    print(f"|x| min/mean/max : {mag.min():.2f} / {mag.mean():.2f} / {mag.max():.2f}")
    print(f"NaNs   : {int(np.isnan(mag).sum())}")
    print(f"Zero frames (all-zero, possibly dropped): "
          f"{int(np.sum(mag.reshape(data.shape[0], -1).max(axis=1) == 0))}"
          f" / {data.shape[0]}")

    if args.frame >= data.shape[0]:
        print(f"\n[warn] --frame {args.frame} >= n_frames {data.shape[0]}; "
              f"using frame 0")
        args.frame = 0

    frame = data[args.frame]                      # (n_chirps, n_samples, n_rx)
    range_fft = np.fft.fft(frame, axis=1)         # FFT across fast-time
    range_fft = range_fft[:, : frame.shape[1] // 2, :]
    mean_over_chirps = np.mean(np.abs(range_fft), axis=0)  # (n_samples/2, n_rx)

    range_res = float(params["range_res"])
    range_axis = np.arange(mean_over_chirps.shape[0]) * range_res

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for rx in range(mean_over_chirps.shape[1]):
        ax.plot(range_axis, 20 * np.log10(mean_over_chirps[:, rx] + 1e-9),
                label=f"RX{rx}")
    ax.set_xlabel("Range (m)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title(
        f"Range FFT — frame {args.frame} of {data.shape[0]} "
        f"({args.data.name})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
