"""End-to-end IBI/HRV pipeline runner.

Usage:
    # Real capture (MakeyMakey-style .lua + .bin or .npz):
    uv run python scripts/run_pipeline.py \\
        --cfg data/your_config.lua \\
        --data data/your_capture.npz \\
        --out  data/pipeline_run/

    # Self-test (no real data; uses radar_analysis.synthetic.synthetic_range_cube):
    uv run python scripts/run_pipeline.py --synthetic --out data/pipeline_synthetic/

Outputs in `--out`:
    range_fft.png   range_time.png   phase.png   heartbeat.png   ibi.png
    metrics.json    (chest bin, range, fs_slow, HRV summary, run params)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from radar_analysis import (  # noqa: E402
    RadarConfig,
    load_capture,
    run_pipeline,
    synthetic_range_cube,
)


def _load_real(cfg_path: Path, data_path: Path) -> tuple[np.ndarray, float, float]:
    """Returns (rfft_cube, range_res_m, fs_slow_hz).

    For complex-1x ADC (IWR1443 default) all N FFT bins are physical positive
    ranges 0 … (N-1)·range_res; we keep them all. Only with a real-ADC config
    would we slice to the first half (conjugate symmetry).
    """
    cfg = RadarConfig(cfg_path)
    params = cfg.get_params()
    raw = load_capture(data_path, params)
    rfft = np.fft.fft(raw, axis=2)
    if int(params.get("adc_output_fmt", 1)) == 0:
        rfft = rfft[..., : int(params["n_samples"]) // 2, :]
    fs_slow_hz = 1000.0 / float(params["frame_time"])  # frame_time is in ms
    return rfft, float(params["range_res"]), fs_slow_hz


def _load_synthetic() -> tuple[np.ndarray, float, float]:
    """Build a 30-s synthetic capture: HR=72 BPM (1.2 Hz), respiration 0.25 Hz."""
    range_res_m = 0.04
    fs_slow_hz = 25.0
    cube = synthetic_range_cube(
        n_frames=int(30 * fs_slow_hz),
        n_chirps=16,
        n_range_bins=64,
        n_rx=4,
        range_res_m=range_res_m,
        target_range_m=0.7,
        target_amp=2000.0,
        target_motion_amp_mm=1.0,
        target_motion_hz=1.2,
        clutter_range_m=1.2,
        clutter_amp=300.0,
        noise_std=10.0,
        fs_frame=fs_slow_hz,
        seed=0,
    )
    return cube, range_res_m, fs_slow_hz


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cfg", type=Path, help="Path to MakeyMakey-style .lua config")
    p.add_argument("--data", type=Path, help="Path to .bin or .npz capture")
    p.add_argument("--synthetic", action="store_true",
                   help="Run on a synthetic cube (no --cfg/--data needed)")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "pipeline_run",
                   help="Where to write plots + metrics.json")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation (still writes metrics.json)")
    args = p.parse_args()

    if args.synthetic:
        if args.cfg or args.data:
            p.error("--synthetic is mutually exclusive with --cfg/--data")
        rfft, range_res_m, fs_slow_hz = _load_synthetic()
        print("=== synthetic mode ===")
    else:
        if not (args.cfg and args.data):
            p.error("--cfg and --data are required (or use --synthetic)")
        if not args.cfg.exists():
            p.error(f"--cfg path does not exist: {args.cfg}")
        if not args.data.exists():
            p.error(f"--data path does not exist: {args.data}")
        try:
            rfft, range_res_m, fs_slow_hz = _load_real(args.cfg, args.data)
        except (KeyError, ValueError) as e:
            # RadarConfig regex doesn't yet handle TI's stock DataCaptureDemo_xWR.lua.
            p.error(f"failed to parse {args.cfg.name}: {e}")
        print(f"=== {args.data.name} ===")

    print(f"cube shape   : {rfft.shape}")
    print(f"range_res_m  : {range_res_m:.4f}")
    print(f"fs_slow_hz   : {fs_slow_hz:.2f}")
    print(f"duration_s   : {rfft.shape[0] / fs_slow_hz:.2f}")

    result = run_pipeline(
        rfft,
        range_res_m=range_res_m,
        fs_slow_hz=fs_slow_hz,
        out_dir=args.out,
        save_plots=not args.no_plots,
    )

    print()
    print(json.dumps(result.to_summary(), indent=2))
    print(f"\nartifacts: {args.out}/")


if __name__ == "__main__":
    main()
