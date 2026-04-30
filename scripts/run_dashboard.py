"""Launch the mmWave HRV dashboard.

Usage:
    # synthetic 5-minute capture (HR=72 BPM ground truth):
    uv run python scripts/run_dashboard.py --synthetic

    # real capture (MakeyMakey-style .lua + .npz/.bin):
    uv run python scripts/run_dashboard.py --cfg path/to.lua --data path/to.npz

The browser opens automatically at http://127.0.0.1:8050. Append
`?view=subject` for the calm full-screen subject view.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.app import make_app  # noqa: E402
from dashboard.state import DashboardState  # noqa: E402
from radar_analysis.streams import (  # noqa: E402
    NpzReplaySource,
    SyntheticReplaySource,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--cfg", type=Path, help="Path to MakeyMakey-style .lua")
    p.add_argument("--data", type=Path, help="Path to .npz or .bin")
    p.add_argument("--synthetic", action="store_true",
                   help="Use a 5-min synthetic capture (HR=72 BPM ground truth)")
    p.add_argument("--duration", type=float, default=300.0,
                   help="Synthetic capture length in seconds (default 300)")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--no-browser", action="store_true",
                   help="Don't auto-open the browser")
    args = p.parse_args()

    if args.synthetic:
        if args.cfg or args.data:
            p.error("--synthetic is mutually exclusive with --cfg/--data")
        source = SyntheticReplaySource(
            duration_s=args.duration, fs_slow_hz=25.0, target_motion_hz=1.2,
            seed=0, realtime=True,
        )
        mode = "synthetic"
    else:
        if not (args.cfg and args.data):
            p.error("--cfg and --data are required (or use --synthetic)")
        try:
            source = NpzReplaySource(args.cfg, args.data, realtime=True)
        except Exception as e:
            # RadarConfig + load_capture surface a wide variety of errors
            # (KeyError on missing Lua keys, ValueError on shape mismatch,
            # OSError on missing files, etc.).
            p.error(f"failed to load capture: {type(e).__name__}: {e}")
        mode = "replay"

    state = DashboardState(source, window_s=10.0, recompute_period_s=0.25,
                           plot_window_s=10.0)
    state.start()
    try:
        app = make_app(state, tick_ms=200)

        url = f"http://127.0.0.1:{args.port}/"
        print(f"=== mmWave HRV dashboard ({mode}) ===")
        print(f"operator view: {url}")
        print(f"subject  view: {url}?view=subject")
        print("Ctrl+C to stop.")

        if not args.no_browser:
            threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)),
                             daemon=True).start()

        app.run(host="127.0.0.1", port=args.port, debug=False)
    finally:
        state.stop()


if __name__ == "__main__":
    main()
