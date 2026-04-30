"""Quick range-FFT + slow-time phase plot for the TI DataCaptureDemo capture.

Params are hand-extracted because `RadarConfig` doesn't yet parse TI's stock
`DataCaptureDemo_xWR.lua` style (parameters live inside `ar1.ProfileConfig` /
`ar1.FrameConfig` arg lists, gated by partId branches). When the parser is
extended to handle that flavor, replace the constants below with
`RadarConfig(...).get_params()`.

Run from project root:
    uv run python scripts/quick_phase.py

Outputs (in data/):
    quick_range_fft.png    -- mean |range-FFT| across chirps × frames per RX
    quick_range_time.png   -- range-bin heatmap over chirp time (RX0)
    quick_phase.png        -- unwrapped phase (per-chirp + per-frame mm)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from radar_analysis import load_bin  # noqa: E402

# --- Hand-extracted from DataCaptureDemo_xWR.lua (partId == 1443 branch) -------
N_FRAMES = 8                       # FrameConfig arg 3
N_CHIRPS = 128                     # FrameConfig arg 4 (chirps per frame)
N_SAMPLES = 256                    # ProfileConfig numAdcSamples
N_RX = 4                           # ChanNAdcConfig RX1..RX4 = 1
FRAME_PERIOD_S = 40e-3             # FrameConfig periodicity
CHIRP_PERIOD_S = (100 + 60) * 1e-6 # ProfileConfig idle + ramp_end
SAMPLE_RATE_HZ = 10e6              # ProfileConfig digOutSampleRate (ksps)
FREQ_SLOPE_HZ_PER_S = 29.982e12    # ProfileConfig freqSlope (MHz/us → Hz/s)
START_FREQ_HZ = 77e9

WAVELENGTH_M = 3e8 / START_FREQ_HZ
T_SWEEP_S = N_SAMPLES / SAMPLE_RATE_HZ
BANDWIDTH_HZ = FREQ_SLOPE_HZ_PER_S * T_SWEEP_S
RANGE_RES_M = 3e8 / (2 * BANDWIDTH_HZ)
RANGE_MAX_M = (N_SAMPLES // 2) * RANGE_RES_M

BIN_PATH = ROOT / "data" / "adc_data.bin"
OUT_DIR = ROOT / "data"


def main() -> None:
    print(f"=== {BIN_PATH.name} ===")
    print(f"range_res {RANGE_RES_M * 100:.1f} cm   range_max {RANGE_MAX_M:.1f} m   "
          f"BW {BANDWIDTH_HZ / 1e6:.1f} MHz")
    print(f"frame rate {1 / FRAME_PERIOD_S:.1f} Hz   capture {N_FRAMES * FRAME_PERIOD_S:.2f} s")

    data = load_bin(BIN_PATH, N_CHIRPS, N_SAMPLES, N_RX, n_frames=N_FRAMES)
    assert data.shape == (N_FRAMES, N_CHIRPS, N_SAMPLES, N_RX), data.shape
    mag = np.abs(data)
    print(f"shape {data.shape}   |x| mean {mag.mean():.1f}   max {mag.max():.0f}   "
          f"NaN {int(np.isnan(mag).sum())}")

    # Range FFT along fast time, keep physical (positive) half.
    rfft = np.fft.fft(data, axis=2)[:, :, : N_SAMPLES // 2, :]
    range_axis = np.arange(N_SAMPLES // 2) * RANGE_RES_M

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Plot 1: mean range profile per RX ───────────────────────────────────
    profile = np.abs(rfft).mean(axis=(0, 1))  # (S/2, R)
    fig, ax = plt.subplots(figsize=(10, 4))
    for r in range(N_RX):
        ax.plot(range_axis, 20 * np.log10(profile[:, r] + 1e-9), label=f"RX{r}")
    ax.set(xlabel="Range (m)", ylabel="Magnitude (dB)",
           title="Range FFT — mean over all chirps × frames")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out1 = OUT_DIR / "quick_range_fft.png"
    fig.savefig(out1, dpi=120)
    plt.close(fig)
    print(f"wrote {out1.name}")

    # ── Plot 2: range-time heatmap (RX0) ────────────────────────────────────
    rfft_flat = rfft[:, :, :, 0].reshape(-1, N_SAMPLES // 2)  # (F*C, S/2)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        20 * np.log10(np.abs(rfft_flat).T + 1e-9),
        aspect="auto", origin="lower",
        extent=[0, rfft_flat.shape[0], 0, range_axis[-1]],
        cmap="viridis",
    )
    ax.set(xlabel="Chirp index (concatenated; 19.5 ms inter-frame gap not shown)",
           ylabel="Range (m)", title="Range-time magnitude (RX0, dB)")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    out2 = OUT_DIR / "quick_range_time.png"
    fig.savefig(out2, dpi=120)
    plt.close(fig)
    print(f"wrote {out2.name}")

    # ── Plot 3: phase at chest bin ──────────────────────────────────────────
    # Pick the strongest bin in a plausible chest-distance window.
    bin_min = max(1, int(0.3 / RANGE_RES_M))
    bin_max = max(bin_min + 1, int(1.5 / RANGE_RES_M))
    chest_window = np.abs(rfft[:, :, bin_min:bin_max, :]).mean(axis=(0, 1, 3))
    chest_bin = bin_min + int(np.argmax(chest_window))
    chest_range = chest_bin * RANGE_RES_M
    print(f"chest bin {chest_bin} ({chest_range:.2f} m); searched "
          f"{bin_min * RANGE_RES_M:.2f}–{bin_max * RANGE_RES_M:.2f} m")

    chest_complex = rfft[:, :, chest_bin, 0]              # (F, C)
    phase_per_chirp = np.unwrap(np.angle(chest_complex.ravel()))
    chirp_t_ms = np.arange(phase_per_chirp.size) * CHIRP_PERIOD_S * 1e3

    per_frame_complex = chest_complex.mean(axis=1)        # coherent integration
    phase_per_frame = np.unwrap(np.angle(per_frame_complex))
    disp_mm = 1e3 * WAVELENGTH_M / (4 * np.pi) * phase_per_frame
    frame_t_ms = np.arange(N_FRAMES) * FRAME_PERIOD_S * 1e3

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(chirp_t_ms, phase_per_chirp)
    axes[0].set(xlabel="Chirp time (ms, inter-frame gaps not shown)",
                ylabel="Unwrapped phase (rad)",
                title=f"Phase per chirp at bin {chest_bin} (~{chest_range:.2f} m), RX0")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frame_t_ms, disp_mm, marker="o")
    axes[1].set(xlabel="Frame time (ms)", ylabel="Displacement (mm)",
                title="Frame-rate displacement (chirp-averaged), λ/(4π)·Δφ")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out3 = OUT_DIR / "quick_phase.png"
    fig.savefig(out3, dpi=120)
    plt.close(fig)
    print(f"wrote {out3.name}")


if __name__ == "__main__":
    main()
