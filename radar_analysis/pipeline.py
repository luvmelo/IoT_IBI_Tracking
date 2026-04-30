"""End-to-end IBI/HRV pipeline: post-range-FFT cube → HRV metrics.

Stages (matches the research-note recipe; VMD/Stage 4 deferred):

    rfft (F, C, S, R)
        → select_chest_bin                            (chest_bin, score)
        → coherent integration (RX + chirps)          z(t)  complex (F,)
        → remove_dc                                   z_centered
        → extract_phase (atan2 + unwrap)              φ_raw   (F,)
        → detrend_median + despike_hampel             φ_clean
        → motion_mask                                 motion_ok bool (F,)
        → extract_heartbeat (Butter 0.8–4 Hz)         h(t)
        → detect_beats + parabolic refine             peak_times_s
        → peaks_to_ibi_ms + clean_ibi                 nn (ms)
        → mean_ibi/HR, SDNN, RMSSD, pNN50

`run_pipeline(...)` returns a dict with metrics, the chosen bin, the cleaned
phase, the heartbeat waveform, and the NN intervals. Pass `out_dir` to also
save diagnostic plots and a metrics.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

from radar_analysis.beat_detection import (
    clean_ibi,
    detect_beats,
    peaks_to_ibi_ms,
)
from radar_analysis.chest_bin_selection import select_chest_bin
from radar_analysis.heartbeat_extractors import extract_heartbeat
from radar_analysis.hrv_metrics import (
    mean_hr_bpm,
    mean_ibi_ms,
    pnn50,
    rmssd_ms,
    sdnn_ms,
)
from radar_analysis.phase_processing import (
    despike_hampel,
    detrend_median,
    extract_phase,
    motion_mask,
    remove_dc,
)


@dataclass
class PipelineResult:
    chest_bin: int
    chest_range_m: float
    chest_bin_score: float
    fs_slow_hz: float
    duration_s: float
    n_frames: int
    phi_clean: np.ndarray = field(repr=False)
    motion_mask: np.ndarray = field(repr=False)
    heartbeat: np.ndarray = field(repr=False)
    peak_times_s: np.ndarray = field(repr=False)
    ibi_ms: np.ndarray = field(repr=False)
    nn_ms: np.ndarray = field(repr=False)
    metrics: dict[str, float]

    def to_summary(self) -> dict[str, Any]:
        """JSON-safe summary (drops the big arrays)."""
        d = asdict(self)
        for k in ("phi_clean", "motion_mask", "heartbeat", "peak_times_s", "ibi_ms", "nn_ms"):
            d.pop(k, None)
        d["n_beats_detected"] = int(self.peak_times_s.size)
        d["n_intervals_kept"] = int(self.nn_ms.size)
        return d


def run_pipeline(
    rfft: np.ndarray,
    *,
    range_res_m: float,
    fs_slow_hz: float,
    chest_window_m: tuple[float, float] = (0.3, 1.5),
    hr_band_hz: tuple[float, float] = (0.8, 4.0),
    detrend_window_s: float = 2.0,
    out_dir: Path | str | None = None,
    save_plots: bool = True,
) -> PipelineResult:
    """Run the post-range-FFT pipeline end-to-end."""
    if rfft.ndim != 4:
        raise ValueError(f"rfft must be 4-D (F, C, S, R); got shape {rfft.shape}")
    if fs_slow_hz <= 0:
        raise ValueError(f"fs_slow_hz must be positive, got {fs_slow_hz}")
    if range_res_m <= 0:
        raise ValueError(f"range_res_m must be positive, got {range_res_m}")

    n_frames = rfft.shape[0]
    duration_s = n_frames / fs_slow_hz

    chest_bin, score = select_chest_bin(
        rfft, range_res_m=range_res_m, search_window_m=chest_window_m
    )
    chest_range_m = chest_bin * range_res_m

    # Coherent integration: average chirps within frame, then RX → 1 complex per frame
    z_at_bin = rfft[:, :, chest_bin, :].mean(axis=(1, 2))      # (F,) complex
    z_centered = remove_dc(z_at_bin)
    phi_raw = extract_phase(z_centered)
    phi_detrended = detrend_median(phi_raw, fs=fs_slow_hz, window_s=detrend_window_s)
    phi_clean = despike_hampel(phi_detrended)
    motion_ok = motion_mask(phi_clean, fs=fs_slow_hz)

    heartbeat = extract_heartbeat(phi_clean, fs=fs_slow_hz, low_hz=hr_band_hz[0], high_hz=hr_band_hz[1])

    peak_times_s = detect_beats(heartbeat, fs=fs_slow_hz)
    ibi_ms = peaks_to_ibi_ms(peak_times_s)
    keep_mask = clean_ibi(ibi_ms)
    nn_ms = ibi_ms[keep_mask]

    metrics = _compute_hrv_metrics(nn_ms)

    result = PipelineResult(
        chest_bin=chest_bin,
        chest_range_m=float(chest_range_m),
        chest_bin_score=float(score),
        fs_slow_hz=float(fs_slow_hz),
        duration_s=float(duration_s),
        n_frames=int(n_frames),
        phi_clean=phi_clean,
        motion_mask=motion_ok,
        heartbeat=heartbeat,
        peak_times_s=peak_times_s,
        ibi_ms=ibi_ms,
        nn_ms=nn_ms,
        metrics=metrics,
    )

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Plots first so a plot crash doesn't leave a metrics.json claiming success.
        if save_plots:
            _save_plots(rfft, range_res_m, result, out)
        with (out / "metrics.json").open("w") as f:
            json.dump(result.to_summary(), f, indent=2)

    return result


def _compute_hrv_metrics(nn_ms: np.ndarray) -> dict[str, float]:
    """HRV metrics with NaN if insufficient data; never raise from the pipeline."""
    if nn_ms.size < 2:
        return {
            "mean_ibi_ms": float("nan"),
            "mean_hr_bpm": float("nan"),
            "sdnn_ms": float("nan"),
            "rmssd_ms": float("nan"),
            "pnn50_pct": float("nan"),
        }
    return {
        "mean_ibi_ms": mean_ibi_ms(nn_ms),
        "mean_hr_bpm": mean_hr_bpm(nn_ms),
        "sdnn_ms": sdnn_ms(nn_ms),
        "rmssd_ms": rmssd_ms(nn_ms),
        "pnn50_pct": pnn50(nn_ms),
    }


def _save_plots(rfft: np.ndarray, range_res_m: float, r: PipelineResult, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_range_bins = rfft.shape[2]
    range_axis = np.arange(n_range_bins) * range_res_m
    rfft_mag = np.abs(rfft)

    # 1) Mean range profile per RX with chest bin marked
    profile = rfft_mag.mean(axis=(0, 1))                   # (S, R)
    fig, ax = plt.subplots(figsize=(10, 4))
    for rx in range(profile.shape[1]):
        ax.plot(range_axis, 20 * np.log10(profile[:, rx] + 1e-9), label=f"RX{rx}")
    ax.axvline(r.chest_range_m, color="red", linestyle="--", label=f"chest bin {r.chest_bin}")
    ax.set(xlabel="Range (m)", ylabel="dB", title="Mean range FFT — chest bin highlighted")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "range_fft.png", dpi=120)
    plt.close(fig)

    # 2) Range-time heatmap (RX0)
    flat = rfft_mag[:, :, :, 0].reshape(-1, n_range_bins)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(
        20 * np.log10(flat.T + 1e-9),
        aspect="auto", origin="lower", cmap="viridis",
        extent=[0, flat.shape[0], 0, n_range_bins * range_res_m],
    )
    ax.axhline(r.chest_range_m, color="red", linestyle="--", linewidth=1)
    ax.set(xlabel="Chirp index (concatenated)", ylabel="Range (m)",
           title=f"Range-time RX0 (dB) — chest at {r.chest_range_m:.2f} m")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(out / "range_time.png", dpi=120)
    plt.close(fig)

    # 3) Phase trace (cleaned) + motion mask shading
    t = np.arange(r.phi_clean.size) / r.fs_slow_hz
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, r.phi_clean, label="phi_clean (rad)")
    bad = ~r.motion_mask
    if bad.any():
        ax.fill_between(t, r.phi_clean.min(), r.phi_clean.max(),
                        where=bad, alpha=0.15, color="red", label="motion-flagged")
    ax.set(xlabel="Time (s)", ylabel="Phase (rad)",
           title=f"Detrended + despiked phase at bin {r.chest_bin}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "phase.png", dpi=120)
    plt.close(fig)

    # 4) Heartbeat-band signal + detected peaks
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, r.heartbeat, label="HR-band (rad)")
    if r.peak_times_s.size:
        peak_y = np.interp(r.peak_times_s, t, r.heartbeat)
        ax.scatter(r.peak_times_s, peak_y, color="red", marker="x",
                   label=f"detected ({r.peak_times_s.size})")
    ax.set(xlabel="Time (s)", ylabel="Filtered phase (rad)",
           title="Heartbeat-band signal + detected beats")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "heartbeat.png", dpi=120)
    plt.close(fig)

    # 5) IBI series
    fig, ax = plt.subplots(figsize=(10, 4))
    has_artists = False
    if r.ibi_ms.size:
        keep = clean_ibi(r.ibi_ms)
        beat_idx = np.arange(r.ibi_ms.size)
        ax.scatter(beat_idx[keep], r.ibi_ms[keep], color="green", s=14, label="kept")
        if (~keep).any():
            ax.scatter(beat_idx[~keep], r.ibi_ms[~keep], color="red", s=14,
                       marker="x", label="rejected")
        has_artists = True
    title = (
        f"IBI: kept {r.nn_ms.size}/{r.ibi_ms.size}   "
        f"HR={r.metrics['mean_hr_bpm']:.1f} BPM   "
        f"SDNN={r.metrics['sdnn_ms']:.1f}   RMSSD={r.metrics['rmssd_ms']:.1f}   "
        f"pNN50={r.metrics['pnn50_pct']:.1f}%"
    )
    ax.set(xlabel="Beat index", ylabel="IBI (ms)", title=title)
    ax.grid(True, alpha=0.3)
    if has_artists:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "ibi.png", dpi=120)
    plt.close(fig)
