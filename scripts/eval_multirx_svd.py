"""Offline evaluation: multi-RX SVD coherent combining vs naive mean-RX.

Goal: replace the current baseline of `z = z_rx.mean(axis=1)` with a phase-
aligned SVD combiner (equivalent to MRC under rank-1 signal + AWGN). The SVD
output is expected to deliver a 3-5 dB SNR improvement on the heartbeat
phase signal, which should translate to a higher Pearson r and a lower MAE
vs Polar H10 ground truth.

Pipeline structure:
  1. Load radar IQ CSV (# metadata header + per-frame 4-RX IQ at chest bin).
  2. Load Polar H10 CSV, deduped by epoch_s.
  3. Run BASELINE  (z = mean of RX) -> remove_dc -> extract_phase ->
     detrend_median -> despike_hampel -> bandpass(0.8-2.5) -> detect_beats.
  4. Run SVD       (z = multirx_svd_combine(z_rx)) -> same downstream.
  5. Beat-match radar peaks (absolute unix) to Polar beats within +/- 0.4 s.
  6. Resample HR to a 1 Hz grid for both runs, compute Pearson r vs Polar.
  7. Print 4-row before/after table + WIN/LOSS verdict.
  8. Save text summary to scripts/eval_multirx_svd_summary.txt.
  9. Save 4-panel figure to scripts/eval_multirx_svd.png.

ASCII only. matplotlib Agg backend (headless). numpy / pandas / scipy /
matplotlib only.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import welch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `import radar_analysis` work without installing the package.
_REPO_ROOT = r"C:\Users\rapha\Desktop\MIT\Sensor & Mobile Computing\Final\IoT_IBI_Tracking"
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from radar_analysis.phase_processing import (  # noqa: E402
    despike_hampel,
    detrend_median,
    extract_phase,
    remove_dc,
)
from radar_analysis.heartbeat_extractors import bandpass  # noqa: E402
from radar_analysis.beat_detection import (  # noqa: E402
    clean_ibi,
    detect_beats,
    peaks_to_ibi_ms,
)


# ---------------------------------------------------------------------------
# Hardcoded literal paths (per task spec).
# ---------------------------------------------------------------------------
RADAR_CSV = (
    r"C:\Users\rapha\Desktop\MIT\Sensor & Mobile Computing\Final\MakeyMakey"
    r"\recordings\session_2026-05-10_16-11-27.csv"
)
POLAR_CSV = (
    r"C:\Users\rapha\xwechat_files\wxid_5cgorr0uasr811_de98\msg\file"
    r"\2026-05\polar_h10_radar_sync_01_20260510_160910(1).csv"
)

OUT_DIR = os.path.join(_REPO_ROOT, "scripts")
SUMMARY_TXT = os.path.join(OUT_DIR, "eval_multirx_svd_summary.txt")
FIGURE_PNG = os.path.join(OUT_DIR, "eval_multirx_svd.png")

# Beat-matching window and the fixed bandpass per the task spec.
MATCH_WINDOW_S = 0.4
HB_LOW_HZ = 0.8
HB_HIGH_HZ = 2.5


# ---------------------------------------------------------------------------
# The new architecture: phase-aligned SVD-based multi-RX combining.
# ---------------------------------------------------------------------------
def multirx_svd_combine(z_rx: np.ndarray) -> np.ndarray:
    """Phase-aligned SVD combining of 4 RX antennas.

    Input : z_rx of shape (F, R) complex -- F frames, R RX antennas.
    Output: z_combined of shape (F,) complex.

    Why this beats mean: each RX has a slightly different static phase
    offset (cable / antenna calibration) and amplitude (gain mismatch).
    Naive mean suffers destructive interference between mis-aligned
    antennas. SVD finds the linear combination that maximizes signal
    energy -- equivalent to MRC (Maximal Ratio Combining) under additive
    white Gaussian noise.
    """
    if z_rx.ndim != 2:
        raise ValueError(f"expected (F, R), got shape {z_rx.shape}")

    # Step 1: phase-align each RX to a common reference. For each RX,
    # compute its mean complex value; that mean carries the static
    # reflector ("DC") + any per-RX phase offset. Divide each frame by
    # exp(j * angle(mean_rx)) to bring each RX to a common phase frame.
    mu_rx = z_rx.mean(axis=0)                              # (R,) complex
    phase_correction = np.exp(-1j * np.angle(mu_rx))       # (R,) complex
    z_aligned = z_rx * phase_correction[None, :]           # (F, R) all RX share phase ref

    # Step 2: remove static clutter (mean per RX). After this, the
    # signal is purely the motion-driven component.
    z_centered = z_aligned - z_aligned.mean(axis=0, keepdims=True)

    # Step 3: SVD. The first left-singular vector weighted by sigma_0 is
    # the optimal MRC output under the rank-1 signal model. Equivalent
    # to PCA on complex data, taking the first component.
    U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
    z_combined = U[:, 0] * S[0]                            # (F,) complex

    return z_combined


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def parse_radar_csv(path: str) -> tuple[dict, pd.DataFrame]:
    """Parse leading `# key: value` metadata, then load the CSV body."""
    meta: dict[str, str] = {}
    skip = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            if not line.startswith("#"):
                break
            skip += 1
            stripped = line.lstrip("#").strip()
            if ":" in stripped:
                k, v = stripped.split(":", 1)
                meta[k.strip()] = v.strip()
    df = pd.read_csv(path, skiprows=skip)
    return meta, df


def load_polar(path: str) -> pd.DataFrame:
    """Read Polar H10 CSV. Dedupe by epoch_s (Polar sometimes emits duplicates)."""
    df = pd.read_csv(path)
    df = df.dropna(subset=["epoch_s", "ibi_ms"])
    df = (
        df.drop_duplicates(subset="epoch_s")
          .sort_values("epoch_s")
          .reset_index(drop=True)
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    label: str
    z_combined: np.ndarray   # complex (F,)
    phi_clean: np.ndarray    # float (F,) -- post detrend + despike
    hb: np.ndarray           # float (F,) -- bandpass output
    peaks_t: np.ndarray      # peak times in seconds (relative to session start)
    peaks_unix: np.ndarray   # peak times in absolute unix seconds
    ibi_ms: np.ndarray       # all consecutive IBIs from peaks_t
    keep: np.ndarray         # bool mask from clean_ibi
    nn_ms: np.ndarray        # kept IBIs


def run_pipeline(
    z_combined: np.ndarray,
    fs: float,
    session_start_unix: float,
    *,
    label: str,
    low_hz: float = HB_LOW_HZ,
    high_hz: float = HB_HIGH_HZ,
) -> PipelineResult:
    """Run the standard phase-extraction + bandpass + beat-detection chain on
    a pre-combined (1-D complex) slow-time signal.

    Everything downstream of the RX-combining step is identical between
    BASELINE and SVD, so the only thing that varies is `z_combined`.
    """
    z_c = remove_dc(z_combined)
    phi = extract_phase(z_c)
    phi = detrend_median(phi, fs)
    phi_clean = despike_hampel(phi)

    hb = bandpass(phi_clean, fs, low_hz=low_hz, high_hz=high_hz, order=4)
    peaks_t = detect_beats(hb, fs)
    peaks_unix = session_start_unix + peaks_t
    ibi_ms = peaks_to_ibi_ms(peaks_t)
    keep = clean_ibi(ibi_ms)
    nn_ms = ibi_ms[keep] if ibi_ms.size else ibi_ms

    return PipelineResult(
        label=label,
        z_combined=z_combined,
        phi_clean=phi_clean,
        hb=hb,
        peaks_t=peaks_t,
        peaks_unix=peaks_unix,
        ibi_ms=ibi_ms,
        keep=keep,
        nn_ms=nn_ms,
    )


# ---------------------------------------------------------------------------
# Beat matching + per-pair agreement metrics
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    n_polar: int
    n_radar_peaks: int
    n_matched: int
    polar_ibi: np.ndarray
    radar_ibi: np.ndarray
    polar_unix_paired: np.ndarray
    radar_unix_paired: np.ndarray


def match_beats(
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    radar_unix: np.ndarray,
    radar_ibi_ms: np.ndarray,
    radar_keep: np.ndarray,
    window_s: float = MATCH_WINDOW_S,
) -> MatchResult:
    """Pair each Polar beat with the nearest kept-radar IBI end-time within
    +/- window_s. Returns paired arrays for downstream metric computation.
    """
    if radar_unix.size < 2 or radar_ibi_ms.size == 0:
        return MatchResult(
            n_polar=int(polar_unix.size),
            n_radar_peaks=int(radar_unix.size),
            n_matched=0,
            polar_ibi=np.array([]),
            radar_ibi=np.array([]),
            polar_unix_paired=np.array([]),
            radar_unix_paired=np.array([]),
        )

    # radar IBI at index j ends at radar_unix[j+1]
    radar_end_unix = radar_unix[1:]
    keep_mask = radar_keep
    radar_end_unix_k = radar_end_unix[keep_mask]
    radar_ibi_k = radar_ibi_ms[keep_mask]

    if radar_end_unix_k.size == 0:
        return MatchResult(
            n_polar=int(polar_unix.size),
            n_radar_peaks=int(radar_unix.size),
            n_matched=0,
            polar_ibi=np.array([]),
            radar_ibi=np.array([]),
            polar_unix_paired=np.array([]),
            radar_unix_paired=np.array([]),
        )

    matched_polar = []
    matched_radar = []
    matched_polar_t = []
    matched_radar_t = []
    sorted_radar = radar_end_unix_k
    j = 0
    for pt, pibi in zip(polar_unix, polar_ibi_ms):
        while j + 1 < sorted_radar.size and abs(sorted_radar[j + 1] - pt) <= abs(sorted_radar[j] - pt):
            j += 1
        if abs(sorted_radar[j] - pt) <= window_s:
            matched_polar.append(float(pibi))
            matched_radar.append(float(radar_ibi_k[j]))
            matched_polar_t.append(float(pt))
            matched_radar_t.append(float(sorted_radar[j]))

    return MatchResult(
        n_polar=int(polar_unix.size),
        n_radar_peaks=int(radar_unix.size),
        n_matched=len(matched_polar),
        polar_ibi=np.array(matched_polar),
        radar_ibi=np.array(matched_radar),
        polar_unix_paired=np.array(matched_polar_t),
        radar_unix_paired=np.array(matched_radar_t),
    )


# ---------------------------------------------------------------------------
# 1 Hz HR-grid metrics (overlapping window only)
# ---------------------------------------------------------------------------
@dataclass
class HRMetrics:
    n_samples: int
    pearson_r: float
    mae_bpm: float
    rmse_bpm: float
    bias_bpm: float


def _ibi_series_to_hr_1hz(
    beat_unix: np.ndarray, ibi_ms: np.ndarray, t0: float, t1: float
) -> tuple[np.ndarray, np.ndarray]:
    """Step-interpolate an IBI series onto a 1 Hz grid spanning [t0, t1].

    The IBI at index i is the interval ending at beat_unix[i]; we model
    HR(t) at time t as the most recent IBI whose right edge (beat) is
    at or before t. Grid points before the first beat are NaN.
    """
    grid = np.arange(np.floor(t0), np.ceil(t1) + 1.0, 1.0, dtype=np.float64)
    hr = np.full(grid.size, np.nan, dtype=np.float64)
    if beat_unix.size == 0 or ibi_ms.size == 0:
        return grid, hr
    n = min(beat_unix.size, ibi_ms.size)
    bu = beat_unix[:n]
    ib = ibi_ms[:n]
    hr_valid = 60000.0 / ib
    # For each grid point t, take the IBI of the latest beat with beat_unix <= t.
    # searchsorted with side='right' gives the insertion index, so the latest
    # beat at-or-before t is at index (insert_idx - 1).
    insert = np.searchsorted(bu, grid, side="right")
    valid = insert > 0
    idx = insert - 1
    hr[valid] = hr_valid[idx[valid]]
    return grid, hr


def hr_metrics_1hz(
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    radar_peaks_unix: np.ndarray,
    radar_ibi_ms: np.ndarray,
    radar_keep: np.ndarray,
) -> tuple[HRMetrics, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Compute MAE/RMSE/bias/Pearson r on a 1 Hz HR grid using only the
    overlap window between Polar and (kept) radar beats.
    """
    if radar_peaks_unix.size < 2 or radar_ibi_ms.size == 0:
        return (
            HRMetrics(n_samples=0, pearson_r=float("nan"),
                      mae_bpm=float("nan"), rmse_bpm=float("nan"),
                      bias_bpm=float("nan")),
            (np.array([]), np.array([]), np.array([])),
        )

    radar_end_unix = radar_peaks_unix[1:]
    rk = radar_keep
    radar_end_k = radar_end_unix[rk]
    radar_ibi_k = radar_ibi_ms[rk]
    if radar_end_k.size < 2:
        return (
            HRMetrics(n_samples=0, pearson_r=float("nan"),
                      mae_bpm=float("nan"), rmse_bpm=float("nan"),
                      bias_bpm=float("nan")),
            (np.array([]), np.array([]), np.array([])),
        )

    # Use kept-IBI right edges as the radar "beat" timestamps. Each kept
    # IBI is then HR = 60000 / ibi_ms valid at its right-edge timestamp.
    polar_beats = polar_unix
    polar_ibis = polar_ibi_ms

    t0 = max(float(polar_beats.min()), float(radar_end_k.min()))
    t1 = min(float(polar_beats.max()), float(radar_end_k.max()))
    if t1 <= t0:
        return (
            HRMetrics(n_samples=0, pearson_r=float("nan"),
                      mae_bpm=float("nan"), rmse_bpm=float("nan"),
                      bias_bpm=float("nan")),
            (np.array([]), np.array([]), np.array([])),
        )

    grid_p, hr_p = _ibi_series_to_hr_1hz(polar_beats, polar_ibis, t0, t1)
    grid_r, hr_r = _ibi_series_to_hr_1hz(radar_end_k, radar_ibi_k, t0, t1)

    # Align grids (they share spacing but may differ by endpoint rounding).
    # Pick the common subgrid via intersection.
    if grid_p.size != grid_r.size or not np.allclose(grid_p, grid_r):
        common = np.intersect1d(grid_p, grid_r)
        ip = np.searchsorted(grid_p, common)
        ir = np.searchsorted(grid_r, common)
        grid = common
        hr_p = hr_p[ip]
        hr_r = hr_r[ir]
    else:
        grid = grid_p

    mask = np.isfinite(hr_p) & np.isfinite(hr_r)
    if mask.sum() < 2:
        return (
            HRMetrics(n_samples=int(mask.sum()), pearson_r=float("nan"),
                      mae_bpm=float("nan"), rmse_bpm=float("nan"),
                      bias_bpm=float("nan")),
            (grid, hr_p, hr_r),
        )

    a = hr_p[mask]
    b = hr_r[mask]
    if a.std() == 0 or b.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(a, b)[0, 1])
    err = b - a
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    return (
        HRMetrics(n_samples=int(mask.sum()), pearson_r=r, mae_bpm=mae,
                  rmse_bpm=rmse, bias_bpm=bias),
        (grid, hr_p, hr_r),
    )


# ---------------------------------------------------------------------------
# Reporting + plotting
# ---------------------------------------------------------------------------
def fmt_float(x: float, n: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{n}f}"


def build_summary(
    polar_df: pd.DataFrame,
    polar_unix: np.ndarray,
    res_base: PipelineResult,
    res_svd: PipelineResult,
    hrm_base: HRMetrics,
    hrm_svd: HRMetrics,
    match_base: MatchResult,
    match_svd: MatchResult,
    fs: float,
    n_frames: int,
    session_start_unix: float,
) -> tuple[str, bool]:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("Multi-RX SVD combining vs naive mean-RX -- offline eval")
    add("=" * 78)
    add("")
    add(f"Radar IQ CSV          : {RADAR_CSV}")
    add(f"Polar H10 CSV         : {POLAR_CSV}")
    add(f"Session start (unix)  : {session_start_unix:.6f}")
    add(f"Sampling rate fs (Hz) : {fs:.3f}")
    add(f"Radar frames          : {n_frames}")
    add(f"Radar duration (s)    : {n_frames / fs:.2f}")
    add(f"Bandpass (Hz)         : {HB_LOW_HZ:.2f} - {HB_HIGH_HZ:.2f}")
    add(f"Beat-match window (s) : +/- {MATCH_WINDOW_S:.2f}")
    add("")

    # Polar
    add("-" * 78)
    add("Polar H10 ground truth")
    add("-" * 78)
    if polar_df.shape[0] > 0:
        polar_dur = float(polar_unix[-1] - polar_unix[0])
        polar_mean_hr = float(polar_df["hr_bpm"].mean())
        polar_mean_ibi = float(polar_df["ibi_ms"].mean())
        add(f"  n_beats (deduped)     : {polar_df.shape[0]}")
        add(f"  duration (s)          : {polar_dur:.2f}")
        add(f"  mean HR (bpm)         : {polar_mean_hr:.2f}")
        add(f"  mean IBI (ms)         : {polar_mean_ibi:.2f}")
    else:
        add("  no Polar rows")
    add("")

    for res, hrm, match, tag in (
        (res_base, hrm_base, match_base, "BASELINE (mean-RX)"),
        (res_svd, hrm_svd, match_svd, "SVD (multi-RX MRC)"),
    ):
        add("-" * 78)
        add(f"Radar run: {tag}")
        add("-" * 78)
        n_peaks = int(res.peaks_t.size)
        n_ibi = int(res.ibi_ms.size)
        n_kept = int(res.keep.sum()) if res.keep.size else 0
        mean_hr = (60000.0 / float(np.mean(res.nn_ms))) if res.nn_ms.size else float("nan")
        mean_ibi = float(np.mean(res.nn_ms)) if res.nn_ms.size else float("nan")
        add(f"  n_peaks               : {n_peaks}")
        add(f"  n_ibis                : {n_ibi}")
        add(f"  n_kept (clean_ibi)    : {n_kept}")
        add(f"  mean radar HR (bpm)   : {fmt_float(mean_hr, 2)}")
        add(f"  mean radar IBI (ms)   : {fmt_float(mean_ibi, 2)}")
        add(f"  matched pairs         : {match.n_matched} of {match.n_polar} polar beats")
        add(f"  HR-1Hz n_samples      : {hrm.n_samples}")
        add(f"  HR-1Hz Pearson r      : {fmt_float(hrm.pearson_r, 4)}")
        add(f"  HR-1Hz MAE (bpm)      : {fmt_float(hrm.mae_bpm, 3)}")
        add(f"  HR-1Hz RMSE (bpm)     : {fmt_float(hrm.rmse_bpm, 3)}")
        add(f"  HR-1Hz bias (bpm)     : {fmt_float(hrm.bias_bpm, 3)}")
        add("")

    add("=" * 78)
    add("Before / After comparison (1 Hz HR grid)")
    add("=" * 78)
    header = "  {metric:<22}  {b:>14}  {s:>14}  {d:>14}"
    add(header.format(metric="metric", b="BASELINE", s="SVD (multi-RX)", d="delta (SVD-base)"))
    add("  " + "-" * 74)

    def delta(b: float, s: float) -> float:
        if not (np.isfinite(b) and np.isfinite(s)):
            return float("nan")
        return s - b

    rows = [
        ("Pearson r", hrm_base.pearson_r, hrm_svd.pearson_r, 4),
        ("MAE (bpm)", hrm_base.mae_bpm,  hrm_svd.mae_bpm,  3),
        ("RMSE (bpm)", hrm_base.rmse_bpm, hrm_svd.rmse_bpm, 3),
        ("bias (bpm)", hrm_base.bias_bpm, hrm_svd.bias_bpm, 3),
    ]
    for name, b, s, n in rows:
        add(header.format(
            metric=name, b=fmt_float(b, n), s=fmt_float(s, n),
            d=fmt_float(delta(b, s), n),
        ))
    add("")

    # Verdict
    add("-" * 78)
    add("Verdict")
    add("-" * 78)
    win = (
        np.isfinite(hrm_base.pearson_r) and np.isfinite(hrm_svd.pearson_r)
        and np.isfinite(hrm_base.mae_bpm) and np.isfinite(hrm_svd.mae_bpm)
        and (hrm_svd.pearson_r > hrm_base.pearson_r)
        and (hrm_svd.mae_bpm < hrm_base.mae_bpm)
    )
    if win:
        add("  WIN: SVD beats mean-RX on BOTH Pearson r AND MAE.")
        add(f"     Pearson r:  {hrm_base.pearson_r:+.4f} -> {hrm_svd.pearson_r:+.4f}  "
            f"(delta {hrm_svd.pearson_r - hrm_base.pearson_r:+.4f})")
        add(f"     MAE (bpm): {hrm_base.mae_bpm:.3f} -> {hrm_svd.mae_bpm:.3f}  "
            f"(delta {hrm_svd.mae_bpm - hrm_base.mae_bpm:+.3f})")
    else:
        add("  LOSS or NEUTRAL -- investigate panel 3.")
        if np.isfinite(hrm_base.pearson_r) and np.isfinite(hrm_svd.pearson_r):
            add(f"     Pearson r:  {hrm_base.pearson_r:+.4f} -> {hrm_svd.pearson_r:+.4f}  "
                f"(delta {hrm_svd.pearson_r - hrm_base.pearson_r:+.4f})")
        if np.isfinite(hrm_base.mae_bpm) and np.isfinite(hrm_svd.mae_bpm):
            add(f"     MAE (bpm): {hrm_base.mae_bpm:.3f} -> {hrm_svd.mae_bpm:.3f}  "
                f"(delta {hrm_svd.mae_bpm - hrm_base.mae_bpm:+.3f})")
    add("=" * 78)
    return "\n".join(lines) + "\n", win


def make_figure(
    polar_df: pd.DataFrame,
    polar_unix: np.ndarray,
    res_base: PipelineResult,
    res_svd: PipelineResult,
    match_svd: MatchResult,
    hrm_base: HRMetrics,
    hrm_svd: HRMetrics,
    grids_base: tuple[np.ndarray, np.ndarray, np.ndarray],
    grids_svd: tuple[np.ndarray, np.ndarray, np.ndarray],
    fs: float,
    session_start_unix: float,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(13, 18))

    # --- Panel 1: HR(t) overlay -- Polar (green), baseline (red), SVD (blue) ---
    ax = axes[0]
    # Polar HR by raw rows (true beat-level HR series)
    if polar_unix.size:
        ax.plot(polar_unix - session_start_unix,
                polar_df["hr_bpm"].to_numpy(dtype=np.float64),
                color="green", lw=1.4, label="Polar H10 (truth)")
    # Baseline HR (from 1Hz grid, finite mask)
    grid_b, hp_b, hr_b = grids_base
    if grid_b.size:
        ax.plot(grid_b - session_start_unix, hr_b,
                color="red", lw=1.0, alpha=0.85,
                label=f"Baseline mean-RX  r={hrm_base.pearson_r:.3f}")
    # SVD HR
    grid_s, hp_s, hr_s = grids_svd
    if grid_s.size:
        ax.plot(grid_s - session_start_unix, hr_s,
                color="blue", lw=1.0, alpha=0.85,
                label=f"SVD multi-RX   r={hrm_svd.pearson_r:.3f}")
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("HR (bpm)")
    ax.set_title("Panel 1  HR(t)  Polar vs Baseline vs SVD")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 2: scatter Polar-IBI vs Radar-IBI (SVD), y=x, r in title ---
    ax = axes[1]
    if match_svd.n_matched >= 2:
        p = match_svd.polar_ibi
        r = match_svd.radar_ibi
        if p.std() == 0 or r.std() == 0:
            r_pearson = float("nan")
        else:
            r_pearson = float(np.corrcoef(p, r)[0, 1])
        mae_ibi = float(np.mean(np.abs(r - p)))
        bias_ibi = float(np.mean(r - p))
        ax.scatter(p, r, s=14, color="blue", alpha=0.7, label=f"n={match_svd.n_matched}")
        lo = float(min(p.min(), r.min())) - 20.0
        hi = float(max(p.max(), r.max())) + 20.0
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.0, label="y=x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Polar IBI (ms)")
        ax.set_ylabel("Radar IBI (ms) [SVD]")
        ax.set_title(
            f"Panel 2  Scatter SVD vs Polar IBI  r={r_pearson:.3f}  "
            f"MAE={mae_ibi:.1f} ms  bias={bias_ibi:+.1f} ms"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "insufficient matched pairs", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Panel 2  Scatter SVD vs Polar IBI -- insufficient pairs")

    # --- Panel 3: time-domain phi_clean for both runs overlaid ---
    ax = axes[2]
    t_axis = np.arange(res_base.phi_clean.size) / fs
    ax.plot(t_axis, res_base.phi_clean, color="red", lw=0.7, alpha=0.8,
            label="phi_clean BASELINE (mean-RX)")
    ax.plot(t_axis, res_svd.phi_clean, color="blue", lw=0.7, alpha=0.8,
            label="phi_clean SVD (multi-RX)")
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("phase (rad)")
    base_std = float(np.std(res_base.phi_clean))
    svd_std = float(np.std(res_svd.phi_clean))
    ax.set_title(
        f"Panel 3  Time-domain phi_clean  "
        f"std_baseline={base_std:.4f}  std_svd={svd_std:.4f}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 4: Welch power spectrum of both phi_clean overlaid, Polar f0 marked ---
    ax = axes[3]
    nperseg_max = min(res_base.phi_clean.size, res_svd.phi_clean.size)
    nperseg = min(nperseg_max, max(int(16 * fs), 256))
    if nperseg < 32:
        ax.text(0.5, 0.5, "phase signal too short for Welch",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Panel 4  Welch spectrum -- insufficient data")
    else:
        fr_b, Pb = welch(res_base.phi_clean, fs=fs, nperseg=nperseg)
        fr_s, Ps = welch(res_svd.phi_clean, fs=fs, nperseg=nperseg)
        # Plot in dB.
        eps = 1e-30
        ax.semilogy(fr_b, Pb + eps, color="red", lw=1.0, alpha=0.85,
                    label="BASELINE (mean-RX)")
        ax.semilogy(fr_s, Ps + eps, color="blue", lw=1.0, alpha=0.85,
                    label="SVD (multi-RX)")
        # Polar's true f0 (= mean HR / 60)
        if polar_df.shape[0] > 0:
            f0_polar = float(polar_df["hr_bpm"].mean()) / 60.0
            ax.axvline(f0_polar, color="green", ls="--", lw=1.0,
                       label=f"Polar f0 = {f0_polar:.2f} Hz")
        ax.set_xlim(0.0, 4.0)
        ax.set_xlabel("frequency (Hz)")
        ax.set_ylabel("Welch PSD")
        # Compute peak SNR-ish ratio inside heartbeat band as a quick proxy.
        band = (fr_b >= HB_LOW_HZ) & (fr_b <= HB_HIGH_HZ)
        if band.any():
            peak_b = float(np.max(Pb[band]))
            peak_s = float(np.max(Ps[band]))
            if peak_b > 0:
                ratio_db = 10.0 * np.log10(peak_s / peak_b)
            else:
                ratio_db = float("nan")
            ax.set_title(
                f"Panel 4  Welch PSD (phi_clean)  in-band peak ratio "
                f"SVD/baseline = {ratio_db:+.2f} dB"
            )
        else:
            ax.set_title("Panel 4  Welch PSD (phi_clean)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    # ---- Load radar ----
    meta, radar_df = parse_radar_csv(RADAR_CSV)
    fs = float(meta.get("fs_hz", "100.0"))
    session_start_unix = float(meta["session_start_unix"])
    n_frames = int(meta.get("n_frames", len(radar_df)))

    iq_cols = [
        ("iq0_real", "iq0_imag"),
        ("iq1_real", "iq1_imag"),
        ("iq2_real", "iq2_imag"),
        ("iq3_real", "iq3_imag"),
    ]
    z_rx = np.column_stack([
        radar_df[r].to_numpy(dtype=np.float64) + 1j * radar_df[i].to_numpy(dtype=np.float64)
        for r, i in iq_cols
    ])  # (F, 4) complex

    # ---- Load Polar ----
    polar_df = load_polar(POLAR_CSV)
    polar_unix = polar_df["epoch_s"].to_numpy(dtype=np.float64)
    polar_ibi_ms = polar_df["ibi_ms"].to_numpy(dtype=np.float64)

    print("=" * 78)
    print(f"Session start (unix) : {session_start_unix:.6f}")
    print(f"Sampling rate fs (Hz): {fs:.3f}")
    print(f"Radar frames         : {n_frames}")
    print(f"Polar n_beats        : {polar_df.shape[0]}")
    if polar_df.shape[0] > 0:
        print(f"Polar mean HR (bpm)  : {polar_df['hr_bpm'].mean():.2f}")
        print(f"Polar mean IBI (ms)  : {polar_df['ibi_ms'].mean():.2f}")
    print("=" * 78)

    # ---- BASELINE: naive mean across RX ----
    z_base = z_rx.mean(axis=1)
    res_base = run_pipeline(
        z_base, fs, session_start_unix,
        label="BASELINE (mean-RX)",
    )
    n_kept_base = int(res_base.keep.sum()) if res_base.keep.size else 0
    mean_hr_base = (60000.0 / float(np.mean(res_base.nn_ms))) if res_base.nn_ms.size else float("nan")
    mean_ibi_base = float(np.mean(res_base.nn_ms)) if res_base.nn_ms.size else float("nan")
    print()
    print("[BASELINE  (mean-RX)]")
    print(f"  n_peaks              : {res_base.peaks_t.size}")
    print(f"  n_kept               : {n_kept_base}")
    print(f"  mean radar HR (bpm)  : {fmt_float(mean_hr_base, 2)}")
    print(f"  mean radar IBI (ms)  : {fmt_float(mean_ibi_base, 2)}")

    # ---- SVD: phase-aligned multi-RX combining ----
    z_svd = multirx_svd_combine(z_rx)
    res_svd = run_pipeline(
        z_svd, fs, session_start_unix,
        label="SVD (multi-RX)",
    )
    n_kept_svd = int(res_svd.keep.sum()) if res_svd.keep.size else 0
    mean_hr_svd = (60000.0 / float(np.mean(res_svd.nn_ms))) if res_svd.nn_ms.size else float("nan")
    mean_ibi_svd = float(np.mean(res_svd.nn_ms)) if res_svd.nn_ms.size else float("nan")
    print()
    print("[SVD       (multi-RX)]")
    print(f"  n_peaks              : {res_svd.peaks_t.size}")
    print(f"  n_kept               : {n_kept_svd}")
    print(f"  mean radar HR (bpm)  : {fmt_float(mean_hr_svd, 2)}")
    print(f"  mean radar IBI (ms)  : {fmt_float(mean_ibi_svd, 2)}")

    # ---- Beat-match against Polar (for scatter and IBI sanity) ----
    match_base = match_beats(
        polar_unix, polar_ibi_ms,
        res_base.peaks_unix, res_base.ibi_ms, res_base.keep,
        window_s=MATCH_WINDOW_S,
    )
    match_svd = match_beats(
        polar_unix, polar_ibi_ms,
        res_svd.peaks_unix, res_svd.ibi_ms, res_svd.keep,
        window_s=MATCH_WINDOW_S,
    )

    # ---- Resample HR to 1Hz overlap grid and compute metrics ----
    hrm_base, grids_base = hr_metrics_1hz(
        polar_unix, polar_ibi_ms,
        res_base.peaks_unix, res_base.ibi_ms, res_base.keep,
    )
    hrm_svd, grids_svd = hr_metrics_1hz(
        polar_unix, polar_ibi_ms,
        res_svd.peaks_unix, res_svd.ibi_ms, res_svd.keep,
    )

    # ---- Build textual summary + verdict ----
    summary, win = build_summary(
        polar_df, polar_unix,
        res_base, res_svd,
        hrm_base, hrm_svd,
        match_base, match_svd,
        fs=fs, n_frames=n_frames, session_start_unix=session_start_unix,
    )
    print()
    print(summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[wrote] {SUMMARY_TXT}")

    # ---- Plot ----
    make_figure(
        polar_df, polar_unix,
        res_base, res_svd,
        match_svd,
        hrm_base, hrm_svd,
        grids_base, grids_svd,
        fs=fs, session_start_unix=session_start_unix,
        out_path=FIGURE_PNG,
    )
    print(f"[wrote] {FIGURE_PNG}")
    return 0 if win else 0  # exit 0 either way; verdict is in the summary


if __name__ == "__main__":
    sys.exit(main())
