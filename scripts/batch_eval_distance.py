"""
Batch evaluator for distance/duration ablation.

Walks the 5_11data/ directory tree:
    5_11data/{distance}{duration}/{trial}/
        session_*.csv         (radar IQ, ~4-RX complex at chest bin)
        polar_h10_*.csv       (ECG-grade ground truth)
        session_*_peaks.csv   (ignored)
        session_*_sync.csv    (ignored)

For each (distance, duration, trial) it runs BOTH:
    BASELINE  Butter4 0.8-2.5 Hz bandpass + find_peaks
    VMD+ACF   VMD K=5 + cardiac mode + find_peaks (also reports ACF BPM)

Beat-matches against Polar within +/- 0.4s; reports paired-IBI Pearson r,
MAE, bias, 1-Hz HR MAE, beat-match rate.

Outputs:
    scripts/batch_eval_results.csv          12 rows, one per trial, all metrics
    scripts/batch_eval_summary.png           4-panel summary across conditions
    scripts/batch_eval_summary.txt           Pretty-printed text summary
"""
from __future__ import annotations

import os
import re
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Set up imports from the IoT_IBI_Tracking project
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from radar_analysis.phase_processing import (
    remove_dc, extract_phase, detrend_median, despike_hampel
)
from radar_analysis.heartbeat_extractors import bandpass
from radar_analysis.beat_detection import (
    detect_beats, peaks_to_ibi_ms, clean_ibi
)

try:
    from vmdpy import VMD as _vmd_lib
    HAVE_VMDPY = True
except ImportError:
    HAVE_VMDPY = False
    raise SystemExit("Need vmdpy installed: pip install vmdpy")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    r"C:\Users\rapha\Desktop\MIT\Sensor & Mobile Computing\Final"
    r"\IoT_IBI_Tracking\5_11data-20260511T214505Z-3-001\5_11data"
)
OUT_DIR = _REPO_ROOT / "scripts"
RESULTS_CSV = OUT_DIR / "batch_eval_results.csv"
SUMMARY_PNG = OUT_DIR / "batch_eval_summary.png"
SUMMARY_TXT = OUT_DIR / "batch_eval_summary.txt"

MATCH_WINDOW_S = 0.4   # +/- this for beat matching against Polar


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def parse_radar_csv(path: str | Path):
    """Read the #-prefixed metadata header + per-frame IQ rows."""
    meta = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if ":" in line:
                k, v = line[1:].split(":", 1)
                meta[k.strip()] = v.strip()
    df = pd.read_csv(path, comment="#")
    return meta, df


def load_polar(path: str | Path) -> pd.DataFrame:
    """Load Polar H10 export and reconstruct true beat times from cumulative IBI.

    Polar's `epoch_s` is a sampling timestamp (~1 Hz), NOT the actual heartbeat
    time -- a second/third beat in the same sampling second reuses the same
    epoch_s. Naively de-duping by epoch_s throws away real beats. Instead,
    keep every row and reconstruct beat times by anchoring the first beat at
    epoch_s[0] and adding cumulative IBI for each subsequent beat. The result
    matches the actual ECG R-peak sequence to within Polar's sampling jitter
    (~50 ms), regardless of how many beats per sampling-second occurred.
    """
    polar = pd.read_csv(path).reset_index(drop=True)
    if "ibi_ms" not in polar.columns or "epoch_s" not in polar.columns:
        return polar
    polar = polar.dropna(subset=["ibi_ms"]).reset_index(drop=True)
    # Reconstructed beat times: t[0] = epoch_s[0]; t[i] = t[i-1] + ibi[i]/1000
    ibi_s = polar["ibi_ms"].to_numpy(dtype=np.float64) / 1000.0
    n = len(polar)
    beat_t = np.empty(n, dtype=np.float64)
    if n > 0:
        beat_t[0] = polar["epoch_s"].iloc[0]
        for i in range(1, n):
            beat_t[i] = beat_t[i - 1] + ibi_s[i]
    polar["beat_unix"] = beat_t   # canonical beat time (used by scorer)
    return polar


def find_trial_files(trial_dir: Path):
    """Return (radar_csv_path, polar_csv_path) — picks the main IQ CSV (not _peaks/_sync)."""
    sessions = sorted([
        p for p in trial_dir.glob("session_*.csv")
        if not p.name.endswith("_peaks.csv") and not p.name.endswith("_sync.csv")
    ])
    polars = sorted(trial_dir.glob("polar_h10_*.csv"))
    if not sessions or not polars:
        return None, None
    return sessions[0], polars[0]


# ---------------------------------------------------------------------------
# DSP pipelines (mirror eval_vmd_acf.py, kept self-contained)
# ---------------------------------------------------------------------------
def clean_phase_from_rx(z_rx: np.ndarray, fs: float) -> np.ndarray:
    """Single-bin: average 4 RX, remove DC, unwrap, detrend, despike."""
    z = z_rx.mean(axis=1)
    z = remove_dc(z)
    phi = extract_phase(z)
    phi = detrend_median(phi, fs)
    phi = despike_hampel(phi)
    return phi


def pipeline_baseline(phi_clean: np.ndarray, fs: float):
    """Butter4 bandpass 0.8-2.5 Hz + find_peaks."""
    hb = bandpass(phi_clean, fs, low_hz=0.8, high_hz=2.5, order=4)
    peaks_t = detect_beats(hb, fs)
    ibi = peaks_to_ibi_ms(peaks_t)
    keep = clean_ibi(ibi)
    return {
        "hb": hb,
        "peaks_t": peaks_t,
        "ibi": ibi,
        "keep": keep,
        "label": "baseline (Butter4 0.8-2.5)",
    }


def pipeline_vmd_acf(phi_clean: np.ndarray, fs: float):
    """VMD K=5, pick cardiac mode in 0.9-2.0 Hz, find_peaks + ACF cross-check."""
    u, _u_hat, omega = _vmd_lib(phi_clean, 2000.0, 0.0, 5, 0, 1, 1e-7)
    final_omega_hz = omega[-1, :] * fs
    in_band = (final_omega_hz >= 0.9) & (final_omega_hz <= 2.0)
    if in_band.any():
        powers = np.var(u, axis=1)
        in_band_idx = np.where(in_band)[0]
        cardiac_idx = int(in_band_idx[np.argmax(powers[in_band_idx])])
    else:
        cardiac_idx = int(np.argmin(np.abs(final_omega_hz - 1.4)))
    cardiac = u[cardiac_idx]
    cardiac_f0_hz = float(final_omega_hz[cardiac_idx])

    peaks_t = detect_beats(cardiac, fs, max_bpm=200, prominence_factor=0.5)
    ibi = peaks_to_ibi_ms(peaks_t)
    keep = clean_ibi(ibi)

    # ACF cross-check BPM
    c = cardiac - cardiac.mean()
    acf = correlate(c, c, mode="full")[len(c)-1:]
    if acf[0] != 0:
        acf = acf / acf[0]
    lag_min = int(60.0 / 180 * fs)
    lag_max = int(60.0 / 50  * fs)
    acf_lag = lag_min + int(np.argmax(acf[lag_min:lag_max+1]))
    acf_bpm = 60.0 / (acf_lag / fs)

    return {
        "cardiac": cardiac,
        "vmd_omega_hz": list(final_omega_hz),
        "cardiac_idx": cardiac_idx,
        "cardiac_f0_hz": cardiac_f0_hz,
        "cardiac_bpm": cardiac_f0_hz * 60.0,
        "acf_bpm": float(acf_bpm),
        "peaks_t": peaks_t,
        "ibi": ibi,
        "keep": keep,
        "label": "VMD+ACF (K=5)",
    }


# ---------------------------------------------------------------------------
# Beat matching + scoring vs Polar
# ---------------------------------------------------------------------------
def score_against_polar(
    peaks_t_radar: np.ndarray,
    ibi_radar_ms: np.ndarray,
    keep_radar: np.ndarray,
    session_start_unix: float,
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    fs: float,
    window_s: float = 0.4,
) -> dict:
    """Time-align radar peaks to Polar via Unix epoch, compute metrics."""
    radar_unix = session_start_unix + peaks_t_radar
    rt = radar_unix
    pt = polar_unix

    # Trim to overlap window
    if len(rt) == 0 or len(pt) == 0:
        return _empty_scores()
    t0 = max(rt.min(), pt.min())
    t1 = min(rt.max(), pt.max())
    if t1 <= t0:
        return _empty_scores()

    pt_w_mask = (pt >= t0) & (pt <= t1)
    pt_w = pt[pt_w_mask]
    polar_ibi_w = polar_ibi_ms[pt_w_mask]

    # Build a "peak kept" mask aligned to the FULL peaks array (length N).
    # An IBI is between consecutive peaks: ibi[i] = peaks[i+1] - peaks[i].
    # So clean_ibi mask of length N-1 tells us which IBI is good. By
    # convention the first peak (no preceding IBI) is kept; subsequent
    # peaks are kept iff the IBI ending at them was kept.
    n_peaks = len(rt)
    if n_peaks == 0:
        return _empty_scores()
    peak_kept = np.zeros(n_peaks, dtype=bool)
    peak_kept[0] = True
    if keep_radar.size > 0:
        peak_kept[1:1 + keep_radar.size] = keep_radar.astype(bool)

    rt_w_mask = (rt >= t0) & (rt <= t1)
    rt_w = rt[rt_w_mask]
    # Kept-peak times in window:
    rt_kept = rt[rt_w_mask & peak_kept]
    # IBIs in window (for inst-HR computation): aligned with kept peaks
    ibi_w = ibi_radar_ms[rt_w_mask[1:n_peaks] & peak_kept[1:]] if ibi_radar_ms.size > 0 else np.array([])

    # Beat match: each Polar beat -> nearest kept radar peak within window
    match_idx = np.full(len(pt_w), -1, dtype=int)
    if rt_kept.size > 0:
        j = np.searchsorted(rt_kept, pt_w)
        for i, t in enumerate(pt_w):
            cands = []
            if j[i] < len(rt_kept): cands.append(j[i])
            if j[i] > 0: cands.append(j[i] - 1)
            if cands:
                best = min(cands, key=lambda k: abs(rt_kept[k] - t))
                if abs(rt_kept[best] - t) <= window_s:
                    match_idx[i] = best
    n_matched = int((match_idx >= 0).sum())
    match_rate = n_matched / max(1, len(pt_w))

    # Paired IBI: ONLY pair when CONSECUTIVE Polar beats are both matched.
    # (Previously we paired the current matched Polar beat against the last
    # matched Polar beat regardless of how many Polar beats were skipped in
    # between -- that compared a 1-beat Polar IBI against a multi-beat radar
    # span and produced 300-400 ms "MAE" inflation.)
    paired_polar, paired_radar = [], []
    for i in range(1, len(pt_w)):
        m_curr = match_idx[i]
        m_prev = match_idx[i - 1]
        if m_curr < 0 or m_prev < 0:
            continue                       # need both consecutive Polar beats matched
        if m_curr <= m_prev:
            continue                       # matches should advance monotonically
        r_ibi = (rt_kept[m_curr] - rt_kept[m_prev]) * 1000.0
        p_ibi = polar_ibi_w[i]
        if 300 < r_ibi < 1500 and 300 < p_ibi < 1500:
            paired_polar.append(p_ibi)
            paired_radar.append(r_ibi)
    paired_polar = np.array(paired_polar)
    paired_radar = np.array(paired_radar)

    if paired_polar.size >= 3:
        r_ibi = float(np.corrcoef(paired_polar, paired_radar)[0, 1])
        mae_ibi = float(np.mean(np.abs(paired_radar - paired_polar)))
        bias_ibi = float(np.mean(paired_radar - paired_polar))
    else:
        r_ibi = mae_ibi = bias_ibi = np.nan

    # 1-Hz HR grid
    grid = np.arange(t0, t1, 1.0)
    polar_hr = polar_inst_hr(pt_w, polar_ibi_w, grid)
    radar_inst_bpm = 60_000.0 / ibi_w if ibi_w.size > 0 else np.array([])
    radar_hr = hold_at_grid(rt_kept, radar_inst_bpm, grid)
    mask = ~np.isnan(polar_hr) & ~np.isnan(radar_hr)
    if mask.sum() > 3:
        err = radar_hr[mask] - polar_hr[mask]
        mae_hr = float(np.mean(np.abs(err)))
        rmse_hr = float(np.sqrt(np.mean(err ** 2)))
        bias_hr = float(np.mean(err))
        r_hr = float(np.corrcoef(polar_hr[mask], radar_hr[mask])[0, 1])
    else:
        mae_hr = rmse_hr = bias_hr = r_hr = np.nan

    return {
        "n_polar_beats": int(len(pt_w)),
        "n_radar_peaks_total": int(len(rt_w)),
        "n_radar_kept": int(rt_kept.size),
        "match_rate": float(match_rate),
        "n_matched": n_matched,
        "n_paired_ibi": int(paired_polar.size),
        "paired_r": r_ibi,
        "paired_mae_ms": mae_ibi,
        "paired_bias_ms": bias_ibi,
        "hr_grid_r": r_hr,
        "hr_grid_mae_bpm": mae_hr,
        "hr_grid_rmse_bpm": rmse_hr,
        "hr_grid_bias_bpm": bias_hr,
        "polar_mean_hr": float(60000.0 / np.mean(polar_ibi_w)) if polar_ibi_w.size else np.nan,
        "radar_mean_hr": float(np.mean(radar_inst_bpm)) if radar_inst_bpm.size else np.nan,
    }


def _empty_scores():
    nan = float("nan")
    return {
        "n_polar_beats": 0, "n_radar_peaks_total": 0, "n_radar_kept": 0,
        "match_rate": nan, "n_matched": 0, "n_paired_ibi": 0,
        "paired_r": nan, "paired_mae_ms": nan, "paired_bias_ms": nan,
        "hr_grid_r": nan, "hr_grid_mae_bpm": nan, "hr_grid_rmse_bpm": nan,
        "hr_grid_bias_bpm": nan,
        "polar_mean_hr": nan, "radar_mean_hr": nan,
    }


def polar_inst_hr(beat_t, ibi_ms, grid):
    """Hold-last instantaneous HR (BPM) on a 1-Hz grid."""
    hr = 60_000.0 / ibi_ms
    return hold_at_grid(beat_t, hr, grid)


def hold_at_grid(bt, val, grid):
    out = np.full_like(grid, np.nan, dtype=float); j = 0
    if len(bt) == 0:
        return out
    for i, tg in enumerate(grid):
        while j + 1 < len(bt) and bt[j+1] <= tg: j += 1
        if bt[j] <= tg: out[i] = val[j]
    return out


# ---------------------------------------------------------------------------
# Run all trials
# ---------------------------------------------------------------------------
def parse_condition_dir(name: str):
    """'40cm60s' -> (40, 60). '60cm180s' -> (60, 180)."""
    m = re.match(r"(\d+)cm(\d+)s", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def run_all():
    rows = []
    if not DATA_ROOT.exists():
        raise SystemExit(f"Data root not found: {DATA_ROOT}")

    for cond_dir in sorted(DATA_ROOT.iterdir()):
        if not cond_dir.is_dir(): continue
        distance_cm, duration_s = parse_condition_dir(cond_dir.name)
        if distance_cm is None: continue

        for trial_dir in sorted(cond_dir.iterdir()):
            if not trial_dir.is_dir(): continue
            trial = trial_dir.name

            radar_path, polar_path = find_trial_files(trial_dir)
            if radar_path is None:
                print(f"[skip] no files in {trial_dir}")
                continue

            print(f"\n=== {cond_dir.name}/{trial}  ===")
            print(f"  radar: {radar_path.name}")
            print(f"  polar: {polar_path.name}")

            meta, radar_df = parse_radar_csv(radar_path)
            fs = float(meta.get("fs_hz", "100.0"))
            session_start = float(meta["session_start_unix"])
            target_gate = meta.get("target_gate", "?")
            bandpass_mode = meta.get("bandpass_mode", "?")

            z_rx = np.column_stack([
                radar_df["iq0_real"].to_numpy() + 1j * radar_df["iq0_imag"].to_numpy(),
                radar_df["iq1_real"].to_numpy() + 1j * radar_df["iq1_imag"].to_numpy(),
                radar_df["iq2_real"].to_numpy() + 1j * radar_df["iq2_imag"].to_numpy(),
                radar_df["iq3_real"].to_numpy() + 1j * radar_df["iq3_imag"].to_numpy(),
            ])
            phi_clean = clean_phase_from_rx(z_rx, fs)

            polar = load_polar(polar_path)
            polar_unix = polar["beat_unix"].to_numpy(np.float64)   # reconstructed true beat times
            polar_ibi  = polar["ibi_ms"].to_numpy(np.float64)

            # ---- Baseline ----
            bl = pipeline_baseline(phi_clean, fs)
            bl_scores = score_against_polar(
                bl["peaks_t"], bl["ibi"], bl["keep"],
                session_start, polar_unix, polar_ibi, fs,
                window_s=MATCH_WINDOW_S,
            )
            print(f"  [baseline] paired r={bl_scores['paired_r']:.3f}"
                  f"  MAE={bl_scores['paired_mae_ms']:.1f}ms"
                  f"  HR_grid_MAE={bl_scores['hr_grid_mae_bpm']:.1f}bpm"
                  f"  match={bl_scores['match_rate']*100:.0f}%")

            # ---- VMD+ACF ----
            try:
                vm = pipeline_vmd_acf(phi_clean, fs)
                vm_scores = score_against_polar(
                    vm["peaks_t"], vm["ibi"], vm["keep"],
                    session_start, polar_unix, polar_ibi, fs,
                    window_s=MATCH_WINDOW_S,
                )
                print(f"  [VMD+ACF ] paired r={vm_scores['paired_r']:.3f}"
                      f"  MAE={vm_scores['paired_mae_ms']:.1f}ms"
                      f"  HR_grid_MAE={vm_scores['hr_grid_mae_bpm']:.1f}bpm"
                      f"  match={vm_scores['match_rate']*100:.0f}%"
                      f"  cardiac_bpm={vm['cardiac_bpm']:.1f}  acf_bpm={vm['acf_bpm']:.1f}")
            except Exception as e:
                print(f"  [VMD+ACF ] FAILED: {e}")
                vm = {"cardiac_bpm": np.nan, "acf_bpm": np.nan, "cardiac_f0_hz": np.nan}
                vm_scores = _empty_scores()

            rows.append({
                "distance_cm": distance_cm,
                "duration_s": duration_s,
                "trial": trial,
                "target_gate": target_gate,
                "live_bandpass_mode": bandpass_mode,
                "polar_mean_hr": vm_scores["polar_mean_hr"],
                # Baseline
                "base_paired_r": bl_scores["paired_r"],
                "base_paired_mae_ms": bl_scores["paired_mae_ms"],
                "base_paired_bias_ms": bl_scores["paired_bias_ms"],
                "base_hr_mae_bpm": bl_scores["hr_grid_mae_bpm"],
                "base_hr_bias_bpm": bl_scores["hr_grid_bias_bpm"],
                "base_match_rate": bl_scores["match_rate"],
                "base_radar_mean_hr": bl_scores["radar_mean_hr"],
                "base_n_kept": bl_scores["n_radar_kept"],
                # VMD+ACF
                "vmd_cardiac_bpm": vm["cardiac_bpm"],
                "vmd_acf_bpm": vm["acf_bpm"],
                "vmd_paired_r": vm_scores["paired_r"],
                "vmd_paired_mae_ms": vm_scores["paired_mae_ms"],
                "vmd_paired_bias_ms": vm_scores["paired_bias_ms"],
                "vmd_hr_mae_bpm": vm_scores["hr_grid_mae_bpm"],
                "vmd_hr_bias_bpm": vm_scores["hr_grid_bias_bpm"],
                "vmd_match_rate": vm_scores["match_rate"],
                "vmd_radar_mean_hr": vm_scores["radar_mean_hr"],
                "vmd_n_kept": vm_scores["n_radar_kept"],
                "n_polar_beats": vm_scores["n_polar_beats"],
                "n_paired_ibi": vm_scores["n_paired_ibi"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[wrote] {RESULTS_CSV}")
    return df


# ---------------------------------------------------------------------------
# Aggregation + plotting
# ---------------------------------------------------------------------------
def make_plots(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    distances = sorted(df["distance_cm"].unique())

    def aggregate(metric: str, duration: int):
        sub = df[df["duration_s"] == duration]
        means = sub.groupby("distance_cm")[metric].mean().reindex(distances)
        stds  = sub.groupby("distance_cm")[metric].std().reindex(distances)
        return means.values, stds.values

    x = np.arange(len(distances))
    w = 0.18

    # Panel 1: Paired IBI MAE (ms)
    ax = axes[0, 0]
    for i, (dur, color, label) in enumerate([
        (60,  "#fab005", "60s "),
        (180, "#f76707", "180s"),
    ]):
        bl_mean, bl_std = aggregate("base_paired_mae_ms", dur)
        vm_mean, vm_std = aggregate("vmd_paired_mae_ms", dur)
        ax.bar(x + (-1.5 + 2*i)*w, bl_mean, w, yerr=bl_std, capsize=4,
               color=color, alpha=0.55, edgecolor="black", label=f"baseline {label}")
        ax.bar(x + (-0.5 + 2*i)*w, vm_mean, w, yerr=vm_std, capsize=4,
               color=color, alpha=1.0,  edgecolor="black", label=f"VMD+ACF {label}")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} cm" for d in distances])
    ax.set_ylabel("Paired IBI MAE (ms)")
    ax.set_title("Paired IBI MAE vs distance (lower = better)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel 2: 1-Hz HR MAE (bpm)
    ax = axes[0, 1]
    for i, (dur, color, label) in enumerate([
        (60,  "#fab005", "60s "),
        (180, "#f76707", "180s"),
    ]):
        bl_mean, bl_std = aggregate("base_hr_mae_bpm", dur)
        vm_mean, vm_std = aggregate("vmd_hr_mae_bpm", dur)
        ax.bar(x + (-1.5 + 2*i)*w, bl_mean, w, yerr=bl_std, capsize=4,
               color=color, alpha=0.55, edgecolor="black", label=f"baseline {label}")
        ax.bar(x + (-0.5 + 2*i)*w, vm_mean, w, yerr=vm_std, capsize=4,
               color=color, alpha=1.0,  edgecolor="black", label=f"VMD+ACF {label}")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} cm" for d in distances])
    ax.set_ylabel("1-Hz HR MAE (BPM)")
    ax.set_title("HR(t) MAE vs distance (lower = better)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel 3: Beat match rate (%)
    ax = axes[1, 0]
    for i, (dur, color, label) in enumerate([
        (60,  "#fab005", "60s "),
        (180, "#f76707", "180s"),
    ]):
        bl_mean, bl_std = aggregate("base_match_rate", dur)
        vm_mean, vm_std = aggregate("vmd_match_rate", dur)
        ax.bar(x + (-1.5 + 2*i)*w, 100*bl_mean, w, yerr=100*bl_std, capsize=4,
               color=color, alpha=0.55, edgecolor="black", label=f"baseline {label}")
        ax.bar(x + (-0.5 + 2*i)*w, 100*vm_mean, w, yerr=100*vm_std, capsize=4,
               color=color, alpha=1.0,  edgecolor="black", label=f"VMD+ACF {label}")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} cm" for d in distances])
    ax.set_ylabel("Beat match rate (%)")
    ax.set_title("Beat match rate vs distance (higher = better)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    # Panel 4: Mean HR error |radar - polar| (BPM)
    ax = axes[1, 1]
    df["base_hr_err"] = (df["base_radar_mean_hr"] - df["polar_mean_hr"]).abs()
    df["vmd_hr_err"]  = (df["vmd_radar_mean_hr"] - df["polar_mean_hr"]).abs()
    for i, (dur, color, label) in enumerate([
        (60,  "#fab005", "60s "),
        (180, "#f76707", "180s"),
    ]):
        sub = df[df["duration_s"] == dur]
        bl_mean = sub.groupby("distance_cm")["base_hr_err"].mean().reindex(distances).values
        vm_mean = sub.groupby("distance_cm")["vmd_hr_err"].mean().reindex(distances).values
        bl_std  = sub.groupby("distance_cm")["base_hr_err"].std().reindex(distances).values
        vm_std  = sub.groupby("distance_cm")["vmd_hr_err"].std().reindex(distances).values
        ax.bar(x + (-1.5 + 2*i)*w, bl_mean, w, yerr=bl_std, capsize=4,
               color=color, alpha=0.55, edgecolor="black", label=f"baseline {label}")
        ax.bar(x + (-0.5 + 2*i)*w, vm_mean, w, yerr=vm_std, capsize=4,
               color=color, alpha=1.0,  edgecolor="black", label=f"VMD+ACF {label}")
    ax.set_xticks(x); ax.set_xticklabels([f"{d} cm" for d in distances])
    ax.set_ylabel("|Radar mean HR - Polar mean HR| (BPM)")
    ax.set_title("Mean HR error vs distance (lower = better)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.suptitle("Distance / Duration Ablation: 12 trials (3 distances x 2 durations x 2 trials)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SUMMARY_PNG, dpi=130)
    print(f"[wrote] {SUMMARY_PNG}")


def make_summary_text(df: pd.DataFrame) -> str:
    lines = []
    add = lines.append
    add("=" * 90)
    add("BATCH EVAL: distance / duration ablation, 12 trials")
    add("=" * 90)
    add(f"data root  : {DATA_ROOT}")
    add(f"n_trials   : {len(df)}")
    add("")
    # Aggregated per (distance, duration)
    add("PER-CONDITION MEAN (averaging 2 trials):")
    add(f"  {'distance':>10}  {'duration':>10}  "
        f"{'paired_MAE_ms (BL)':>20}  {'paired_MAE_ms (VMD)':>22}  "
        f"{'HR_MAE_bpm (BL)':>18}  {'HR_MAE_bpm (VMD)':>20}  "
        f"{'match_rate (VMD)':>20}")
    for (dist, dur), sub in df.groupby(["distance_cm", "duration_s"]):
        add(f"  {dist:>10}  {dur:>10}  "
            f"{sub['base_paired_mae_ms'].mean():>20.1f}  "
            f"{sub['vmd_paired_mae_ms'].mean():>22.1f}  "
            f"{sub['base_hr_mae_bpm'].mean():>18.2f}  "
            f"{sub['vmd_hr_mae_bpm'].mean():>20.2f}  "
            f"{sub['vmd_match_rate'].mean()*100:>19.1f}%")
    add("")
    add("PER-TRIAL DETAIL (sorted by distance / duration / trial):")
    cols = ["distance_cm", "duration_s", "trial",
            "polar_mean_hr",
            "base_radar_mean_hr", "base_paired_mae_ms", "base_hr_mae_bpm", "base_match_rate",
            "vmd_radar_mean_hr",  "vmd_paired_mae_ms",  "vmd_hr_mae_bpm",  "vmd_match_rate",
            "vmd_cardiac_bpm", "vmd_acf_bpm",
            "n_polar_beats", "n_paired_ibi"]
    sub_df = df[cols].copy().sort_values(["distance_cm", "duration_s", "trial"])
    add(sub_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    add("")
    add("HEADLINE:")
    bl_mae = df["vmd_paired_mae_ms"].dropna().mean()
    vm_mae = df["base_paired_mae_ms"].dropna().mean()
    add(f"  Mean paired-IBI MAE  baseline={vm_mae:.1f} ms,  VMD+ACF={bl_mae:.1f} ms  "
        f"(delta {bl_mae - vm_mae:+.1f} ms)")
    add(f"  Mean HR-grid MAE     baseline={df['base_hr_mae_bpm'].mean():.2f} bpm,  "
        f"VMD+ACF={df['vmd_hr_mae_bpm'].mean():.2f} bpm")
    return "\n".join(lines)


def main():
    df = run_all()
    text = make_summary_text(df)
    SUMMARY_TXT.write_text(text, encoding="utf-8")
    print(f"[wrote] {SUMMARY_TXT}")
    make_plots(df)
    print()
    print(text)


if __name__ == "__main__":
    main()
