"""Reprocess a recorded radar IQ session and evaluate against Polar H10 ground truth.

Goal: validate that the live recording's poor result (r=0.044, +123 ms bias,
37% beat rejection) was caused by a too-narrow 0.8-2.0 Hz heartbeat bandpass.
Widening the band to 0.8-3.5 Hz should push the Pearson r above 0.5.

This script:
  1. Reads the radar IQ CSV (12 # metadata lines, then frame_idx + 4 RX of IQ)
  2. Reads the Polar H10 ground truth (deduped by epoch_s)
  3. Runs the radar_analysis pipeline TWICE on the same IQ:
       - OLD band: 0.8 - 2.0 Hz  (matches what was used live)
       - NEW band: 0.8 - 3.5 Hz  (the proposed fix)
  4. Time-aligns radar peak unix timestamps with Polar beat unix timestamps,
     beat-matches each Polar beat to its nearest radar peak within +/- 0.4 s,
     and computes Pearson r, MAE (ms), and bias (ms) for matched pairs.
  5. Prints a before/after summary table, writes the same text to
     reprocess_summary.txt, and saves a 4-panel matplotlib figure to
     reprocess_compare.png.

ASCII only. No optional deps. Headless (Agg) matplotlib. Does not modify any
existing algorithm code.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

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
    motion_mask,
    remove_dc,
)
from radar_analysis.heartbeat_extractors import bandpass  # noqa: E402
from radar_analysis.beat_detection import (  # noqa: E402
    clean_ibi,
    detect_beats,
    peaks_to_ibi_ms,
)


# ---------------------------------------------------------------------------
# Hard-coded literal paths (the question requires these exact files).
# ---------------------------------------------------------------------------
RADAR_CSV = (
    r"C:\Users\rapha\xwechat_files\wxid_5cgorr0uasr811_de98\msg\file"
    r"\2026-05\session_2026-05-10_15-37-09.csv"
)
POLAR_CSV = (
    r"C:\Users\rapha\xwechat_files\wxid_5cgorr0uasr811_de98\msg\file"
    r"\2026-05\polar_h10_unlabeled_20260510_153654(1).csv"
)

OUT_DIR = os.path.join(_REPO_ROOT, "scripts")
SUMMARY_TXT = os.path.join(OUT_DIR, "reprocess_summary.txt")
FIGURE_PNG = os.path.join(OUT_DIR, "reprocess_compare.png")

MATCH_WINDOW_S = 0.4  # +/- this for beat matching


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def parse_radar_csv(path: str) -> tuple[dict, pd.DataFrame]:
    """Parse `# key: value` metadata then load the CSV body.

    Metadata block is the leading run of lines starting with `#`; we stop at
    the first non-comment line, which is the CSV header.
    """
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
    """Read Polar H10 CSV. Polar emits duplicate rows occasionally — dedupe by epoch_s."""
    df = pd.read_csv(path)
    # Drop NaNs in the columns we depend on, then dedupe by epoch_s.
    df = df.dropna(subset=["epoch_s", "ibi_ms"])
    df = df.drop_duplicates(subset=["epoch_s"]).sort_values("epoch_s").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    label: str
    low_hz: float
    high_hz: float
    hb: np.ndarray            # bandpassed phase signal
    peaks_t: np.ndarray       # peak times in seconds (relative to session start)
    peaks_unix: np.ndarray    # peak times in unix seconds
    ibi_ms: np.ndarray        # all consecutive IBIs from peaks_t
    keep: np.ndarray          # bool mask from clean_ibi
    nn_ms: np.ndarray         # kept IBIs
    kept_peak_idx: np.ndarray # integer indices in hb for kept-IBI right-endpoint peaks
    motion_mask_frac_clean: float


def run_pipeline(
    z_rx: np.ndarray,
    fs: float,
    session_start_unix: float,
    *,
    low_hz: float,
    high_hz: float,
    label: str,
) -> PipelineResult:
    """Run the single-bin multi-RX coherent integration pipeline."""
    # Coherent average across RX (the question's specified recipe).
    z = z_rx.mean(axis=1)            # (F,) complex
    z_c = remove_dc(z)
    phi = extract_phase(z_c)
    phi = detrend_median(phi, fs)
    phi = despike_hampel(phi)        # defaults: k_w=12, n_sigma=3.0

    # We compute the motion mask for reporting (the question lists it in the API
    # surface) but do not gate the bandpass — the recipe in the question keeps
    # the full-length signal end to end. Reporting the clean fraction gives a
    # quick sanity number.
    mm = motion_mask(phi, fs)
    motion_frac_clean = float(mm.mean()) if mm.size else float("nan")

    hb = bandpass(phi, fs, low_hz=low_hz, high_hz=high_hz)
    peaks_t = detect_beats(hb, fs)                     # seconds, relative
    peaks_unix = session_start_unix + peaks_t          # unix seconds
    ibi_ms = peaks_to_ibi_ms(peaks_t)
    keep = clean_ibi(ibi_ms)
    nn_ms = ibi_ms[keep] if ibi_ms.size else ibi_ms

    # For overlaying kept peaks on hb in the figure: the i-th IBI is the gap
    # between peak i and peak i+1, so a "kept" IBI corresponds to the peak at
    # i+1. We use indices in hb (samples) for the scatter overlay.
    # detect_beats returns float (parabolic-refined) — round to int for index.
    peak_idx_all = np.round(peaks_t * fs).astype(int)
    if keep.size:
        kept_peak_idx = peak_idx_all[1:][keep]
    else:
        kept_peak_idx = np.array([], dtype=int)
    # Clamp to valid range just in case of off-by-one at boundaries.
    if kept_peak_idx.size:
        kept_peak_idx = kept_peak_idx[(kept_peak_idx >= 0) & (kept_peak_idx < hb.size)]

    return PipelineResult(
        label=label,
        low_hz=low_hz,
        high_hz=high_hz,
        hb=hb,
        peaks_t=peaks_t,
        peaks_unix=peaks_unix,
        ibi_ms=ibi_ms,
        keep=keep,
        nn_ms=nn_ms,
        kept_peak_idx=kept_peak_idx,
        motion_mask_frac_clean=motion_frac_clean,
    )


# ---------------------------------------------------------------------------
# Beat matching + agreement metrics
# ---------------------------------------------------------------------------
@dataclass
class MatchResult:
    n_polar: int
    n_radar: int
    n_matched: int
    polar_ibi: np.ndarray
    radar_ibi: np.ndarray
    pearson_r: float
    mae_ms: float
    bias_ms: float


def match_and_score(
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    radar_unix: np.ndarray,
    radar_ibi_ms: np.ndarray,
    radar_keep: np.ndarray,
    window_s: float = MATCH_WINDOW_S,
) -> MatchResult:
    """Match each Polar beat to nearest radar peak within +/- window_s, then
    pair the Polar IBI at beat i with the radar IBI ending at the matched
    radar peak. Only kept (clean_ibi) radar intervals participate.

    Returns Pearson r, MAE, and bias (radar - polar) on the matched pairs.
    """
    # The first Polar row's `ibi_ms` is the interval *ending* at that timestamp
    # (Polar emits the IBI when each beat lands). The first radar IBI is the
    # interval between peaks_unix[0] and peaks_unix[1], so it ends at
    # peaks_unix[1]. Indexing model: a radar IBI at radar index j ends at
    # radar_unix[j+1]. We pair using the timestamp of the IBI's right edge.
    if radar_unix.size < 2 or radar_ibi_ms.size == 0:
        return MatchResult(
            n_polar=int(polar_unix.size),
            n_radar=int(radar_unix.size),
            n_matched=0,
            polar_ibi=np.array([]),
            radar_ibi=np.array([]),
            pearson_r=float("nan"),
            mae_ms=float("nan"),
            bias_ms=float("nan"),
        )

    radar_end_unix = radar_unix[1:]                       # one per radar IBI
    radar_ibi = radar_ibi_ms
    keep_mask = radar_keep

    # Restrict to kept radar IBIs for fair scoring.
    radar_end_unix_k = radar_end_unix[keep_mask]
    radar_ibi_k = radar_ibi[keep_mask]

    if radar_end_unix_k.size == 0:
        return MatchResult(
            n_polar=int(polar_unix.size),
            n_radar=int(radar_unix.size),
            n_matched=0,
            polar_ibi=np.array([]),
            radar_ibi=np.array([]),
            pearson_r=float("nan"),
            mae_ms=float("nan"),
            bias_ms=float("nan"),
        )

    # For each Polar beat, find the nearest kept radar IBI end-timestamp.
    matched_polar = []
    matched_radar = []
    sorted_radar = radar_end_unix_k  # already sorted (peaks_t monotonic)
    j = 0  # walking pointer (radar is sorted, polar is sorted)
    for pt, pibi in zip(polar_unix, polar_ibi_ms):
        # advance j while next radar timestamp is closer
        while j + 1 < sorted_radar.size and abs(sorted_radar[j + 1] - pt) <= abs(sorted_radar[j] - pt):
            j += 1
        if abs(sorted_radar[j] - pt) <= window_s:
            matched_polar.append(float(pibi))
            matched_radar.append(float(radar_ibi_k[j]))

    if len(matched_polar) < 2:
        return MatchResult(
            n_polar=int(polar_unix.size),
            n_radar=int(radar_unix.size),
            n_matched=len(matched_polar),
            polar_ibi=np.array(matched_polar),
            radar_ibi=np.array(matched_radar),
            pearson_r=float("nan"),
            mae_ms=float("nan"),
            bias_ms=float("nan"),
        )

    polar_arr = np.array(matched_polar)
    radar_arr = np.array(matched_radar)
    # Pearson r — guard against zero variance.
    if polar_arr.std() == 0 or radar_arr.std() == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(polar_arr, radar_arr)[0, 1])
    mae = float(np.mean(np.abs(radar_arr - polar_arr)))
    bias = float(np.mean(radar_arr - polar_arr))

    return MatchResult(
        n_polar=int(polar_unix.size),
        n_radar=int(radar_unix.size),
        n_matched=int(polar_arr.size),
        polar_ibi=polar_arr,
        radar_ibi=radar_arr,
        pearson_r=r,
        mae_ms=mae,
        bias_ms=bias,
    )


# ---------------------------------------------------------------------------
# Reporting + plotting
# ---------------------------------------------------------------------------
def fmt_float(x: float, n: int = 3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  nan"
    return f"{x:.{n}f}"


def build_summary(
    polar_df: pd.DataFrame,
    polar_unix: np.ndarray,
    res_old: PipelineResult,
    res_new: PipelineResult,
    match_old: MatchResult,
    match_new: MatchResult,
    fs: float,
    n_frames: int,
    session_start_unix: float,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("Reprocess and evaluate: radar vs Polar H10")
    add("=" * 78)
    add("")
    add(f"Radar IQ CSV          : {RADAR_CSV}")
    add(f"Polar H10 CSV         : {POLAR_CSV}")
    add(f"Session start (unix)  : {session_start_unix:.6f}")
    add(f"Sampling rate fs      : {fs:.3f} Hz")
    add(f"Radar frames          : {n_frames}")
    add(f"Radar duration        : {n_frames / fs:.2f} s")
    add("")
    add("-" * 78)
    add("Polar H10 ground truth summary")
    add("-" * 78)
    if polar_df.shape[0] > 0:
        polar_dur = float(polar_unix[-1] - polar_unix[0])
        polar_mean_hr = float(polar_df["hr_bpm"].mean())
        polar_mean_ibi = float(polar_df["ibi_ms"].mean())
        add(f"  n_beats (deduped)     : {polar_df.shape[0]}")
        add(f"  duration              : {polar_dur:.2f} s")
        add(f"  mean HR (bpm)         : {polar_mean_hr:.2f}")
        add(f"  mean IBI (ms)         : {polar_mean_ibi:.2f}")
    else:
        add("  no Polar rows")
    add("")

    for res, match in ((res_old, match_old), (res_new, match_new)):
        add("-" * 78)
        add(f"Radar run: {res.label}  (bandpass {res.low_hz:.2f} - {res.high_hz:.2f} Hz)")
        add("-" * 78)
        n_peaks = int(res.peaks_t.size)
        n_ibi = int(res.ibi_ms.size)
        n_kept = int(res.keep.sum()) if res.keep.size else 0
        rej_pct = (100.0 * (1.0 - n_kept / n_ibi)) if n_ibi else float("nan")
        mean_hr = (60000.0 / float(np.mean(res.nn_ms))) if res.nn_ms.size else float("nan")
        mean_ibi = float(np.mean(res.nn_ms)) if res.nn_ms.size else float("nan")
        add(f"  n_peaks               : {n_peaks}")
        add(f"  n_ibis                : {n_ibi}")
        add(f"  n_kept (clean_ibi)    : {n_kept}")
        add(f"  beat rejection (%)    : {fmt_float(rej_pct, 2)}")
        add(f"  mean radar HR (bpm)   : {fmt_float(mean_hr, 2)}")
        add(f"  mean radar IBI (ms)   : {fmt_float(mean_ibi, 2)}")
        add(f"  motion_mask clean (%) : {fmt_float(100.0 * res.motion_mask_frac_clean, 2)}")
        add(f"  matched pairs         : {match.n_matched} of {match.n_polar} Polar beats")
        add(f"  Pearson r             : {fmt_float(match.pearson_r, 4)}")
        add(f"  MAE (ms)              : {fmt_float(match.mae_ms, 2)}")
        add(f"  bias radar - polar ms : {fmt_float(match.bias_ms, 2)}")
        add("")

    add("=" * 78)
    add("Before / After comparison")
    add("=" * 78)
    hdr = "  {metric:<24}  {old:>14}  {new:>14}  {delta:>12}"
    add(hdr.format(metric="metric", old="OLD 0.8-2.0", new="NEW 0.8-3.5", delta="delta"))
    add("  " + "-" * 70)

    def row(metric: str, old_v: float, new_v: float, n: int = 3, lower_is_better: bool = False) -> str:
        if np.isfinite(old_v) and np.isfinite(new_v):
            delta = new_v - old_v
            d_str = f"{delta:+.{n}f}"
        else:
            d_str = "    n/a"
        return hdr.format(
            metric=metric,
            old=fmt_float(old_v, n),
            new=fmt_float(new_v, n),
            delta=d_str,
        )

    n_old_ibi = int(res_old.ibi_ms.size)
    n_new_ibi = int(res_new.ibi_ms.size)
    rej_old = (100.0 * (1.0 - int(res_old.keep.sum()) / n_old_ibi)) if n_old_ibi else float("nan")
    rej_new = (100.0 * (1.0 - int(res_new.keep.sum()) / n_new_ibi)) if n_new_ibi else float("nan")

    add(row("n_peaks", float(res_old.peaks_t.size), float(res_new.peaks_t.size), n=0))
    add(row("n_kept", float(int(res_old.keep.sum())), float(int(res_new.keep.sum())), n=0))
    add(row("beat rejection (%)", rej_old, rej_new, n=2, lower_is_better=True))
    add(row("matched pairs", float(match_old.n_matched), float(match_new.n_matched), n=0))
    add(row("Pearson r", match_old.pearson_r, match_new.pearson_r, n=4))
    add(row("MAE (ms)", match_old.mae_ms, match_new.mae_ms, n=2, lower_is_better=True))
    add(row("bias (ms)", match_old.bias_ms, match_new.bias_ms, n=2))
    add("")

    # Verdict
    add("-" * 78)
    add("Verdict")
    add("-" * 78)
    agree = (
        np.isfinite(match_new.pearson_r)
        and match_new.pearson_r > 0.5
        and (not np.isfinite(match_old.pearson_r) or match_new.pearson_r > match_old.pearson_r)
    )
    if agree:
        add("  PASS: widening the bandpass to 0.8-3.5 Hz pushed Pearson r above 0.5")
        add("        and improved over the 0.8-2.0 Hz baseline. The narrow band was")
        add("        starving the heartbeat extractor of its harmonic content.")
    else:
        add("  FAIL: widening the bandpass did NOT push Pearson r above 0.5 (or did")
        add("        not improve over baseline). The narrow band is not the only")
        add("        problem -- inspect Panel 3 of the figure for residual issues")
        add("        (DC removal, motion, range bin, RX combining).")
    add("=" * 78)
    return "\n".join(lines) + "\n"


def make_figure(
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    res_old: PipelineResult,
    res_new: PipelineResult,
    match_new: MatchResult,
    fs: float,
    session_start_unix: float,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))

    # --- Panel 1: IBI(t) overlay ---
    ax = axes[0]
    if polar_unix.size:
        ax.plot(polar_unix - session_start_unix, polar_ibi_ms,
                color="green", lw=1.4, label="Polar H10 (truth)")
    # radar IBI is plotted at right-edge of each interval
    if res_old.peaks_unix.size >= 2 and res_old.ibi_ms.size:
        t_old = res_old.peaks_unix[1:] - session_start_unix
        keep_old = res_old.keep
        ax.plot(t_old[keep_old], res_old.ibi_ms[keep_old],
                color="red", alpha=0.6, lw=1.0, marker=".",
                label="Radar 0.8-2.0 Hz (OLD)")
    if res_new.peaks_unix.size >= 2 and res_new.ibi_ms.size:
        t_new = res_new.peaks_unix[1:] - session_start_unix
        keep_new = res_new.keep
        ax.plot(t_new[keep_new], res_new.ibi_ms[keep_new],
                color="blue", lw=1.0, marker=".",
                label="Radar 0.8-3.5 Hz (NEW)")
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("IBI (ms)")
    ax.set_title("IBI(t): Polar truth vs radar (old vs new band)")
    ax.set_ylim(300, 1500)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 2: scatter NEW radar IBI vs Polar IBI ---
    ax = axes[1]
    if match_new.n_matched >= 2:
        ax.scatter(match_new.polar_ibi, match_new.radar_ibi, s=14,
                   color="blue", alpha=0.7, label=f"n={match_new.n_matched}")
        lo = float(min(match_new.polar_ibi.min(), match_new.radar_ibi.min())) - 20.0
        hi = float(max(match_new.polar_ibi.max(), match_new.radar_ibi.max())) + 20.0
        ax.plot([lo, hi], [lo, hi], color="gray", ls="--", lw=1.0, label="y=x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Polar IBI (ms)")
        ax.set_ylabel("Radar IBI (ms) [0.8-3.5 Hz]")
        ax.set_title(
            f"Scatter NEW band  r={match_new.pearson_r:.3f}  "
            f"MAE={match_new.mae_ms:.1f} ms  bias={match_new.bias_ms:+.1f} ms"
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
    else:
        ax.text(0.5, 0.5, "not enough matched pairs",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Scatter NEW band: insufficient data")

    # --- Panel 3: hb NEW with kept peaks overlay ---
    ax = axes[2]
    t_axis = np.arange(res_new.hb.size) / fs
    ax.plot(t_axis, res_new.hb, color="blue", lw=0.7, label="hb (0.8-3.5 Hz)")
    if res_new.kept_peak_idx.size:
        idx = res_new.kept_peak_idx
        ax.plot(t_axis[idx], res_new.hb[idx], "ro", ms=3.0, label="kept peaks")
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("hb (rad)")
    ax.set_title("Bandpass output  NEW  0.8-3.5 Hz  (heartbeat-like ideally)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 4: hb OLD with kept peaks overlay ---
    ax = axes[3]
    ax.plot(t_axis, res_old.hb, color="red", lw=0.7, label="hb (0.8-2.0 Hz)")
    if res_old.kept_peak_idx.size:
        idx = res_old.kept_peak_idx
        ax.plot(t_axis[idx], res_old.hb[idx], "ko", ms=3.0, label="kept peaks")
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("hb (rad)")
    ax.set_title("Bandpass output  OLD  0.8-2.0 Hz  (narrower band -> more resonant)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    # --- Load radar ---
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
    # Build (F, 4) complex from CSV columns.
    z_rx = np.column_stack([
        radar_df[r].to_numpy(dtype=np.float64) + 1j * radar_df[i].to_numpy(dtype=np.float64)
        for r, i in iq_cols
    ])

    # --- Load Polar ---
    polar_df = load_polar(POLAR_CSV)
    polar_unix = polar_df["epoch_s"].to_numpy(dtype=np.float64)
    polar_ibi_ms = polar_df["ibi_ms"].to_numpy(dtype=np.float64)

    # --- Run pipeline twice ---
    res_old = run_pipeline(
        z_rx, fs, session_start_unix,
        low_hz=0.8, high_hz=2.0, label="OLD (0.8 - 2.0 Hz)",
    )
    res_new = run_pipeline(
        z_rx, fs, session_start_unix,
        low_hz=0.8, high_hz=3.5, label="NEW (0.8 - 3.5 Hz)",
    )

    # --- Match & score ---
    match_old = match_and_score(
        polar_unix, polar_ibi_ms,
        res_old.peaks_unix, res_old.ibi_ms, res_old.keep,
        window_s=MATCH_WINDOW_S,
    )
    match_new = match_and_score(
        polar_unix, polar_ibi_ms,
        res_new.peaks_unix, res_new.ibi_ms, res_new.keep,
        window_s=MATCH_WINDOW_S,
    )

    # --- Report ---
    summary = build_summary(
        polar_df, polar_unix,
        res_old, res_new,
        match_old, match_new,
        fs=fs,
        n_frames=n_frames,
        session_start_unix=session_start_unix,
    )
    print(summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[wrote] {SUMMARY_TXT}")

    make_figure(
        polar_unix, polar_ibi_ms,
        res_old, res_new, match_new,
        fs=fs, session_start_unix=session_start_unix,
        out_path=FIGURE_PNG,
    )
    print(f"[wrote] {FIGURE_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
