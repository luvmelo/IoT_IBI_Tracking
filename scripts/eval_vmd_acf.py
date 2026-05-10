"""VMD + autocorrelation heartbeat extractor vs the bandpass baseline.

This script replaces the per-frame Butterworth bandpass + find_peaks pipeline
with a Variational Mode Decomposition (VMD) front-end followed by an
autocorrelation cross-check on the cardiac mode. The bandpass baseline has two
known failure modes:

  1. Butter4 bandpass *resonates* on low-SNR input, generating fake periodic
     peaks at the filter center.
  2. find_peaks doesn't disambiguate the true fundamental from the 2nd
     harmonic, so BPM doubles when systolic + diastolic both pass the
     prominence threshold.

VMD decomposes the cleaned phase into K modes whose center frequencies are
*data-found* (no designer-imposed band -> no resonance). We pick the mode whose
center sits in [0.9, 2.0] Hz (cardiac band) and has highest variance. Beat
detection then runs on that mode directly; an autocorrelation of the same mode
provides a robust BPM cross-check from the full signal energy.

Hardcoded inputs (per the task spec):

    Radar IQ CSV : C:\\Users\\rapha\\Desktop\\MIT\\Sensor & Mobile Computing\\
                   Final\\MakeyMakey\\recordings\\session_2026-05-10_16-11-27.csv
    Polar H10    : C:\\Users\\rapha\\xwechat_files\\wxid_5cgorr0uasr811_de98\\
                   msg\\file\\2026-05\\
                   polar_h10_radar_sync_01_20260510_160910(1).csv

ASCII only, headless matplotlib, no live UI. Writes:

    scripts/eval_vmd_acf_summary.txt      summary text (printed and saved)
    scripts/eval_vmd_acf.png              5-panel figure
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import correlate, welch

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

# Prefer vmdpy if available; fall back to the inline implementation otherwise.
try:
    from vmdpy import VMD as _VMD_PYPI  # type: ignore

    _HAS_VMDPY = True
except Exception:  # pragma: no cover - exercised only when pip install fails
    _HAS_VMDPY = False


# ---------------------------------------------------------------------------
# Hard-coded literal paths (the task requires these exact files).
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
SUMMARY_TXT = os.path.join(OUT_DIR, "eval_vmd_acf_summary.txt")
FIGURE_PNG = os.path.join(OUT_DIR, "eval_vmd_acf.png")

MATCH_WINDOW_S = 0.4  # +/- this for beat matching against Polar


# ---------------------------------------------------------------------------
# Inline VMD reference implementation (fallback if vmdpy is unavailable).
# Adapted from the canonical Dragomiretskiy & Zosso 2014 algorithm.
# ---------------------------------------------------------------------------
def _vmd_inline(signal, alpha, tau, K, DC, init, tol):
    """Reference VMD implementation. Returns u (K, len(signal)), u_hat, omega."""
    save_T = len(signal)
    T = save_T
    f_mirror = np.concatenate(
        [signal[T // 2 - 1::-1], signal, signal[-1:-T // 2 - 1:-1]]
    )
    T = len(f_mirror)
    t = np.arange(1, T + 1) / T
    freqs = t - 0.5 - 1 / T
    N = 500
    Alpha = alpha * np.ones(K)
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    f_hat_plus = np.copy(f_hat)
    f_hat_plus[: T // 2] = 0
    u_hat_plus = np.zeros((N, len(freqs), K), dtype=complex)
    omega_plus = np.zeros((N, K))
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    lam_hat = np.zeros((N, len(freqs)), dtype=complex)
    uDiff = tol + np.spacing(1)
    n = 0
    sum_uk = 0
    while uDiff > tol and n < N - 1:
        k = 0
        sum_uk = u_hat_plus[n, :, K - 1] + sum_uk - u_hat_plus[n, :, 0]
        u_hat_plus[n + 1, :, k] = (
            f_hat_plus - sum_uk - lam_hat[n, :] / 2
        ) / (1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
        if not DC:
            omega_plus[n + 1, k] = (
                freqs[T // 2:T]
                @ (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
            ) / np.sum(abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
        for k in range(1, K):
            sum_uk = u_hat_plus[n + 1, :, k - 1] + sum_uk - u_hat_plus[n, :, k]
            u_hat_plus[n + 1, :, k] = (
                f_hat_plus - sum_uk - lam_hat[n, :] / 2
            ) / (1 + Alpha[k] * (freqs - omega_plus[n, k]) ** 2)
            omega_plus[n + 1, k] = (
                freqs[T // 2:T]
                @ (abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
            ) / np.sum(abs(u_hat_plus[n + 1, T // 2:T, k]) ** 2)
        lam_hat[n + 1, :] = lam_hat[n, :] + tau * (
            np.sum(u_hat_plus[n + 1, :, :], axis=1) - f_hat_plus
        )
        n = n + 1
        uDiff = np.spacing(1)
        for i in range(K):
            uDiff = uDiff + 1 / T * (
                u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i]
            ) @ np.conj(u_hat_plus[n, :, i] - u_hat_plus[n - 1, :, i])
        uDiff = np.abs(uDiff)
    N = min(N, n)
    omega = omega_plus[:N, :]
    u_hat = np.zeros((len(freqs), K), dtype=complex)
    u_hat[T // 2:T, :] = np.squeeze(u_hat_plus[N - 1, T // 2:T, :])
    u_hat[T // 2:0:-1, :] = np.squeeze(np.conj(u_hat_plus[N - 1, T // 2:T, :]))
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.zeros((K, len(t)))
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))
    u = u[:, T // 4:3 * T // 4]
    return u, u_hat, omega


def _vmd(signal, alpha, tau, K, DC, init, tol):
    """Thin wrapper: prefer vmdpy, fall back to inline implementation."""
    if _HAS_VMDPY:
        return _VMD_PYPI(signal, alpha, tau, K, DC, init, tol)
    return _vmd_inline(signal, alpha, tau, K, DC, init, tol)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def parse_radar_csv(path: str) -> tuple[dict, pd.DataFrame]:
    """Parse `# key: value` metadata then load the CSV body."""
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
    df = pd.read_csv(path)
    df = df.dropna(subset=["epoch_s", "ibi_ms"])
    df = df.drop_duplicates(subset=["epoch_s"]).sort_values("epoch_s").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    label: str
    signal: np.ndarray  # the time-domain trace whose peaks we detect
    peaks_t: np.ndarray
    peaks_unix: np.ndarray
    ibi_ms: np.ndarray
    keep: np.ndarray
    nn_ms: np.ndarray
    kept_peak_idx: np.ndarray  # indices into self.signal for kept-IBI right endpoints


@dataclass
class VMDExtra:
    u: np.ndarray              # (K, N)
    omega_hz: np.ndarray       # (K,) final mode center frequencies in Hz
    cardiac_idx: int
    cardiac_f0_hz: float
    acf: np.ndarray            # autocorrelation (one-sided, normalized)
    acf_lag_peak: int
    acf_bpm: float


def _clean_phase(z_rx: np.ndarray, fs: float) -> np.ndarray:
    """Coherent RX average -> remove DC -> phase -> detrend -> Hampel.

    Returns the cleaned phase that BOTH pipelines consume. Keeping it shared
    ensures the comparison isolates the bandpass-vs-VMD effect.
    """
    z = z_rx.mean(axis=1)
    z = remove_dc(z)
    phi = extract_phase(z)
    phi = detrend_median(phi, fs)
    phi = despike_hampel(phi)
    return phi


def _peaks_to_pipeline_result(
    signal: np.ndarray,
    peaks_t: np.ndarray,
    session_start_unix: float,
    label: str,
    fs: float,
) -> PipelineResult:
    peaks_unix = session_start_unix + peaks_t
    ibi_ms = peaks_to_ibi_ms(peaks_t)
    keep = clean_ibi(ibi_ms)
    nn_ms = ibi_ms[keep] if ibi_ms.size else ibi_ms
    peak_idx_all = np.round(peaks_t * fs).astype(int)
    if keep.size:
        kept_peak_idx = peak_idx_all[1:][keep]
    else:
        kept_peak_idx = np.array([], dtype=int)
    if kept_peak_idx.size:
        kept_peak_idx = kept_peak_idx[
            (kept_peak_idx >= 0) & (kept_peak_idx < signal.size)
        ]
    return PipelineResult(
        label=label,
        signal=signal,
        peaks_t=peaks_t,
        peaks_unix=peaks_unix,
        ibi_ms=ibi_ms,
        keep=keep,
        nn_ms=nn_ms,
        kept_peak_idx=kept_peak_idx,
    )


def run_baseline(
    phi_clean: np.ndarray,
    fs: float,
    session_start_unix: float,
) -> PipelineResult:
    """Bandpass 0.8-2.5 Hz + find_peaks. Same cleaning stage as the VMD path."""
    hb = bandpass(phi_clean, fs, low_hz=0.8, high_hz=2.5, order=4)
    peaks_t = detect_beats(hb, fs, max_bpm=200.0, prominence_factor=0.5)
    return _peaks_to_pipeline_result(
        hb, peaks_t, session_start_unix, "BASELINE (0.8-2.5 Hz bandpass)", fs
    )


def run_vmd(
    phi_clean: np.ndarray,
    fs: float,
    session_start_unix: float,
    *,
    K: int = 5,
    alpha: float = 2000.0,
    tau: float = 0.0,
    DC: int = 0,
    init: int = 1,
    tol: float = 1e-7,
) -> tuple[PipelineResult, VMDExtra]:
    """VMD decomposition -> cardiac-mode selection -> find_peaks on that mode.

    Also computes an autocorrelation-based BPM cross-check on the same mode.
    """
    # Run VMD on the cleaned phase. vmdpy's omega is normalized by sampling
    # frequency 1/T with T=len(signal), so multiply by fs to get Hz.
    u, u_hat, omega = _vmd(phi_clean, alpha, tau, K, DC, init, tol)
    # If vmdpy trimmed the signal by one sample (even-length symmetry), pad.
    if u.shape[1] < phi_clean.size:
        pad = phi_clean.size - u.shape[1]
        u = np.pad(u, ((0, 0), (0, pad)), mode="edge")
    elif u.shape[1] > phi_clean.size:
        u = u[:, : phi_clean.size]

    # omega[-1, :] are the final center frequencies (normalized). Multiply by fs.
    final_omega_hz = omega[-1, :] * fs

    # Pick cardiac mode: in-band [0.9, 2.0] Hz, highest variance.
    in_band = (final_omega_hz >= 0.9) & (final_omega_hz <= 2.0)
    if not in_band.any():
        cardiac_idx = int(np.argmin(np.abs(final_omega_hz - 1.4)))
    else:
        powers = np.var(u, axis=1)
        in_band_idx = np.where(in_band)[0]
        cardiac_idx = int(in_band_idx[np.argmax(powers[in_band_idx])])

    cardiac = u[cardiac_idx]
    cardiac_f0_hz = float(final_omega_hz[cardiac_idx])

    # Beat detection on the cardiac mode directly (no extra filtering).
    peaks_t = detect_beats(cardiac, fs, max_bpm=200.0, prominence_factor=0.5)

    # Autocorrelation cross-check (BPM estimate over full signal energy).
    cardiac_centered = cardiac - cardiac.mean()
    acf_full = correlate(cardiac_centered, cardiac_centered, mode="full")
    acf = acf_full[len(cardiac_centered) - 1:]
    if acf[0] > 0:
        acf = acf / acf[0]
    lag_min = max(1, int(60.0 / 180.0 * fs))  # min lag at 180 BPM
    lag_max = min(len(acf) - 1, int(60.0 / 50.0 * fs))  # max lag at 50 BPM
    if lag_max <= lag_min:
        acf_bpm = float("nan")
        acf_lag_peak = lag_min
    else:
        acf_window = acf[lag_min:lag_max + 1]
        acf_lag_peak = int(lag_min + int(np.argmax(acf_window)))
        if acf_lag_peak <= 0:
            acf_bpm = float("nan")
        else:
            acf_bpm = float(60.0 / (acf_lag_peak / fs))

    pipeline_label = (
        f"VMD+ACF (K={K}, alpha={alpha:.0f}, cardiac f0={cardiac_f0_hz:.2f} Hz)"
    )
    result = _peaks_to_pipeline_result(
        cardiac, peaks_t, session_start_unix, pipeline_label, fs
    )
    extra = VMDExtra(
        u=u,
        omega_hz=final_omega_hz,
        cardiac_idx=cardiac_idx,
        cardiac_f0_hz=cardiac_f0_hz,
        acf=acf,
        acf_lag_peak=acf_lag_peak,
        acf_bpm=acf_bpm,
    )
    return result, extra


# ---------------------------------------------------------------------------
# Beat matching + 1 Hz HR resampling
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
    """Match each Polar beat to the nearest kept radar IBI right edge."""
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
    radar_end_unix = radar_unix[1:]
    radar_end_unix_k = radar_end_unix[radar_keep]
    radar_ibi_k = radar_ibi_ms[radar_keep]
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
    sorted_radar = radar_end_unix_k
    matched_polar: list[float] = []
    matched_radar: list[float] = []
    j = 0
    for pt, pibi in zip(polar_unix, polar_ibi_ms):
        while (
            j + 1 < sorted_radar.size
            and abs(sorted_radar[j + 1] - pt) <= abs(sorted_radar[j] - pt)
        ):
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


@dataclass
class HRGridResult:
    grid_unix: np.ndarray
    polar_hr: np.ndarray
    radar_hr: np.ndarray
    pearson_r: float
    mae_bpm: float
    rmse_bpm: float
    bias_bpm: float


def _ibi_to_instantaneous_hr(unix_at_end: np.ndarray, ibi_ms: np.ndarray):
    """Map IBI -> instantaneous HR sample at right-edge of each interval."""
    if ibi_ms.size == 0:
        return np.array([]), np.array([])
    hr = 60000.0 / np.maximum(ibi_ms, 1e-6)
    return unix_at_end, hr


def _interp_hr_grid(
    grid_unix: np.ndarray, t: np.ndarray, hr: np.ndarray
) -> np.ndarray:
    """Linear-interpolate HR(t) onto grid_unix. Outside-range -> NaN."""
    if t.size == 0:
        return np.full_like(grid_unix, np.nan, dtype=np.float64)
    out = np.interp(grid_unix, t, hr, left=np.nan, right=np.nan)
    # np.interp doesn't put NaN by default; replace out-of-range with NaN.
    out = np.where((grid_unix < t[0]) | (grid_unix > t[-1]), np.nan, out)
    return out


def hr_on_1hz_grid(
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    radar_unix_peaks: np.ndarray,
    radar_ibi_ms: np.ndarray,
    radar_keep: np.ndarray,
) -> HRGridResult:
    """Build a 1 Hz HR grid spanning the overlap of Polar and radar coverage,
    then compute scalar metrics on the non-NaN region of both series.
    """
    p_t, p_hr = _ibi_to_instantaneous_hr(polar_unix, polar_ibi_ms)
    if radar_unix_peaks.size >= 2 and radar_ibi_ms.size:
        r_t_all = radar_unix_peaks[1:]
        r_t = r_t_all[radar_keep]
        r_ibi = radar_ibi_ms[radar_keep]
    else:
        r_t = np.array([])
        r_ibi = np.array([])
    r_t, r_hr = _ibi_to_instantaneous_hr(r_t, r_ibi)
    if p_t.size == 0 or r_t.size == 0:
        return HRGridResult(
            grid_unix=np.array([]),
            polar_hr=np.array([]),
            radar_hr=np.array([]),
            pearson_r=float("nan"),
            mae_bpm=float("nan"),
            rmse_bpm=float("nan"),
            bias_bpm=float("nan"),
        )
    t0 = max(p_t[0], r_t[0])
    t1 = min(p_t[-1], r_t[-1])
    if t1 - t0 < 2.0:
        return HRGridResult(
            grid_unix=np.array([]),
            polar_hr=np.array([]),
            radar_hr=np.array([]),
            pearson_r=float("nan"),
            mae_bpm=float("nan"),
            rmse_bpm=float("nan"),
            bias_bpm=float("nan"),
        )
    grid = np.arange(np.ceil(t0), np.floor(t1) + 1.0, 1.0)
    p_on = _interp_hr_grid(grid, p_t, p_hr)
    r_on = _interp_hr_grid(grid, r_t, r_hr)
    mask = np.isfinite(p_on) & np.isfinite(r_on)
    if mask.sum() < 2:
        pearson_r = float("nan")
        mae = float("nan")
        rmse = float("nan")
        bias = float("nan")
    else:
        a = p_on[mask]
        b = r_on[mask]
        if a.std() == 0 or b.std() == 0:
            pearson_r = float("nan")
        else:
            pearson_r = float(np.corrcoef(a, b)[0, 1])
        mae = float(np.mean(np.abs(b - a)))
        rmse = float(np.sqrt(np.mean((b - a) ** 2)))
        bias = float(np.mean(b - a))
    return HRGridResult(
        grid_unix=grid,
        polar_hr=p_on,
        radar_hr=r_on,
        pearson_r=pearson_r,
        mae_bpm=mae,
        rmse_bpm=rmse,
        bias_bpm=bias,
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
    res_base: PipelineResult,
    res_vmd: PipelineResult,
    extra: VMDExtra,
    match_base: MatchResult,
    match_vmd: MatchResult,
    hr_base: HRGridResult,
    hr_vmd: HRGridResult,
    fs: float,
    n_frames: int,
    session_start_unix: float,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("=" * 78)
    add("VMD + Autocorrelation vs Bandpass baseline -- evaluation report")
    add("=" * 78)
    add("")
    add(f"Radar IQ CSV          : {RADAR_CSV}")
    add(f"Polar H10 CSV         : {POLAR_CSV}")
    add(f"vmdpy installed       : {_HAS_VMDPY}")
    add(f"Session start (unix)  : {session_start_unix:.6f}")
    add(f"Sampling rate fs      : {fs:.3f} Hz")
    add(f"Radar frames          : {n_frames}")
    add(f"Radar duration        : {n_frames / fs:.2f} s")
    add("")
    add("-" * 78)
    add("Polar H10 ground truth")
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
    add("-" * 78)
    add(f"BASELINE: {res_base.label}")
    add("-" * 78)
    _add_pipeline_block(add, res_base, match_base, hr_base)

    add("-" * 78)
    add(f"VMD: {res_vmd.label}")
    add("-" * 78)
    add(f"  VMD mode count K      : {extra.u.shape[0]}")
    omega_str = ", ".join(f"{f:.3f}" for f in extra.omega_hz)
    add(f"  VMD center freqs (Hz) : [{omega_str}]")
    add(f"  Cardiac mode index    : {extra.cardiac_idx}")
    add(f"  Cardiac f0 (Hz)       : {extra.cardiac_f0_hz:.3f}")
    add(f"  Cardiac f0 -> BPM     : {extra.cardiac_f0_hz * 60.0:.2f}")
    add(f"  ACF-based BPM         : {fmt_float(extra.acf_bpm, 2)}")
    _add_pipeline_block(add, res_vmd, match_vmd, hr_vmd)

    add("=" * 78)
    add("Before / After comparison  (BASELINE  vs  VMD+ACF)")
    add("=" * 78)
    hdr = "  {metric:<28}  {base:>14}  {vmd:>14}  {delta:>14}"
    add(hdr.format(metric="metric", base="BASELINE", vmd="VMD+ACF", delta="delta"))
    add("  " + "-" * 74)

    def row(metric: str, base_v: float, vmd_v: float, n: int = 3) -> str:
        if (isinstance(base_v, float) and np.isfinite(base_v)) and (
            isinstance(vmd_v, float) and np.isfinite(vmd_v)
        ):
            delta = vmd_v - base_v
            delta_str = f"{delta:+.{n}f}"
        else:
            delta_str = " nan"
        return hdr.format(
            metric=metric,
            base=fmt_float(base_v, n),
            vmd=fmt_float(vmd_v, n),
            delta=delta_str,
        )

    n_base = int(res_base.ibi_ms.size)
    n_vmd_i = int(res_vmd.ibi_ms.size)
    rej_base = (
        100.0 * (1.0 - int(res_base.keep.sum()) / n_base) if n_base else float("nan")
    )
    rej_vmd = (
        100.0 * (1.0 - int(res_vmd.keep.sum()) / n_vmd_i) if n_vmd_i else float("nan")
    )
    mean_hr_base = (
        60000.0 / float(np.mean(res_base.nn_ms)) if res_base.nn_ms.size else float("nan")
    )
    mean_hr_vmd = (
        60000.0 / float(np.mean(res_vmd.nn_ms)) if res_vmd.nn_ms.size else float("nan")
    )

    add(row("n_peaks",          float(res_base.peaks_t.size), float(res_vmd.peaks_t.size), n=0))
    add(row("n_kept",           float(int(res_base.keep.sum())), float(int(res_vmd.keep.sum())), n=0))
    add(row("beat rejection (%)", rej_base, rej_vmd, n=2))
    add(row("mean radar HR (bpm)", mean_hr_base, mean_hr_vmd, n=2))
    add(row("matched pairs",    float(match_base.n_matched), float(match_vmd.n_matched), n=0))
    add(row("paired Pearson r", match_base.pearson_r, match_vmd.pearson_r, n=4))
    add(row("paired MAE (ms)",  match_base.mae_ms,    match_vmd.mae_ms,    n=2))
    add(row("paired bias (ms)", match_base.bias_ms,   match_vmd.bias_ms,   n=2))
    add(row("1Hz HR Pearson r", hr_base.pearson_r,    hr_vmd.pearson_r,    n=4))
    add(row("1Hz HR MAE (bpm)", hr_base.mae_bpm,      hr_vmd.mae_bpm,      n=3))
    add(row("1Hz HR RMSE (bpm)", hr_base.rmse_bpm,    hr_vmd.rmse_bpm,     n=3))
    add(row("1Hz HR bias (bpm)", hr_base.bias_bpm,    hr_vmd.bias_bpm,     n=3))
    add("")

    add("-" * 78)
    add("Verdict")
    add("-" * 78)
    win_r = (
        np.isfinite(match_base.pearson_r)
        and np.isfinite(match_vmd.pearson_r)
        and (match_vmd.pearson_r > match_base.pearson_r)
    )
    win_mae = (
        np.isfinite(match_base.mae_ms)
        and np.isfinite(match_vmd.mae_ms)
        and (match_vmd.mae_ms < match_base.mae_ms)
    )
    if win_r and win_mae:
        add("  VERDICT: WIN -- VMD+ACF beats baseline on BOTH Pearson r AND MAE.")
        add(
            f"    delta Pearson r: {match_vmd.pearson_r - match_base.pearson_r:+.4f}"
        )
        add(
            f"    delta MAE (ms) : {match_vmd.mae_ms - match_base.mae_ms:+.2f}"
        )
    else:
        add("  VERDICT: LOSS or NEUTRAL -- investigate panel 3 (Welch spectrum + VMD")
        add("           center frequencies). Likely causes: cardiac mode landed on a")
        add("           respiration harmonic, alpha bandwidth was too loose / tight,")
        add("           or K=5 is wrong for this SNR regime.")
        add(
            f"    delta Pearson r: {match_vmd.pearson_r - match_base.pearson_r:+.4f}"
        )
        add(
            f"    delta MAE (ms) : {match_vmd.mae_ms - match_base.mae_ms:+.2f}"
        )
    add("=" * 78)
    return "\n".join(lines) + "\n"


def _add_pipeline_block(
    add, res: PipelineResult, match: MatchResult, hr: HRGridResult
) -> None:
    n_peaks = int(res.peaks_t.size)
    n_ibi = int(res.ibi_ms.size)
    n_kept = int(res.keep.sum()) if res.keep.size else 0
    rej_pct = 100.0 * (1.0 - n_kept / n_ibi) if n_ibi else float("nan")
    mean_hr = 60000.0 / float(np.mean(res.nn_ms)) if res.nn_ms.size else float("nan")
    mean_ibi = float(np.mean(res.nn_ms)) if res.nn_ms.size else float("nan")
    add(f"  n_peaks               : {n_peaks}")
    add(f"  n_ibis                : {n_ibi}")
    add(f"  n_kept (clean_ibi)    : {n_kept}")
    add(f"  beat rejection (%)    : {fmt_float(rej_pct, 2)}")
    add(f"  mean radar HR (bpm)   : {fmt_float(mean_hr, 2)}")
    add(f"  mean radar IBI (ms)   : {fmt_float(mean_ibi, 2)}")
    add(f"  matched pairs         : {match.n_matched} of {match.n_polar} Polar beats")
    add(f"  paired Pearson r      : {fmt_float(match.pearson_r, 4)}")
    add(f"  paired MAE (ms)       : {fmt_float(match.mae_ms, 2)}")
    add(f"  paired bias (ms)      : {fmt_float(match.bias_ms, 2)}")
    add(f"  1Hz HR Pearson r      : {fmt_float(hr.pearson_r, 4)}")
    add(f"  1Hz HR MAE (bpm)      : {fmt_float(hr.mae_bpm, 3)}")
    add(f"  1Hz HR RMSE (bpm)     : {fmt_float(hr.rmse_bpm, 3)}")
    add(f"  1Hz HR bias (bpm)     : {fmt_float(hr.bias_bpm, 3)}")
    add("")


def make_figure(
    polar_unix: np.ndarray,
    polar_ibi_ms: np.ndarray,
    phi_clean: np.ndarray,
    fs: float,
    session_start_unix: float,
    res_base: PipelineResult,
    res_vmd: PipelineResult,
    extra: VMDExtra,
    hr_base: HRGridResult,
    hr_vmd: HRGridResult,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(13, 20))

    # --- Panel 1: HR(t) overlay -----------------------------------------
    ax = axes[0]
    if polar_unix.size:
        p_hr = 60000.0 / np.maximum(polar_ibi_ms, 1e-6)
        ax.plot(
            polar_unix - session_start_unix,
            p_hr,
            color="green",
            lw=1.5,
            label="Polar H10 (truth)",
        )
    if res_base.peaks_unix.size >= 2 and res_base.ibi_ms.size:
        t_b = res_base.peaks_unix[1:] - session_start_unix
        k_b = res_base.keep
        hr_b = 60000.0 / np.maximum(res_base.ibi_ms, 1e-6)
        ax.plot(
            t_b[k_b],
            hr_b[k_b],
            color="red",
            lw=1.0,
            marker=".",
            alpha=0.75,
            label="Baseline (0.8-2.5 Hz bandpass)",
        )
    if res_vmd.peaks_unix.size >= 2 and res_vmd.ibi_ms.size:
        t_v = res_vmd.peaks_unix[1:] - session_start_unix
        k_v = res_vmd.keep
        hr_v = 60000.0 / np.maximum(res_vmd.ibi_ms, 1e-6)
        ax.plot(
            t_v[k_v],
            hr_v[k_v],
            color="blue",
            lw=1.0,
            marker=".",
            alpha=0.85,
            label=f"VMD+ACF (cardiac f0={extra.cardiac_f0_hz:.2f} Hz)",
        )
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("HR (bpm)")
    ax.set_title(
        f"Panel 1 | HR(t): Polar truth vs Baseline (red) vs VMD+ACF (blue) | "
        f"r_base={hr_base.pearson_r:.3f} r_vmd={hr_vmd.pearson_r:.3f}"
    )
    ax.set_ylim(40, 200)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 2: VMD modes (vertically offset, cardiac highlighted) ----
    ax = axes[1]
    K, N = extra.u.shape
    t_axis = np.arange(N) / fs
    # Normalize each mode for plot clarity.
    norm = np.max(np.abs(extra.u), axis=1)
    norm = np.where(norm > 0, norm, 1.0)
    offsets = np.arange(K) * 2.5
    palette = plt.get_cmap("tab10")
    for k in range(K):
        col = palette(k % 10)
        if k == extra.cardiac_idx:
            ax.plot(
                t_axis,
                extra.u[k] / norm[k] + offsets[k],
                color="crimson",
                lw=1.8,
                label=f"mode {k} (CARDIAC, f0={extra.omega_hz[k]:.2f} Hz)",
            )
        else:
            ax.plot(
                t_axis,
                extra.u[k] / norm[k] + offsets[k],
                color=col,
                lw=0.8,
                alpha=0.85,
                label=f"mode {k} (f0={extra.omega_hz[k]:.2f} Hz)",
            )
    ax.set_yticks(offsets, [f"u{k}" for k in range(K)])
    ax.set_xlabel("time since session start (s)")
    ax.set_title("Panel 2 | VMD modes (normalized, offset vertically) -- cardiac mode in crimson")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    # --- Panel 3: Welch spectrum of phi_clean + VMD omega lines ---------
    ax = axes[2]
    nperseg = int(min(len(phi_clean), max(8 * fs, 64)))
    if nperseg >= 32:
        fr, P = welch(phi_clean, fs=fs, nperseg=nperseg)
        ax.semilogy(fr, P, color="black", lw=1.0, label="Welch PSD(phi_clean)")
    for k, f0 in enumerate(extra.omega_hz):
        col = "crimson" if k == extra.cardiac_idx else palette(k % 10)
        ls = "-" if k == extra.cardiac_idx else "--"
        ax.axvline(
            f0,
            color=col,
            ls=ls,
            lw=1.8 if k == extra.cardiac_idx else 1.0,
            label=f"u{k}: {f0:.2f} Hz",
        )
    ax.set_xlim(0, min(fs / 2, 5.0))
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD (rad^2/Hz)")
    ax.set_title("Panel 3 | Welch PSD of cleaned phase + VMD-found center frequencies")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)

    # --- Panel 4: Autocorrelation of cardiac mode -----------------------
    ax = axes[3]
    lags = np.arange(extra.acf.size) / fs
    ax.plot(lags, extra.acf, color="navy", lw=1.0, label="ACF(cardiac mode)")
    # Mark detected peak.
    if 0 < extra.acf_lag_peak < extra.acf.size:
        ax.axvline(
            extra.acf_lag_peak / fs,
            color="crimson",
            ls="--",
            lw=1.5,
            label=(
                f"first major peak @ {extra.acf_lag_peak / fs:.3f} s "
                f"-> {extra.acf_bpm:.1f} BPM"
            ),
        )
    # Visualize the search range.
    ax.axvspan(60.0 / 180.0, 60.0 / 50.0, color="gold", alpha=0.10, label="50-180 BPM search window")
    ax.set_xlim(0, min(3.0, lags[-1] if lags.size else 3.0))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("lag (s)")
    ax.set_ylabel("autocorrelation (normalized)")
    ax.set_title("Panel 4 | Autocorrelation of cardiac mode (ACF BPM cross-check)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # --- Panel 5: VMD cardiac signal + detected beats -------------------
    ax = axes[4]
    cardiac = extra.u[extra.cardiac_idx]
    t_axis2 = np.arange(cardiac.size) / fs
    ax.plot(t_axis2, cardiac, color="steelblue", lw=0.8, label="cardiac mode")
    if res_vmd.kept_peak_idx.size:
        idx = res_vmd.kept_peak_idx
        ax.plot(t_axis2[idx], cardiac[idx], "ro", ms=3.5, label="kept peaks")
    # also show all peaks (faint) for sanity.
    all_idx = np.round(res_vmd.peaks_t * fs).astype(int)
    all_idx = all_idx[(all_idx >= 0) & (all_idx < cardiac.size)]
    if all_idx.size:
        ax.plot(
            t_axis2[all_idx],
            cardiac[all_idx],
            "x",
            color="orange",
            ms=4.0,
            alpha=0.6,
            label="all peaks",
        )
    ax.set_xlabel("time since session start (s)")
    ax.set_ylabel("amplitude (rad)")
    ax.set_title("Panel 5 | VMD cardiac mode with detected beats")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="VMD+ACF vs bandpass-baseline evaluation")
    ap.add_argument("--radar", default=RADAR_CSV, help="Radar session_*.csv path (per-frame 4-RX IQ)")
    ap.add_argument("--polar", default=POLAR_CSV, help="Polar H10 CSV (elapsed_s,epoch_s,hr_bpm,ibi_ms)")
    args = ap.parse_args()
    radar_path = args.radar
    polar_path = args.polar
    print(f"[input] radar = {radar_path}")
    print(f"[input] polar = {polar_path}")

    # --- Load radar ---
    meta, radar_df = parse_radar_csv(radar_path)
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
        radar_df[r].to_numpy(dtype=np.float64)
        + 1j * radar_df[i].to_numpy(dtype=np.float64)
        for r, i in iq_cols
    ])

    # --- Load Polar ---
    polar_df = load_polar(polar_path)
    polar_unix = polar_df["epoch_s"].to_numpy(dtype=np.float64)
    polar_ibi_ms = polar_df["ibi_ms"].to_numpy(dtype=np.float64)

    # --- Shared cleaned phase ---
    phi_clean = _clean_phase(z_rx, fs)

    # --- Baseline pipeline (0.8-2.5 Hz bandpass + find_peaks) ---
    res_base = run_baseline(phi_clean, fs, session_start_unix)

    # --- VMD + ACF pipeline ---
    res_vmd, extra = run_vmd(phi_clean, fs, session_start_unix)

    # --- Pre-prints (the task asks for these step-by-step prints) ---
    print(
        f"[radar] fs={fs:.2f} Hz  n_frames={n_frames}  "
        f"duration={n_frames / fs:.2f} s"
    )
    print(
        f"[polar] n_beats={polar_df.shape[0]}  "
        f"mean_HR={float(polar_df['hr_bpm'].mean()):.2f} bpm  "
        f"mean_IBI={float(polar_df['ibi_ms'].mean()):.2f} ms"
    )
    print(f"[vmdpy installed] {_HAS_VMDPY}")
    print(
        f"[baseline] n_peaks={int(res_base.peaks_t.size)}  "
        f"n_kept={int(res_base.keep.sum())}  "
        f"mean_HR={(60000.0/float(np.mean(res_base.nn_ms))) if res_base.nn_ms.size else float('nan'):.2f} bpm"
    )
    print(
        f"[vmd]      cardiac_idx={extra.cardiac_idx}  "
        f"cardiac_f0={extra.cardiac_f0_hz:.3f} Hz "
        f"({extra.cardiac_f0_hz * 60.0:.2f} BPM)  "
        f"acf_bpm={extra.acf_bpm:.2f}  "
        f"n_peaks={int(res_vmd.peaks_t.size)}  "
        f"n_kept={int(res_vmd.keep.sum())}  "
        f"mean_HR={(60000.0/float(np.mean(res_vmd.nn_ms))) if res_vmd.nn_ms.size else float('nan'):.2f} bpm"
    )
    print(
        "[vmd]      omega_hz=["
        + ", ".join(f"{f:.3f}" for f in extra.omega_hz)
        + "]"
    )

    # --- Match + score ---
    match_base = match_and_score(
        polar_unix, polar_ibi_ms,
        res_base.peaks_unix, res_base.ibi_ms, res_base.keep,
        window_s=MATCH_WINDOW_S,
    )
    match_vmd = match_and_score(
        polar_unix, polar_ibi_ms,
        res_vmd.peaks_unix, res_vmd.ibi_ms, res_vmd.keep,
        window_s=MATCH_WINDOW_S,
    )
    hr_base = hr_on_1hz_grid(
        polar_unix, polar_ibi_ms,
        res_base.peaks_unix, res_base.ibi_ms, res_base.keep,
    )
    hr_vmd = hr_on_1hz_grid(
        polar_unix, polar_ibi_ms,
        res_vmd.peaks_unix, res_vmd.ibi_ms, res_vmd.keep,
    )

    # --- Report ---
    summary = build_summary(
        polar_df, polar_unix,
        res_base, res_vmd, extra,
        match_base, match_vmd,
        hr_base, hr_vmd,
        fs=fs, n_frames=n_frames,
        session_start_unix=session_start_unix,
    )
    print(summary)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"[wrote] {SUMMARY_TXT}")

    make_figure(
        polar_unix, polar_ibi_ms, phi_clean,
        fs, session_start_unix,
        res_base, res_vmd, extra,
        hr_base, hr_vmd,
        out_path=FIGURE_PNG,
    )
    print(f"[wrote] {FIGURE_PNG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
