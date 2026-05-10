"""
Real-time heart rate / HRV pipeline for IWR1443 + DCA1000.

Standalone live capture + PyQt visualization. The DSP stages (DC removal,
phase extraction, detrend/despike, motion gating, bandpass, beat
detection, IBI cleanup) are imported from the project's vendored
``radar_analysis`` package so the live viewer and the offline pipeline
share **identical** algorithms — the BPM you see live is the BPM the
saved CSV will report.

Architecture:
    Background thread:  Radar.run_polling -> per-frame Range FFT,
                        publishes (t_unix, complex spectrum) into a
                        thread-safe SharedState. Recording append also
                        happens here (so frames aren't dropped if GUI is
                        slow).
    Main thread:        Qt event loop (app.exec()), QTimer pulls the
                        latest spectrum from SharedState every 50 ms
                        and refreshes the plots.

This separation is what makes the GUI stop hanging: recvfrom and FFT
never block the Qt event loop.

DSP pipeline (single-bin, post-range-FFT):
    DCA1000 UDP frame
        -> reshape (n_chirps, n_samples, n_rx)
        -> Range FFT (along samples axis)
        -> mean over chirps (denoise)
        -> select RX antenna
        -> auto-pick target gate (strongest reflector in 0.2-1.5 m)
        -> ring buffer of complex IQ at that gate
        ===  compute_beats() below — shared by live tick + save: ===
        -> remove_dc                     (TI vital-signs guide §2.3)
        -> np.angle + np.unwrap          ->  raw phase
        -> detrend_median (2 s window)   -> phi_detrended
        -> despike_hampel (MAD-based)    -> phi_clean
        -> motion_mask (energy threshold) -> motion_ok bool
        -> bandpass 0.8-4.0 Hz (Butter4) -> heartbeat
        -> find_peaks (prominence + parabolic refine) -> peak_times_s
        -> peaks_to_ibi_ms + clean_ibi   -> nn intervals
        -> SDNN / RMSSD / pNN50

Recording (Start / Stop / Sync buttons in toolbar):
    Start            -> begin buffering (frame_idx, t_unix, IQ, phase)
    Mark Sync Event  -> append a timestamp to the sync log (e.g. for
                        ECG alignment via hand-clap markers)
    Stop & Save      -> write three CSVs to ./recordings/:
                            session_<ts>.csv         per-frame IQ
                            session_<ts>_peaks.csv   detected beats + IBI
                                                     + kept flag + HRV
                                                     metrics in header
                            session_<ts>_sync.csv    sync event timestamps
                                                     + motion_flag column

Dependencies: this script reuses MakeyMakey's UDP listener
(`from src.radar import Radar`) AND the project's offline DSP package
(`from radar_analysis...` — audit condition C5).

To run, clone MakeyMakey alongside this repo, drop hrv_live.py + the
radar_analysis package into the MakeyMakey root, and run from there:

    git clone https://github.com/r-bt/MakeyMakey.git
    cp scripts/hrv_live.py MakeyMakey/
    cp -r radar_analysis MakeyMakey/                    # vendored DSP
    cp data/1443_mmwavestudio_config.lua MakeyMakey/    # or pass full path
    cd MakeyMakey
    python -m venv .venv
    .venv/Scripts/python.exe -m pip install numpy scipy pyqt6 pyqtgraph numba pyserial matplotlib
    .venv/Scripts/python.exe -u hrv_live.py --cfg ./1443_mmwavestudio_config.lua

See `docs/hardware_setup_notes.md` for the full first-run sequence
(static IP, mmWave Studio, firewall, killing the CLI recorder, etc.)
and `docs/realtime_pipeline.md` for the math behind each stage.

Usage:
    python hrv_live.py --cfg ./1443_mmwavestudio_config.lua
    python hrv_live.py --cfg ./1443_mmwavestudio_config.lua --gate 5 --rx 0
"""

import argparse
import csv
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
from scipy.fft import fft
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from src.radar import Radar
from radar_analysis.phase_processing import (
    remove_dc, extract_phase, detrend_median, despike_hampel, motion_mask
)
from radar_analysis.heartbeat_extractors import extract_heartbeat
from radar_analysis.beat_detection import (
    detect_beats, peaks_to_ibi_ms, clean_ibi
)
from radar_analysis.hrv_metrics import (
    mean_ibi_ms, mean_hr_bpm, sdnn_ms, rmssd_ms, pnn50
)


REC_DIR = "recordings"

# Minimum samples for filtfilt(order=4 bandpass) — see audit point 2:
#   len(a) = len(b) = 2*order+1 = 9 → 3*9 = 27.
_FILTFILT_MIN_SAMPLES = 27

# Auto-band defaults: search window for f0, narrow band half-width,
# and the safety cap relative to f0 (must stay < 2*f0 to exclude the
# 2nd cardiac harmonic — empirically 1.7*f0 is a comfortable margin).
#
# F0_SEARCH_BAND upper is capped at 2.0 Hz: the 2nd harmonic of any
# resting HR (≤120 BPM = 2.0 Hz fundamental) lands above this cap and
# cannot be confused for the fundamental. For exercise / high HR,
# pass --low/--high explicitly.
_F0_SEARCH_BAND = (0.8, 2.0)        # 48-120 BPM
_NARROW_HALF_HZ = 0.5               # ±0.5 Hz around f0
_NARROW_LOW_FLOOR = 0.7
_NARROW_HIGH_CAP = 2.5
_HARMONIC_SAFETY_RATIO = 1.7        # narrow_high <= 1.7 * f0
_AUTO_BAND_FALLBACK = (0.8, 2.0)    # used if Welch fails to find a peak


def _estimate_f0_welch(phi, fs, search_band=_F0_SEARCH_BAND):
    """Find heartbeat fundamental via Welch periodogram peak.

    Pre-bandpasses to ``search_band`` so respiration sidebands
    (0.1-0.5 Hz, often 4-8× stronger than heartbeat in raw phase)
    don't dominate. ``search_band`` upper is capped at 2.0 Hz by
    default — the 2nd harmonic of any resting HR (≤120 BPM)
    lands above 2.0 Hz and can't fool global argmax inside the band.

    Returns Hz or None if the buffer is too short / no peak in band.
    """
    from scipy.signal import welch
    nperseg = min(len(phi), max(int(8 * fs), 64))
    if nperseg < 32 or len(phi) < nperseg:
        return None
    try:
        coarse = extract_heartbeat(phi, fs=fs,
                                   low_hz=search_band[0],
                                   high_hz=search_band[1])
    except Exception:
        coarse = phi
    fr, P = welch(coarse, fs=fs, nperseg=nperseg)
    mask = (fr >= search_band[0]) & (fr <= search_band[1])
    if not mask.any():
        return None
    return float(fr[mask][np.argmax(P[mask])])


def _adaptive_band(phi, fs):
    """Estimate HR fundamental, return (band, f0_hz) — narrow band
    centered on f0 with safety margin from the 2nd harmonic.

    Falls back to ``_AUTO_BAND_FALLBACK`` if f0 cannot be estimated
    (e.g. buffer too short). The 1.7*f0 cap is the key insight from
    the gate-8 sweep: at ~67 BPM the 2nd harmonic at 1.95 Hz had 58%
    of the fundamental power and was being detected as a separate
    beat by find_peaks, doubling the apparent BPM. This cap keeps it
    out of the band for any HR.
    """
    f0 = _estimate_f0_welch(phi, fs)
    if f0 is None or f0 <= 0:
        return _AUTO_BAND_FALLBACK, None
    low = max(_NARROW_LOW_FLOOR, f0 - _NARROW_HALF_HZ)
    high = min(_NARROW_HIGH_CAP, f0 + _NARROW_HALF_HZ,
               _HARMONIC_SAFETY_RATIO * f0)
    if high <= low + 0.1:
        # degenerate, fall back
        return _AUTO_BAND_FALLBACK, f0
    return (low, high), f0


def compute_beats(z, fs, *, band=(0.8, 2.0), detrend_window_s=2.0,
                   max_bpm=200.0, prominence_factor=0.5,
                   adaptive_band=False):
    """Run the radar_analysis pipeline on a complex IQ stream.

    Single source of truth for both the live preview and the Stop & Save
    post-processor. Audit condition C1: live BPM == saved BPM whenever
    this function is called with the same parameters.

    Parameters
    ----------
    z : (F,) or (F, R) complex ndarray
        Complex IQ at the chosen target gate, one sample per frame.
        - 1-D: single-RX legacy path. Direct ``remove_dc`` over time
          then atan2+unwrap.
        - 2-D: multi-RX coherent integration (Option C from the
          implementation audit). Each RX gets its own DC removal
          (per-time mean) and its own atan2+unwrap; the resulting R
          unwrapped phase streams are individually detrended (subtract
          per-stream mean) then averaged. This is robust to the inter-
          RX phase offset on the IWR1443's λ/2-spaced array — the
          alternative complex-domain averaging would partially cancel
          off-boresight targets.
    fs : float
        Slow-time sample rate (frames per second).
    band : (low_hz, high_hz)
        Heartbeat bandpass. Default ``(0.8, 2.0)`` covers 48-120 BPM
        and — critically — excludes the 2nd cardiac harmonic for any
        HR ≤ 120 BPM (the 2nd harmonic of 2.0 Hz fundamental lands at
        4.0 Hz, well above this cap). The wider 0.8-4.0 default we
        used previously let the 2nd harmonic in and find_peaks
        double-detected beats (gate-8 backtest: BPM 113 vs truth 67;
        narrowing to 0.8-2.0 brought it down to BPM 77).

        For exercise / high HR (>120 BPM), pass band=(0.8, 3.0) or
        wider explicitly.
    adaptive_band : bool
        If True, run a Welch periodogram first to find f0 and narrow
        ``band`` to ``[max(0.7, f0-0.5), min(band[1], f0+0.5, 1.7*f0)]``.
        Empirically unreliable on low-SNR recordings (the f0 estimator
        locks onto noise sidebands), so default False. Off by default;
        opt in for clean signals.
    detrend_window_s : float
        Sliding-median detrend window (seconds). 2 s passes respiration
        but suppresses anything slower than 0.5 Hz. The HR low-cutoff
        at 0.8 Hz already protects the heartbeat band.

    Returns
    -------
    dict with keys:
        phi_clean       : (F,) detrended + despiked phase, radians
        heartbeat       : (F,) bandpassed phase
        motion_ok       : (F,) bool — True where motion energy is low
        peak_times_s    : (P,) sub-sample peak times, seconds
        ibi_ms          : (P-1,) successive intervals, ms
        kept_mask       : (P-1,) bool — True for physiologic + outlier-
                          rejected intervals
        nn_ms           : (K,) the kept intervals only
        hrv             : dict with mean_ibi_ms / mean_hr_bpm / sdnn_ms
                          / rmssd_ms / pnn50_pct (NaN if <2 NN)
        n_rx_used       : int — number of RX antennas integrated
        band_used       : (low_hz, high_hz) — actual bandpass applied
        f0_estimated_hz : float or None — Welch f0 if adaptive
    """
    z = np.asarray(z, dtype=np.complex128)
    n_frames = z.shape[0]
    n_rx_used = z.shape[1] if z.ndim == 2 else 1

    if n_frames < _FILTFILT_MIN_SAMPLES:
        return {
            "phi_clean": np.zeros(n_frames, dtype=np.float64),
            "heartbeat": np.zeros(n_frames, dtype=np.float64),
            "motion_ok": np.ones(n_frames, dtype=bool),
            "peak_times_s": np.array([], dtype=np.float64),
            "ibi_ms": np.array([], dtype=np.float64),
            "kept_mask": np.array([], dtype=bool),
            "nn_ms": np.array([], dtype=np.float64),
            "hrv": _empty_hrv(),
            "n_too_short": True,
            "n_rx_used": n_rx_used,
            "band_used": tuple(band) if band is not None else _AUTO_BAND_FALLBACK,
            "f0_estimated_hz": None,
        }

    if z.ndim == 1:
        # Single-RX legacy path
        z_centered = remove_dc(z)               # remove_dc operates on axis=-1
        phi_raw = extract_phase(z_centered)
    else:
        # Multi-RX Option C: per-RX DC removal over TIME (axis=0), per-RX
        # atan2+unwrap, then average the R unwrapped phase streams.
        # Note: remove_dc has axis=-1 hard-coded, so for shape (F, R) we
        # need axis=0. Do it explicitly.
        z_centered = z - z.mean(axis=0, keepdims=True)             # (F, R)
        phi_per_rx = np.unwrap(np.angle(z_centered), axis=0)        # (F, R)
        # Each RX's unwrap can land on a different ±2π plateau; remove
        # per-RX mean before averaging so the constant offsets don't
        # bias the result.
        phi_per_rx = phi_per_rx - phi_per_rx.mean(axis=0, keepdims=True)
        phi_raw = phi_per_rx.mean(axis=1)                           # (F,)

    phi_detrended = detrend_median(phi_raw, fs=fs, window_s=detrend_window_s)
    phi_clean = despike_hampel(phi_detrended)
    motion_ok = motion_mask(phi_clean, fs=fs)

    # Bandpass selection. Three modes:
    #   1) band is None        — full adaptive: call _adaptive_band(phi, fs)
    #                            to pick a narrow band centered on Welch f0.
    #   2) band tuple + adaptive_band=True — narrow the explicit band around
    #                            the Welch-estimated f0 (legacy hybrid path).
    #   3) band tuple + adaptive_band=False — use the explicit band as-is.
    if band is None:
        band_resolved, f0_estimated_hz = _adaptive_band(phi_clean, fs)
    else:
        band_resolved = tuple(band)
        f0_estimated_hz = _estimate_f0_welch(phi_clean, fs) if adaptive_band else None
        if adaptive_band and f0_estimated_hz is not None and f0_estimated_hz > 0:
            narrow_low = max(_NARROW_LOW_FLOOR, f0_estimated_hz - _NARROW_HALF_HZ)
            narrow_high = min(band[1], f0_estimated_hz + _NARROW_HALF_HZ,
                              _HARMONIC_SAFETY_RATIO * f0_estimated_hz)
            if narrow_high > narrow_low + 0.1:
                band_resolved = (narrow_low, narrow_high)

    heartbeat = extract_heartbeat(phi_clean, fs=fs,
                                  low_hz=band_resolved[0],
                                  high_hz=band_resolved[1])
    peak_times_s = detect_beats(heartbeat, fs=fs, max_bpm=max_bpm,
                                prominence_factor=prominence_factor,
                                refine=True)
    ibi_ms = peaks_to_ibi_ms(peak_times_s)
    kept_mask = clean_ibi(ibi_ms)
    nn_ms = ibi_ms[kept_mask] if kept_mask.size else np.array([])

    if nn_ms.size >= 2:
        hrv = {
            "mean_ibi_ms": mean_ibi_ms(nn_ms),
            "mean_hr_bpm": mean_hr_bpm(nn_ms),
            "sdnn_ms": sdnn_ms(nn_ms),
            "rmssd_ms": rmssd_ms(nn_ms),
            "pnn50_pct": pnn50(nn_ms),
        }
    else:
        hrv = _empty_hrv()

    return {
        "phi_clean": phi_clean,
        "heartbeat": heartbeat,
        "motion_ok": motion_ok,
        "peak_times_s": peak_times_s,
        "ibi_ms": ibi_ms,
        "kept_mask": kept_mask,
        "nn_ms": nn_ms,
        "hrv": hrv,
        "n_too_short": False,
        "n_rx_used": n_rx_used,
        "band_used": band_resolved,
        "f0_estimated_hz": f0_estimated_hz,
    }


def _empty_hrv():
    nan = float("nan")
    return {"mean_ibi_ms": nan, "mean_hr_bpm": nan,
            "sdnn_ms": nan, "rmssd_ms": nan, "pnn50_pct": nan}


# =============================================================================
#                              SHARED STATE
# =============================================================================
class SharedState:
    """Lock-protected handoff between radar thread and Qt main thread."""

    def __init__(self, n_bins):
        self.lock = threading.Lock()

        # Latest spectrum (overwritten each frame)
        self.latest_t_unix = None
        self.latest_spec = None  # complex, shape (n_bins,)

        # Init / gate
        self.target_gate = None        # set by user or auto-detect
        self.init_mags = []            # accumulator for auto-detect
        self.init_done = False

        # Recording (worker-side; main thread snapshots on Stop)
        self.recording = False
        self.rec_iq = []
        self.rec_t_unix = []
        self.rec_sync = []
        self.rec_start_unix = None

        # Worker shutdown
        self.shutdown = False

    # ---- worker side (radar thread) ----
    def publish_frame(self, t_unix, spec):
        """spec is (n_bins,) complex for single-RX or (n_bins, n_rx) for multi-RX."""
        with self.lock:
            self.latest_t_unix = t_unix
            self.latest_spec = spec
            if self.recording and self.target_gate is not None:
                if spec.ndim == 1:
                    iq_at_gate = complex(spec[self.target_gate])
                else:
                    # (n_rx,) complex array — keep all RX for downstream Option C
                    iq_at_gate = np.asarray(spec[self.target_gate, :],
                                             dtype=np.complex128)
                self.rec_iq.append(iq_at_gate)
                self.rec_t_unix.append(t_unix)

    def push_init_mag(self, mag):
        with self.lock:
            if self.target_gate is None:
                self.init_mags.append(mag)

    def maybe_set_gate_auto(self, threshold_n, range_res, n_bins):
        with self.lock:
            if self.target_gate is not None or len(self.init_mags) < threshold_n:
                return None
            avg = np.mean(np.stack(self.init_mags), axis=0)
            lo = max(2, int(0.2 / range_res))
            hi = min(n_bins - 1, int(1.5 / range_res))
            gate = lo + int(np.argmax(avg[lo:hi]))
            self.target_gate = gate
            self.init_done = True
            self.init_mags = []  # free memory
            return gate

    def set_gate_manual(self, gate):
        """Override the target gate from the GUI. Locks init as done."""
        with self.lock:
            self.target_gate = int(gate)
            self.init_done = True
            self.init_mags = []

    # ---- main thread side ----
    def get_latest(self):
        with self.lock:
            return self.latest_t_unix, self.latest_spec

    def start_recording(self):
        with self.lock:
            if self.target_gate is None:
                return False
            self.recording = True
            self.rec_iq = []
            self.rec_t_unix = []
            self.rec_sync = []
            self.rec_start_unix = time.time()
            return True

    def stop_recording_snapshot(self):
        with self.lock:
            self.recording = False
            return {
                "iq": list(self.rec_iq),
                "t_unix": list(self.rec_t_unix),
                "sync": list(self.rec_sync),
                "start_unix": self.rec_start_unix,
                "gate": self.target_gate,
            }

    def add_sync(self, t_unix, label):
        with self.lock:
            if not self.recording:
                return False
            self.rec_sync.append((t_unix, label))
            return True

    def recording_stats(self):
        with self.lock:
            n = len(self.rec_iq)
            n_sync = len(self.rec_sync)
            start = self.rec_start_unix
            recording = self.recording
        return recording, n, n_sync, start


# =============================================================================
#                          RADAR WORKER (BACKGROUND)
# =============================================================================
def radar_worker(radar, args, n_bins, shared: SharedState):
    """Runs in a daemon thread. Polls UDP, computes Range-FFT, publishes.

    args.rx == -1 means coherent multi-RX mode (Option C from the audit):
    keep all 4 RX, do per-RX phase extraction downstream in compute_beats.
    args.rx in 0..3 means single-RX (legacy / debug).
    """
    multi_rx = (args.rx == -1)

    def on_frame(msg):
        if shared.shutdown:
            raise KeyboardInterrupt
        frame = msg.get("data")
        if frame is None:
            return
        try:
            t_unix = time.time()
            if multi_rx:
                sig = frame                                         # (n_chirps, n_samples, n_rx)
                sig = sig - sig.mean(axis=1, keepdims=True)         # per-RX, per-chirp DC removal
                rfft = fft(sig, axis=1)[:, :n_bins, :]              # (n_chirps, n_bins, n_rx)
                spec = rfft.mean(axis=0)                            # (n_bins, n_rx) complex
                # |.| collapsed over RX for gate selection / display
                mag_for_init = np.abs(spec).mean(axis=-1)
            else:
                sig = frame[:, :, args.rx]                          # (n_chirps, n_samples)
                sig = sig - sig.mean(axis=1, keepdims=True)
                rfft = fft(sig, axis=1)[:, :n_bins]
                spec = rfft.mean(axis=0)                            # (n_bins,) complex
                mag_for_init = np.abs(spec)
            shared.publish_frame(t_unix, spec)
            if shared.target_gate is None:
                shared.push_init_mag(mag_for_init)
        except Exception as e:
            print(f"[worker] frame error: {e}")

    try:
        radar.run_polling(cb=on_frame)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[worker] polling stopped: {e}")
    finally:
        try:
            radar.close()
        except Exception:
            pass


# =============================================================================
#                          POST-PROCESS / SAVE
# =============================================================================
def post_process_and_save(rec, args, fs, range_res):
    """Run the full pipeline on the recorded IQ and write 3 CSV files."""
    os.makedirs(REC_DIR, exist_ok=True)
    ts = datetime.fromtimestamp(rec["start_unix"]).strftime("%Y-%m-%d_%H-%M-%S")
    base = os.path.join(REC_DIR, f"session_{ts}")

    iq = np.asarray(rec["iq"], dtype=np.complex128)
    t_unix = np.asarray(rec["t_unix"], dtype=np.float64)
    t_rel = t_unix - rec["start_unix"] if len(t_unix) else np.array([])

    # iq is shape (F,) for single-RX or (F, n_rx) for multi-RX (Option C).
    n_rx_in_csv = iq.shape[1] if iq.ndim == 2 else 1

    # ---- main per-frame CSV ----
    main_path = base + ".csv"
    with open(main_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# version: 2\n")
        f.write(f"# fs_hz: {fs:.4f}\n")
        f.write(f"# range_res_m: {range_res:.6f}\n")
        f.write(f"# target_gate: {rec['gate']}\n")
        gate = rec["gate"] if rec["gate"] is not None else 0
        f.write(f"# target_distance_m: {gate * range_res:.4f}\n")
        rx_label = "all_combined" if args.rx == -1 else str(args.rx)
        f.write(f"# rx_antenna: {rx_label}\n")
        f.write(f"# n_rx_recorded: {n_rx_in_csv}\n")
        f.write(f"# session_start_unix: {rec['start_unix']:.6f}\n")
        f.write(f"# session_start_iso: "
                f"{datetime.fromtimestamp(rec['start_unix']).isoformat()}\n")
        f.write(f"# n_frames: {len(iq)}\n")
        f.write(f"# duration_s: {t_rel[-1] if len(t_rel) else 0:.3f}\n")
        w = csv.writer(f)
        if n_rx_in_csv == 1:
            w.writerow(["frame_idx", "t_unix", "t_iso_local", "t_rel_s",
                        "iq_real", "iq_imag"])
            for i, (tu, tr, c) in enumerate(zip(t_unix, t_rel, iq)):
                iso = datetime.fromtimestamp(tu).isoformat(timespec="milliseconds")
                w.writerow([i, f"{tu:.6f}", iso, f"{tr:.6f}",
                            f"{c.real:.6f}", f"{c.imag:.6f}"])
        else:
            # Multi-RX: 2 columns per RX
            cols = ["frame_idx", "t_unix", "t_iso_local", "t_rel_s"]
            for r in range(n_rx_in_csv):
                cols += [f"iq{r}_real", f"iq{r}_imag"]
            w.writerow(cols)
            for i in range(len(iq)):
                tu = t_unix[i]; tr = t_rel[i]
                iso = datetime.fromtimestamp(tu).isoformat(timespec="milliseconds")
                row = [i, f"{tu:.6f}", iso, f"{tr:.6f}"]
                for r in range(n_rx_in_csv):
                    c = iq[i, r]
                    row += [f"{c.real:.6f}", f"{c.imag:.6f}"]
                w.writerow(row)

    # ---- run radar_analysis pipeline ----
    pipe = compute_beats(iq, fs, band=None if args.auto else (args.low, args.high),
                         adaptive_band=args.adaptive_band)
    peak_times_s = pipe["peak_times_s"]
    ibi_ms = pipe["ibi_ms"]
    kept_mask = pipe["kept_mask"]
    motion_ok = pipe["motion_ok"]
    hrv = pipe["hrv"]
    n_peaks = int(peak_times_s.size)
    n_kept = int(kept_mask.sum()) if kept_mask.size else 0

    # ---- peaks / IBI CSV (audit C2: derive t_unix from start + t_rel_s; C3: kept column) ----
    start_unix = float(rec["start_unix"])
    peaks_path = base + "_peaks.csv"
    with open(peaks_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# session_start_unix: {start_unix:.6f}\n")
        bp_used = pipe.get("band_used", (args.low, args.high))
        f0_used = pipe.get("f0_estimated_hz")
        bp_mode = "adaptive" if args.auto else "manual"
        f.write(f"# bandpass_hz: {bp_used[0]:.3f}-{bp_used[1]:.3f}\n")
        f.write(f"# bandpass_mode: {bp_mode}\n")
        if f0_used is not None:
            f.write(f"# f0_welch_hz: {f0_used:.4f}\n")
            f.write(f"# f0_welch_bpm: {f0_used*60:.2f}\n")
        else:
            f.write(f"# f0_welch_hz: nan\n")
        f.write(f"# detect_max_bpm: 200\n")
        f.write(f"# detect_prominence_factor: 0.5\n")
        f.write(f"# n_peaks: {n_peaks}\n")
        f.write(f"# n_intervals: {ibi_ms.size}\n")
        f.write(f"# n_kept: {n_kept}\n")
        f.write(f"# n_rejected: {ibi_ms.size - n_kept}\n")
        for k in ("mean_ibi_ms", "mean_hr_bpm", "sdnn_ms",
                  "rmssd_ms", "pnn50_pct"):
            v = hrv.get(k, float("nan"))
            f.write(f"# {k}: {v:.4f}\n" if not np.isnan(v)
                    else f"# {k}: nan\n")
        if pipe.get("n_too_short"):
            f.write(f"# warning: recording shorter than "
                    f"{_FILTFILT_MIN_SAMPLES} samples; pipeline skipped.\n")
        w = csv.writer(f)
        w.writerow(["peak_idx", "frame_idx", "t_rel_s", "t_unix",
                    "t_iso_local", "ibi_ms", "kept"])
        for pi, t_rel_s in enumerate(peak_times_s):
            # audit C2: frame_idx is rounded; t_unix is start + t_rel_s
            frame_idx = int(round(float(t_rel_s) * fs))
            tu = start_unix + float(t_rel_s)
            iso = datetime.fromtimestamp(tu).isoformat(timespec="milliseconds")
            if pi == 0:
                ibi = ""
                kept = ""
            else:
                ibi = f"{ibi_ms[pi - 1]:.2f}"
                kept = "1" if kept_mask[pi - 1] else "0"
            w.writerow([pi, frame_idx, f"{float(t_rel_s):.6f}",
                        f"{tu:.6f}", iso, ibi, kept])

    # ---- sync events CSV (with motion_flag column per audit point 9) ----
    sync_path = base + "_sync.csv"
    with open(sync_path, "w", newline="", encoding="utf-8") as f:
        f.write(f"# session_start_unix: {start_unix:.6f}\n")
        f.write(f"# n_events: {len(rec['sync'])}\n")
        w = csv.writer(f)
        w.writerow(["event_idx", "t_rel_s", "t_unix", "t_iso_local",
                    "label", "motion_flag"])
        for i, (tu, lbl) in enumerate(rec["sync"]):
            t_rel_s = tu - start_unix
            # nearest frame index (clamped)
            fidx = int(np.clip(round(t_rel_s * fs), 0,
                               max(0, motion_ok.size - 1)))
            mflag = 0 if (motion_ok.size and motion_ok[fidx]) else 1
            iso = datetime.fromtimestamp(tu).isoformat(timespec="milliseconds")
            w.writerow([i, f"{t_rel_s:.6f}", f"{tu:.6f}", iso,
                        lbl, mflag])

    summary = {
        "main": main_path,
        "peaks": peaks_path,
        "sync": sync_path,
        "n_frames": len(iq),
        "n_peaks": n_peaks,
        "n_kept": n_kept,
        "duration_s": float(t_rel[-1]) if len(t_rel) else 0.0,
        "mean_bpm": hrv.get("mean_hr_bpm"),
        "sdnn_ms": hrv.get("sdnn_ms"),
        "rmssd_ms": hrv.get("rmssd_ms"),
    }
    return summary


# =============================================================================
#                                   GUI
# =============================================================================
def build_window(args, fs, n_bins, range_res, shared: SharedState):
    win = QtWidgets.QMainWindow()
    win.setWindowTitle("HRV Live (IWR1443)")
    win.resize(1200, 900)

    # ---- toolbar ----
    tb = QtWidgets.QToolBar("Recording")
    tb.setMovable(False)
    tb.setStyleSheet("QToolBar { padding: 6px; spacing: 8px; }"
                     "QToolButton { padding: 6px 14px; font-size: 12pt; }")
    win.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, tb)

    act_start = QtGui.QAction("▶  Start Recording", win)
    act_stop = QtGui.QAction("■  Stop && Save", win)
    act_sync = QtGui.QAction("⚑  Mark Sync Event", win)
    tb.addAction(act_start)
    tb.addAction(act_stop)
    tb.addAction(act_sync)

    act_stop.setEnabled(False)
    act_sync.setEnabled(False)

    tb.addSeparator()
    lbl_gate = QtWidgets.QLabel("  Gate (bin): ")
    lbl_gate.setStyleSheet("font-size: 11pt;")
    tb.addWidget(lbl_gate)
    spin_gate = QtWidgets.QSpinBox()
    spin_gate.setRange(2, n_bins - 1)
    spin_gate.setValue(args.gate if args.gate is not None else 2)
    spin_gate.setFixedWidth(70)
    spin_gate.setStyleSheet("font-size: 11pt; padding: 2px 4px;")
    tb.addWidget(spin_gate)
    lbl_gate_dist = QtWidgets.QLabel(f"  ({(args.gate or 2) * range_res:.2f} m)  ")
    lbl_gate_dist.setStyleSheet("font-size: 11pt; color: #ffaa00;")
    tb.addWidget(lbl_gate_dist)

    tb.addSeparator()
    lbl_status = QtWidgets.QLabel("  Idle  ")
    lbl_status.setStyleSheet("font-size: 11pt; padding: 0 12px;")
    tb.addWidget(lbl_status)

    # ---- plots ----
    gl = pg.GraphicsLayoutWidget()
    win.setCentralWidget(gl)

    p1 = gl.addPlot(title="Range FFT |magnitude|  -- current frame")
    p1.setLabel("bottom", "Distance (m)")
    p1.setLabel("left", "Intensity")
    p1_curve = p1.plot(pen="c")
    p1_gate_line = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen("r", width=2))
    p1.addItem(p1_gate_line)
    p1.setXRange(0, 3.0)

    gl.nextRow()
    p2 = gl.addPlot(title="Raw unwrapped phase @ target gate")
    p2.setLabel("bottom", "Time (s)")
    p2.setLabel("left", "Phase (rad)")
    p2_curve = p2.plot(pen="y")

    gl.nextRow()
    _title_band = "AUTO (per-tick adaptive)" if args.auto else f"{args.low}-{args.high} Hz"
    p3 = gl.addPlot(title=f"Bandpass {_title_band} (heartbeat)")
    p3.setLabel("bottom", "Time (s)")
    p3.setLabel("left", "Amplitude")
    p3_curve = p3.plot(pen="m")
    p3_peaks = pg.ScatterPlotItem(symbol="o", size=10,
                                  brush=pg.mkBrush("r"), pen=None)
    p3.addItem(p3_peaks)
    bpm_text = pg.TextItem("BPM: --", color="w", anchor=(0, 1))
    p3.addItem(bpm_text)

    distances = np.arange(n_bins) * range_res
    buf_size = int(args.buffer_sec * fs)
    iq_buf = deque(maxlen=buf_size)
    last_seen_t = [None]
    sync_lines_p2 = []
    motion_regions = []  # red-shaded LinearRegionItems for motion overlay
    INIT_FRAMES = max(50, int(0.5 * fs))

    # ---- button handlers ----
    def update_status_label():
        recording, n, n_sync, start = shared.recording_stats()
        if not recording:
            lbl_status.setText("  Idle  ")
            lbl_status.setStyleSheet("font-size: 11pt; padding: 0 12px;")
            return
        dur = (time.time() - start) if start else 0
        lbl_status.setText(f"  ● REC   {dur:6.1f}s   {n} frames   "
                           f"sync events: {n_sync}  ")
        lbl_status.setStyleSheet(
            "font-size: 11pt; padding: 0 12px; color: #ff5050; font-weight: bold;")

    def on_start():
        if not shared.start_recording():
            QtWidgets.QMessageBox.warning(
                win, "Wait",
                "Target gate not selected yet. Sit still and wait ~0.5s "
                "for auto-detect, then click Start again.")
            return
        act_start.setEnabled(False)
        act_stop.setEnabled(True)
        act_sync.setEnabled(True)
        update_status_label()
        gate = shared.target_gate
        print(f"[REC] Started recording (gate={gate}, "
              f"{gate*range_res:.2f} m)")

    def on_stop():
        rec = shared.stop_recording_snapshot()
        if rec["start_unix"] is None:
            return
        act_start.setEnabled(True)
        act_stop.setEnabled(False)
        act_sync.setEnabled(False)
        try:
            summary = post_process_and_save(rec, args, fs, range_res)
            mean_bpm = summary.get("mean_bpm")
            bpm_str = f"{mean_bpm:.1f}" if mean_bpm else "--"
            QtWidgets.QMessageBox.information(
                win, "Saved",
                f"Saved to {REC_DIR}/\n\n"
                f"Frames: {summary['n_frames']}\n"
                f"Duration: {summary['duration_s']:.1f}s\n"
                f"Detected peaks: {summary['n_peaks']}\n"
                f"Mean BPM: {bpm_str}\n\n"
                f"Files:\n"
                f"  {os.path.basename(summary['main'])}\n"
                f"  {os.path.basename(summary['peaks'])}\n"
                f"  {os.path.basename(summary['sync'])}")
            print(f"[REC] Saved {summary['n_frames']} frames, "
                  f"{summary['n_peaks']} peaks, mean BPM {bpm_str}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(win, "Save error", str(e))
            print(f"[REC] Save error: {e}")
        update_status_label()

    def on_sync():
        t = time.time()
        label = "sync"
        if shared.add_sync(t, label):
            update_status_label()
            print(f"[REC] Sync event @ {datetime.fromtimestamp(t).isoformat(timespec='milliseconds')}")

    act_start.triggered.connect(on_start)
    act_stop.triggered.connect(on_stop)
    act_sync.triggered.connect(on_sync)

    def on_gate_changed(v):
        # Don't allow gate changes while recording (would corrupt the IQ trace)
        if shared.recording:
            spin_gate.blockSignals(True)
            spin_gate.setValue(shared.target_gate or 2)
            spin_gate.blockSignals(False)
            QtWidgets.QMessageBox.warning(
                win, "Stop recording first",
                "Stop the current recording before changing the gate. "
                "Changing gate mid-recording would mix IQ from two ranges.")
            return
        shared.set_gate_manual(v)
        iq_buf.clear()        # discard stale IQ from previous gate
        sync_lines_p2.clear()  # markers reference old timestamps
        lbl_gate_dist.setText(f"  ({v * range_res:.2f} m)  ")
        p1_gate_line.setPos(v * range_res)  # move red line immediately, don't wait for next frame
        print(f"[INFO] Gate manually set to bin {v} ({v * range_res:.2f} m)")

    spin_gate.valueChanged.connect(on_gate_changed)

    # ---- GUI tick (called by QTimer at 20 Hz) ----
    tick_state = {"counter": 0}

    def tick():
        # Auto-gate detection
        if shared.target_gate is None:
            new_gate = shared.maybe_set_gate_auto(INIT_FRAMES, range_res, n_bins)
            if new_gate is not None:
                print(f"[INFO] Target gate auto-selected: bin {new_gate} "
                      f"({new_gate*range_res:.2f} m)")
                # Reflect auto-selected gate in the spinbox without retriggering
                spin_gate.blockSignals(True)
                spin_gate.setValue(new_gate)
                spin_gate.blockSignals(False)
                lbl_gate_dist.setText(f"  ({new_gate * range_res:.2f} m)  ")

        t_unix, spec = shared.get_latest()
        if spec is None:
            return
        if last_seen_t[0] == t_unix:
            return  # no new frame since last tick
        last_seen_t[0] = t_unix

        # spec: (n_bins,) for single-RX, (n_bins, n_rx) for multi-RX (Option C).
        # For the Range-FFT plot we collapse magnitude over RX.
        if spec.ndim == 2:
            mag = np.abs(spec).mean(axis=-1)
        else:
            mag = np.abs(spec)
        p1_curve.setData(distances, mag)

        gate = shared.target_gate
        if gate is None:
            return  # still calibrating

        p1_gate_line.setPos(gate * range_res)
        if spec.ndim == 2:
            # keep all RX so compute_beats can do per-RX phase extraction
            iq_buf.append(np.asarray(spec[gate, :], dtype=np.complex128))
        else:
            iq_buf.append(complex(spec[gate]))

        tick_state["counter"] += 1
        if tick_state["counter"] % 2 != 0:   # heavy panels update at 10 Hz (was 4 Hz)
            update_status_label()
            return

        if len(iq_buf) < int(2 * fs):
            update_status_label()
            return

        # Audit C1: live preview shares the exact same DSP as the saved CSV.
        # All filtering / detection happens inside compute_beats() so the
        # BPM number you see live is what the next Stop & Save will report.
        iq = np.asarray(iq_buf)
        band_arg = None if args.auto else (args.low, args.high)
        pipe = compute_beats(iq, fs, band=band_arg)
        phi_clean = pipe["phi_clean"]
        hb = pipe["heartbeat"]
        peak_times_s = pipe["peak_times_s"]
        kept_mask = pipe["kept_mask"]
        nn_ms = pipe["nn_ms"]
        motion_ok = pipe["motion_ok"]

        t_axis = np.arange(len(phi_clean)) / fs
        p2_curve.setData(t_axis, phi_clean)

        # Sync markers in live window
        for ln in sync_lines_p2:
            p2.removeItem(ln)
        sync_lines_p2.clear()
        recording, _, _, _ = shared.recording_stats()
        if recording:
            with shared.lock:
                sync_events = list(shared.rec_sync)
            now = time.time()
            window_start = now - len(iq) / fs
            for tu, _ in sync_events:
                if window_start <= tu <= now:
                    x = tu - window_start
                    ln = pg.InfiniteLine(
                        pos=x, angle=90,
                        pen=pg.mkPen("g", width=1,
                                     style=QtCore.Qt.PenStyle.DashLine))
                    p2.addItem(ln)
                    sync_lines_p2.append(ln)

        # Audit C4: motion mask is *displayed* on the phase plot but NEVER
        # gates the signal — bandpass and detection always run on the
        # full continuous buffer.
        for region in motion_regions:
            p2.removeItem(region)
        motion_regions.clear()
        if motion_ok.size and not motion_ok.all():
            # Find contiguous "motion bad" runs and shade them red.
            bad = ~motion_ok
            d = np.diff(bad.astype(np.int8), prepend=0, append=0)
            starts = np.where(d == 1)[0]
            ends = np.where(d == -1)[0]
            for s, e in zip(starts, ends):
                if e - s < 2:  # ignore single-sample blips
                    continue
                region = pg.LinearRegionItem(
                    values=(t_axis[s], t_axis[min(e, len(t_axis) - 1)]),
                    brush=pg.mkBrush(255, 80, 80, 40),
                    movable=False,
                )
                p2.addItem(region)
                motion_regions.append(region)

        p3_curve.setData(t_axis, hb)

        if peak_times_s.size:
            # Show all detected peaks; color-code kept (red) vs rejected (gray).
            # peak_times_s are sub-sample seconds; map to the live x-axis
            # which is also seconds-from-start.
            peak_x = peak_times_s
            # peak amplitudes via nearest-sample lookup on hb
            peak_idx_int = np.clip(np.round(peak_times_s * fs).astype(int),
                                   0, len(hb) - 1)
            peak_y = hb[peak_idx_int]
            # kept_mask covers intervals (size = npeaks-1). Treat the first
            # peak as kept by default, rest follow the interval flag.
            keep_pt = np.ones(peak_times_s.size, dtype=bool)
            if kept_mask.size:
                keep_pt[1:] = kept_mask
            brushes = [pg.mkBrush("r") if k else pg.mkBrush(120, 120, 120)
                       for k in keep_pt]
            p3_peaks.setData(peak_x, peak_y, brush=brushes)
        else:
            p3_peaks.setData([], [])

        hrv = pipe["hrv"]
        bp = pipe.get("band_used", (0, 0))
        f0 = pipe.get("f0_estimated_hz")
        f0_str = f"f0: {f0:.2f}Hz ({f0*60:.0f}bpm)" if f0 else "f0: --"
        bp_str = f"band: {bp[0]:.2f}-{bp[1]:.2f}Hz"
        if not np.isnan(hrv["mean_hr_bpm"]):
            bpm_text.setText(
                f"BPM: {hrv['mean_hr_bpm']:5.1f}   "
                f"peaks: {peak_times_s.size}   "
                f"kept: {nn_ms.size}   "
                f"SDNN: {hrv['sdnn_ms']:.0f}ms   "
                f"RMSSD: {hrv['rmssd_ms']:.0f}ms\n"
                f"{f0_str}   {bp_str}")
        else:
            bpm_text.setText(
                f"BPM: --   peaks: {peak_times_s.size}   "
                f"kept: {nn_ms.size}\n{f0_str}   {bp_str}")
        bpm_text.setPos(t_axis[0], float(np.max(hb)) if len(hb) else 0.0)

        update_status_label()

    return win, tick


# =============================================================================
#                                   MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--gate", type=int, default=None,
                    help="Range bin to use as target. Default: auto-detect.")
    def _parse_rx(s):
        if s.lower() in ("all", "combined", "-1"):
            return -1
        n = int(s)
        if not 0 <= n <= 3:
            raise argparse.ArgumentTypeError(
                f"--rx must be 'all' or 0-3, got {n}")
        return n
    ap.add_argument("--rx", type=_parse_rx, default=-1,
                    help="RX antenna: 'all' (default, coherent multi-RX "
                         "via Option C — per-RX phase then average) or "
                         "0/1/2/3 for a single antenna (debug).")
    ap.add_argument("--buffer-sec", type=float, default=10.0)
    ap.add_argument("--low", type=float, default=0.8,
                    help="Heartbeat band low cutoff Hz (default 0.8).")
    ap.add_argument("--high", type=float, default=3.5,
                    help="Heartbeat band high cutoff Hz (default 3.5). "
                         "3.5 covers up to 120 BPM and excludes the 2nd "
                         "cardiac harmonic for any HR ≤120 BPM "
                         "(critical to avoid double-detection — see "
                         "the gate-8 backtest).")
    ap.add_argument("--auto", action="store_true",
                    help="Use adaptive (Welch-based) bandpass — overrides --low/--high. "
                         "Per-tick the script estimates the heartbeat fundamental f0 "
                         "and picks a narrow band around it (f0 ± 0.5 Hz, capped at 1.7*f0 "
                         "to exclude the 2nd cardiac harmonic). Recommended for HR ≥ 80 BPM "
                         "where the 2nd harmonic falls inside a fixed wide band.")
    ap.add_argument("--adaptive-band", action="store_true",
                    help="Narrow the bandpass around the Welch-estimated "
                         "f0 (in addition to the explicit band). Off by "
                         "default; opt in only on clean signals.")
    args = ap.parse_args()

    # ---- radar setup ----
    radar = Radar(args.cfg)
    P = radar.params
    fs = 1000.0 / P["frame_time"]
    n_samples = P["n_samples"]
    n_bins = n_samples // 2
    range_res = P["range_res"]
    print(f"[INFO] fs={fs:.1f} Hz   range_res={range_res:.4f} m   n_bins={n_bins}")

    # ---- shared state ----
    shared = SharedState(n_bins)
    if args.gate is not None:
        shared.target_gate = args.gate
        shared.init_done = True

    # ---- start radar worker thread ----
    print("[INFO] Starting radar worker thread...")
    print(f"[INFO] CSV recordings will be written to ./{REC_DIR}/")
    if args.gate is None:
        print("[INFO] Sit still ~30-100 cm in front of the radar. "
              "Auto-gate calibrating (~0.5s) ...")
    worker = threading.Thread(
        target=radar_worker, args=(radar, args, n_bins, shared),
        daemon=True, name="radar-worker")
    worker.start()

    # ---- Qt main loop ----
    app = QtWidgets.QApplication(sys.argv)
    win, tick = build_window(args, fs, n_bins, range_res, shared)
    win.show()

    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(50)  # 20 Hz GUI refresh

    # On window close, signal worker to stop and (optionally) auto-save
    def on_about_to_quit():
        shared.shutdown = True
        rec = shared.stop_recording_snapshot()
        if rec["start_unix"] is not None and len(rec["iq"]) > 0:
            print("[REC] Window closed while recording; auto-saving...")
            try:
                post_process_and_save(rec, args, fs, range_res)
            except Exception as e:
                print(f"[REC] Auto-save failed: {e}")

    app.aboutToQuit.connect(on_about_to_quit)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
