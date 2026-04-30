import json
import math

import numpy as np
import pytest

from radar_analysis.pipeline import run_pipeline
from radar_analysis.synthetic import synthetic_range_cube


def _build_cube(duration_s=30.0, hr_hz=1.2):
    range_res_m = 0.04
    fs_slow_hz = 25.0
    cube = synthetic_range_cube(
        n_frames=int(duration_s * fs_slow_hz),
        n_chirps=16,
        n_range_bins=64,
        n_rx=4,
        range_res_m=range_res_m,
        target_range_m=0.7,
        target_amp=2000.0,
        target_motion_amp_mm=1.0,
        target_motion_hz=hr_hz,
        clutter_range_m=1.2,
        clutter_amp=300.0,
        noise_std=10.0,
        fs_frame=fs_slow_hz,
        seed=0,
    )
    return cube, range_res_m, fs_slow_hz


def test_pipeline_runs_end_to_end_on_synthetic():
    cube, rr, fs = _build_cube()
    result = run_pipeline(cube, range_res_m=rr, fs_slow_hz=fs)
    # Chest bin landed near the simulated target (0.7 m → bin 17 ± 1)
    assert abs(result.chest_bin - 17) <= 1
    # 30 s × 1.2 Hz ≈ 36 beats; allow generous tolerance for filter edge effects
    assert result.peak_times_s.size > 20
    # Most beats should pass the IBI cleaner
    assert result.nn_ms.size >= 0.7 * result.peak_times_s.size - 1
    # Mean HR within 10 BPM of the simulated 72 BPM
    assert abs(result.metrics["mean_hr_bpm"] - 72.0) < 10.0


def test_pipeline_metrics_are_finite_with_enough_beats():
    cube, rr, fs = _build_cube()
    result = run_pipeline(cube, range_res_m=rr, fs_slow_hz=fs)
    for key in ("mean_ibi_ms", "mean_hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct"):
        assert np.isfinite(result.metrics[key]), f"{key} was {result.metrics[key]}"


def test_pipeline_writes_plots_and_metrics_json(tmp_path):
    cube, rr, fs = _build_cube(duration_s=20.0)
    result = run_pipeline(cube, range_res_m=rr, fs_slow_hz=fs, out_dir=tmp_path)
    expected = ["range_fft.png", "range_time.png", "phase.png", "heartbeat.png", "ibi.png", "metrics.json"]
    for name in expected:
        assert (tmp_path / name).exists(), f"missing {name}"

    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert summary["chest_bin"] == result.chest_bin
    assert summary["n_beats_detected"] == int(result.peak_times_s.size)
    # Big arrays must NOT be in the JSON
    for k in ("phi_clean", "heartbeat", "peak_times_s", "ibi_ms", "nn_ms"):
        assert k not in summary
    # JSON metric values round-trip to the in-memory result (within float repr).
    for k, v in result.metrics.items():
        assert summary["metrics"][k] == pytest.approx(v, nan_ok=True)


def test_pipeline_handles_short_capture_without_crash():
    # 15 s is enough samples for the bandpass + detrend; metrics should be finite,
    # but the test verifies the pipeline doesn't crash on the lower end of usable.
    cube, rr, fs = _build_cube(duration_s=15.0)
    result = run_pipeline(cube, range_res_m=rr, fs_slow_hz=fs)
    assert result.duration_s == pytest.approx(15.0, abs=0.1)


def test_pipeline_returns_nan_metrics_when_no_beats():
    # Constant-amplitude cube → bandpass output is flat → detect_beats returns empty
    # → nn_ms.size == 0 → all HRV metrics must be NaN, not crash, not silently 0.
    cube = np.full((250, 16, 64, 4), 100.0 + 0j, dtype=np.complex64)
    result = run_pipeline(cube, range_res_m=0.04, fs_slow_hz=25.0)
    assert result.peak_times_s.size == 0
    assert result.nn_ms.size == 0
    for v in result.metrics.values():
        assert math.isnan(v)


def test_pipeline_pure_noise_does_not_crash():
    # Adversarial noise; metrics may be finite OR NaN, but the function must not raise.
    rng = np.random.default_rng(42)
    shape = (250, 16, 64, 4)
    cube = (rng.standard_normal(shape, dtype=np.float32)
            + 1j * rng.standard_normal(shape, dtype=np.float32)) * np.float32(20.0)
    result = run_pipeline(cube.astype(np.complex64), range_res_m=0.04, fs_slow_hz=25.0)
    for v in result.metrics.values():
        assert math.isnan(v) or math.isfinite(v)


def test_pipeline_save_plots_with_zero_beats(tmp_path):
    # Same constant-amplitude cube as above, but with out_dir set: the plotting path
    # must handle peak_times_s.size == 0 and ibi_ms.size == 0 without raising.
    cube = np.full((250, 16, 64, 4), 100.0 + 0j, dtype=np.complex64)
    run_pipeline(cube, range_res_m=0.04, fs_slow_hz=25.0, out_dir=tmp_path)
    for name in ("range_fft.png", "range_time.png", "phase.png",
                 "heartbeat.png", "ibi.png", "metrics.json"):
        assert (tmp_path / name).exists(), f"missing {name}"


def test_pipeline_phase_amplitude_reflects_dc_removal():
    # Add strong static clutter at the chest bin to force a DC bias on the slow-time
    # phasor. Without remove_dc, atan2 + unwrap collapses the chest swing to ~0. The
    # pipeline's detrended phase must still preserve a nontrivial swing.
    range_res_m = 0.04
    fs = 25.0
    cube = synthetic_range_cube(
        n_frames=int(15 * fs),
        n_chirps=8,
        n_range_bins=64,
        n_rx=4,
        range_res_m=range_res_m,
        target_range_m=0.7,
        target_amp=200.0,
        target_motion_amp_mm=1.0,
        target_motion_hz=1.2,
        clutter_range_m=0.7,        # CO-LOCATED with the target → biases atan2
        clutter_amp=5000.0,         # 25× target amplitude
        noise_std=5.0,
        fs_frame=fs,
        seed=0,
    )
    result = run_pipeline(cube, range_res_m=range_res_m, fs_slow_hz=fs)
    # Heartbeat-band signal should still have meaningful power post-DC-removal.
    assert result.heartbeat.std() > 0.01
    # Detrended phase mean is near zero (median detrend works) and std is nontrivial.
    assert abs(float(result.phi_clean.mean())) < 0.5
    assert float(result.phi_clean.std()) > 0.005


def test_pipeline_json_metrics_are_finite_or_nan(tmp_path):
    # JSON must round-trip every metric to a Python float (no None, no string "NaN"
    # surprise). With Python's default allow_nan=True, NaN serializes as the literal
    # NaN token and json.loads parses it back to float('nan'); we accept either.
    cube, rr, fs = _build_cube(duration_s=15.0)
    run_pipeline(cube, range_res_m=rr, fs_slow_hz=fs, out_dir=tmp_path)
    summary = json.loads((tmp_path / "metrics.json").read_text())
    for k, v in summary["metrics"].items():
        assert isinstance(v, float), f"{k}={v!r} ({type(v).__name__}) is not a Python float"
        assert math.isfinite(v) or math.isnan(v), f"{k} is non-finite-non-nan: {v}"


def test_pipeline_validates_inputs():
    with pytest.raises(ValueError):
        run_pipeline(np.zeros((5, 5)), range_res_m=0.04, fs_slow_hz=25.0)  # not 4D
    cube, rr, _ = _build_cube(duration_s=10.0)
    with pytest.raises(ValueError):
        run_pipeline(cube, range_res_m=rr, fs_slow_hz=0)
    with pytest.raises(ValueError):
        run_pipeline(cube, range_res_m=0, fs_slow_hz=25.0)
