# Single-User mmWave IBI/HRV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline pipeline that converts `IWR1443 + DCA1000` raw captures into `IBI + five-minute HRV` and validates against ECG in a controlled single-user setting.

**Architecture:** The system is a narrow offline signal-processing stack. Raw ADC is parsed into a radar cube, reduced to one or a few candidate chest slow-time signals, converted into phase, cleaned, transformed into a heartbeat waveform, and then converted into IBI and HRV for comparison with ECG-derived RR/NN intervals.

**Tech Stack:** Python, NumPy, SciPy, Pandas, Matplotlib, optional `sktime`/`vmdpy`-style VMD implementation, WFDB or equivalent ECG tooling.

---

## High-confidence delivery strategy

The plan is intentionally staged so that failure in a later stage does not destroy the whole project:

- `Milestone 0`: lock data contracts and success criteria
- `Milestone 1`: prove radar parsing and chest phase extraction
- `Milestone 2`: deliver a simple IBI baseline
- `Milestone 3`: deliver the main mmHRV-style extractor
- `Milestone 4`: deliver ECG-based five-minute HRV evaluation
- `Milestone 5`: deliver ablations, plots, and report-ready evidence

This is the path with the highest probability of a convincing final result.

Not on the critical path:

- `RF-HRV` high-frequency harmonic extraction

That method is a stretch upgrade, not the anchor of the delivery plan.

## File structure

### Core source files

- Create: `src/radar_io.py`
- Create: `src/radar_config.py`
- Create: `src/range_processing.py`
- Create: `src/chest_bin_selection.py`
- Create: `src/phase_processing.py`
- Create: `src/heartbeat_extractors.py`
- Create: `src/beat_detection.py`
- Create: `src/ecg_reference.py`
- Create: `src/hrv_metrics.py`
- Create: `src/evaluation.py`
- Create: `scripts/run_pipeline.py`

### Tests

- Create: `tests/test_radar_io.py`
- Create: `tests/test_range_processing.py`
- Create: `tests/test_phase_processing.py`
- Create: `tests/test_heartbeat_extractors.py`
- Create: `tests/test_beat_detection.py`
- Create: `tests/test_hrv_metrics.py`
- Create: `tests/test_evaluation.py`

### Notes and notebooks

- Create: `analysis/figures/`
- Create: `analysis/notebooks/01_parse_and_phase.ipynb`
- Create: `analysis/notebooks/02_baseline_ibi.ipynb`
- Create: `analysis/notebooks/03_main_model.ipynb`
- Create: `analysis/notebooks/04_ecg_eval.ipynb`

## Milestone 0: Data contract and evaluation contract

**Files:**
- Create: `src/radar_config.py`
- Create: `src/hrv_metrics.py`
- Create: `tests/test_hrv_metrics.py`

- [ ] **Step 1: Define the radar capture metadata contract**

The parser must know:

- `num_adc_samples`
- `num_rx`
- `num_tx_used`
- `num_chirps_per_frame`
- `num_frames`
- `is_complex`
- `adc_bits`
- `frame_periodicity_s`
- `radar_timebase`
- `ecg_timebase`

```python
from dataclasses import dataclass

@dataclass
class RadarCaptureConfig:
    num_adc_samples: int
    num_rx: int
    num_tx_used: int
    num_chirps_per_frame: int
    num_frames: int
    adc_bits: int
    is_complex: bool
    frame_periodicity_s: float
```

- [ ] **Step 2: Define HRV metric functions**

The project only requires robust time-domain HRV metrics at first.

```python
import numpy as np

def mean_ibi_ms(nn_ms: np.ndarray) -> float:
    return float(np.mean(nn_ms))

def mean_hr_bpm(nn_ms: np.ndarray) -> float:
    return 60000.0 / mean_ibi_ms(nn_ms)

def rmssd_ms(nn_ms: np.ndarray) -> float:
    diff = np.diff(nn_ms)
    return float(np.sqrt(np.mean(diff ** 2)))

def sdnn_ms(nn_ms: np.ndarray) -> float:
    return float(np.std(nn_ms, ddof=1))
```

- [ ] **Step 3: Define an agreement metric beyond correlation**

```python
def concordance_correlation_coefficient(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x), np.var(y)
    cov = np.mean((x - mx) * (y - my))
    return float((2 * cov) / (vx + vy + (mx - my) ** 2 + 1e-12))
```

- [ ] **Step 4: Write tests for the metric definitions**

```python
def test_rmssd_known_example():
    nn = np.array([1000.0, 980.0, 1020.0])
    expected = np.sqrt(((20.0)**2 + (40.0)**2) / 2.0)
    assert np.isclose(rmssd_ms(nn), expected)
```

Run: `pytest tests/test_hrv_metrics.py -v`

Expected: all metric tests pass.

## Milestone 1: Parse raw ADC and recover a chest phase signal

**Paper/doc mapping**

- TI `SWRA581B`:
  - use the `xWR14xx + DCA1000` complex format
  - mirror the MATLAB reshape logic from the binary parsing section
- TI vital-signs developer guide:
  - use range FFT and phase extraction from the selected range bin

**Files:**
- Create: `src/radar_io.py`
- Create: `src/range_processing.py`
- Create: `src/chest_bin_selection.py`
- Create: `src/phase_processing.py`
- Create: `tests/test_radar_io.py`
- Create: `tests/test_range_processing.py`
- Create: `tests/test_phase_processing.py`

- [ ] **Step 1: Write a parser for xWR14xx complex DCA1000 captures**

Python equivalent of TI's MATLAB logic:

```python
import numpy as np

def load_dca1000_xwr14xx_complex(path: str, num_rx: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.int16)
    iq = raw.reshape(-1, 8)
    complex_rows = iq[:, 0:4].astype(np.float32) + 1j * iq[:, 4:8].astype(np.float32)
    return complex_rows[:, :num_rx]
```

Note:

- this snippet mirrors the TI layout conceptually;
- final shape logic must be validated against the actual capture config.

- [ ] **Step 2: Reshape parsed data into chirp-major receiver data**

```python
def build_adc_cube(
    rx_data: np.ndarray,
    num_adc_samples: int,
    num_chirps_total: int,
    num_rx: int,
) -> np.ndarray:
    flat = rx_data.reshape(num_chirps_total, num_adc_samples, num_rx)
    return np.transpose(flat, (0, 2, 1))
```

Target shape:

- `(num_chirps_total, num_rx, num_adc_samples)`

- [ ] **Step 3: Range FFT**

```python
from scipy.signal import windows

def range_fft(adc_cube: np.ndarray) -> np.ndarray:
    win = windows.hann(adc_cube.shape[-1], sym=False)
    return np.fft.fft(adc_cube * win[None, None, :], axis=-1)
```

- [ ] **Step 4: Implement chest-bin scoring**

This is the first major project-specific judgment module.

```python
def candidate_bin_score(z: np.ndarray, fs_slow: float) -> float:
    phase = np.unwrap(np.angle(z))
    phase = phase - np.median(phase)
    spec = np.abs(np.fft.rfft(phase))
    freqs = np.fft.rfftfreq(len(phase), d=1.0 / fs_slow)
    band = (freqs >= 0.1) & (freqs <= 3.0)
    total = np.sum(spec) + 1e-8
    return float(np.sum(spec[band]) / total)
```

The final score should combine:

- allowed range gate,
- phase variance,
- heartbeat/respiration-band energy,
- autocorrelation periodicity.

This is the single-user simplification of the `mmHRV` journal paper's more complex signal-selection stack.

- [ ] **Step 5: Extract the 1D phase signal**

```python
def extract_phase_signal(z: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(z))
```

- [ ] **Step 6: Verification notebook**

Plot:

- one range profile,
- candidate chest bins,
- the chosen slow-time complex signal,
- the raw unwrapped phase.

Run: notebook `analysis/notebooks/01_parse_and_phase.ipynb`

Expected:

- a stable phase trace on clean data,
- obvious respiration,
- at least weak cardiac-band structure.

Go/no-go:

- if no candidate bin inside the physical chest range produces stable periodic phase, stop and inspect acquisition before implementing more algorithms.

## Milestone 2: Deliver a simple IBI baseline

**Paper/doc mapping**

- TI vital-signs guide:
  - phase difference,
  - impulse-noise removal,
  - band-pass filtering,
  - motion-segment rejection

**Files:**
- Modify: `src/phase_processing.py`
- Create: `src/beat_detection.py`
- Create: `src/heartbeat_extractors.py`
- Create: `tests/test_beat_detection.py`

- [ ] **Step 1: Implement phase detrending and despiking**

```python
from scipy.signal import medfilt

def detrend_phase(phi: np.ndarray, kernel_size: int = 201) -> np.ndarray:
    trend = medfilt(phi, kernel_size=kernel_size)
    return phi - trend
```

```python
def phase_difference(phi: np.ndarray) -> np.ndarray:
    return np.diff(phi, prepend=phi[0])
```

- [ ] **Step 2: Implement 1-second motion gating**

```python
def motion_mask(x: np.ndarray, fs: float, energy_thresh: float) -> np.ndarray:
    block = int(round(fs))
    mask = np.ones_like(x, dtype=bool)
    for start in range(0, len(x), block):
        stop = min(len(x), start + block)
        if np.sum(x[start:stop] ** 2) > energy_thresh:
            mask[start:stop] = False
    return mask
```

- [ ] **Step 3: Implement the baseline heartbeat extractor**

```python
from scipy.signal import butter, filtfilt

def bandpass(x: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    return filtfilt(b, a, x)

def baseline_heartbeat_waveform(phi_clean: np.ndarray, fs: float) -> np.ndarray:
    return bandpass(phi_clean, fs, 0.8, 4.0)
```

- [ ] **Step 4: Implement beat detection**

```python
from scipy.signal import find_peaks

def detect_beats(beat_wave: np.ndarray, fs: float):
    peaks, props = find_peaks(
        beat_wave,
        distance=int(0.35 * fs),
        prominence=np.std(beat_wave) * 0.25,
    )
    return peaks, props
```

- [ ] **Step 5: Convert peaks to IBI and reject impossible intervals**

```python
def peaks_to_ibi_ms(peaks: np.ndarray, fs: float) -> np.ndarray:
    t = peaks / fs
    ibi_ms = np.diff(t) * 1000.0
    return ibi_ms

def clean_ibi_basic(ibi_ms: np.ndarray) -> np.ndarray:
    keep = (ibi_ms >= 400.0) & (ibi_ms <= 1500.0)
    return ibi_ms[keep]
```

- [ ] **Step 6: Baseline verification**

Plot:

- cleaned phase,
- baseline heartbeat waveform,
- detected peaks,
- resulting IBI trace.

Run: `analysis/notebooks/02_baseline_ibi.ipynb`

Expected:

- plausible beat spacing,
- coarse but non-random IBI,
- enough evidence that the chain is alive.

Go/no-go:

- if the baseline cannot produce plausible beat spacing on the cleanest minute, do not move to the main extractor until phase cleaning and bin selection are revisited.

## Milestone 3: Deliver the main mmHRV-style extractor

**Paper/doc mapping**

- `mmHRV` journal paper:
  - adaptive decomposition,
  - heartbeat component selection,
  - beat timing recovery

**Files:**
- Modify: `src/heartbeat_extractors.py`
- Modify: `tests/test_heartbeat_extractors.py`

- [ ] **Step 1: Add a variational decomposition implementation**

Use a library implementation if available; otherwise vendor a minimal VMD implementation.

```python
def decompose_modes(phi_clean: np.ndarray, K: int = 5, alpha: float = 2000):
    # placeholder API; replace with actual VMD package call
    modes = vmd(phi_clean, K=K, alpha=alpha)
    return modes
```

Implementation note:

- this is an engineering approximation of the `mmHRV` ADMM/VMD-style search,
- use a small grid over `K` and `alpha` rather than a single frozen setting.

- [ ] **Step 2: Score each mode as a heartbeat candidate**

```python
def dominant_freq_hz(x: np.ndarray, fs: float) -> float:
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    idx = np.argmax(spec[1:]) + 1
    return float(freqs[idx])

def heartbeat_mode_score(mode: np.ndarray, fs: float) -> float:
    f0 = dominant_freq_hz(mode, fs)
    if not (0.8 <= f0 <= 3.5):
        return -1.0
    peaks, _ = detect_beats(mode, fs)
    if len(peaks) < 3:
        return -1.0
    ibi = peaks_to_ibi_ms(peaks, fs)
    consistency = -np.std(ibi)
    snr_like = np.std(mode) / (np.median(np.abs(mode - np.median(mode))) + 1e-8)
    return float(snr_like + 0.01 * consistency)
```

- [ ] **Step 3: Select the best mode**

```python
def main_heartbeat_waveform(phi_clean: np.ndarray, fs: float) -> np.ndarray:
    modes = decompose_modes(phi_clean, K=5, alpha=2000)
    scores = [heartbeat_mode_score(m, fs) for m in modes]
    best = int(np.argmax(scores))
    return modes[best]
```

- [ ] **Step 4: Compare main model against baseline**

Success criterion for this milestone:

- main model should produce fewer obviously spurious peaks than the baseline on the cleanest segments.

- [ ] **Step 5: Optional stretch branch only after the main model works**

If time remains, add a limited `RF-HRV`-inspired branch:

- inspect candidate bins/channels beyond the fundamental heart band,
- try a high-frequency heartbeat reconstruction branch,
- compare it against the main model.

Do not block the main path on this branch.

- [ ] **Step 6: Verification notebook**

Plot:

- baseline extractor output,
- each decomposition mode,
- chosen heartbeat mode,
- main-model peaks and IBI.

Run: `analysis/notebooks/03_main_model.ipynb`

## Milestone 4: ECG reference and HRV evaluation

**Paper/doc mapping**

- Oura paper:
  - five-minute windowing,
  - artifact-aware interval acceptance,
  - rMSSD as the primary short-term HRV metric

**Files:**
- Create: `src/ecg_reference.py`
- Modify: `src/hrv_metrics.py`
- Create: `src/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] **Step 1: Define the ECG reference contract**

Allow either:

- raw ECG waveform input, or
- precomputed R-peak timestamps.

```python
def rpeaks_to_rr_ms(rpeaks_s: np.ndarray) -> np.ndarray:
    return np.diff(rpeaks_s) * 1000.0
```

- [ ] **Step 2: Clean RR into NN**

```python
def clean_nn_intervals(rr_ms: np.ndarray) -> np.ndarray:
    med = np.median(rr_ms)
    keep = (rr_ms >= 400.0) & (rr_ms <= 1500.0) & (np.abs(rr_ms - med) <= 0.2 * med)
    return rr_ms[keep]
```

Reference-signal rule:

- manually inspect ECG R-peaks for the final reference set,
- do not trust fully automatic cleanup for the gold-standard side.

- [ ] **Step 3: Build aligned five-minute windows**

```python
def window_mask(times_s: np.ndarray, start_s: float, dur_s: float = 300.0) -> np.ndarray:
    return (times_s >= start_s) & (times_s < start_s + dur_s)
```

- [ ] **Step 4: Add duration-based window validity**

```python
def valid_duration_ratio(interval_ms: np.ndarray) -> float:
    return float(np.sum(interval_ms) / 300000.0)
```

Window acceptance rule:

- accept a five-minute window only if both radar and ECG keep at least `80%` valid duration.

- [ ] **Step 5: Compute radar vs ECG error metrics**

```python
def ibi_error_summary(radar_ibi_ms: np.ndarray, ecg_rr_ms: np.ndarray) -> dict:
    n = min(len(radar_ibi_ms), len(ecg_rr_ms))
    err = radar_ibi_ms[:n] - ecg_rr_ms[:n]
    ae = np.abs(err)
    return {
        "mean_error_ms": float(np.mean(err)),
        "mae_ms": float(np.mean(ae)),
        "median_ae_ms": float(np.median(ae)),
    }
```

- [ ] **Step 6: Compute metric-level agreement**

For each five-minute window, calculate:

- `mean IBI`
- `mean HR`
- `RMSSD`
- `SDNN`

Then compare radar and ECG with:

- absolute error,
- signed error,
- Pearson correlation,
- concordance correlation coefficient,
- Bland-Altman bias and limits of agreement.

- [ ] **Step 7: Verification notebook**

Run: `analysis/notebooks/04_ecg_eval.ipynb`

Expected:

- one table of per-window metrics,
- one summary table of overall errors,
- one or more plots of radar vs ECG agreement.

## Milestone 5: Ablation and report-ready evidence

**Files:**
- Modify: `src/evaluation.py`
- Create: `analysis/figures/...`

- [ ] **Step 1: Baseline vs main model comparison**

Produce one table with:

- signal quality score,
- number of detected beats,
- IBI median AE,
- RMSSD error,
- SDNN error.

- [ ] **Step 2: Failure taxonomy**

Tag failure windows manually or heuristically:

- motion burst,
- wrong bin,
- respiratory leakage,
- weak signal,
- sync uncertainty.

- [ ] **Step 3: Final figure set**

Required final figures:

- range profile with selected chest bin
- raw phase vs cleaned phase
- baseline heartbeat waveform with peaks
- main-model heartbeat waveform with peaks
- radar IBI vs ECG RR
- radar HRV vs ECG HRV table

## Paper-to-code exact mapping

### TI `SWRA581B`

Use for:

- `src/radar_io.py`
- binary parse shape assumptions

### TI vital-signs xWR1443

Use for:

- `src/range_processing.py`
- `src/chest_bin_selection.py`
- `src/phase_processing.py`

Specific blocks to mirror:

- range FFT
- target range-bin tracking
- phase extraction
- phase unwrap
- phase difference
- impulsive-noise removal
- motion-corrupted segment removal

### `mmHRV`

Use for:

- `src/heartbeat_extractors.py`
- the conceptual logic of decomposition-based heartbeat recovery
- peak-based IBI generation

### DR-MUSIC paper

Use only if needed in:

- `src/phase_processing.py`

Specific optional upgrade:

- respiratory harmonic suppression before the main extractor

### RF-HRV

Use only as a stretch comparison after the main path is stable.

Specific idea to borrow:

- stronger signal-selection logic,
- cleaner heartbeat information may exist outside the fundamental heart-rate band.

### Oura

Use for:

- `src/hrv_metrics.py`
- `src/evaluation.py`

Specific ideas:

- five-minute windows,
- reject abnormal intervals before HRV,
- use `rMSSD` as a primary short-term metric.

## Go / no-go thresholds

### After Milestone 1

Go if:

- chest phase is stable and not dominated by gross unwrap failures.

No-go if:

- no candidate chest bin produces usable phase on a clean recording.

### After Milestone 2

Go if:

- baseline IBI is plausible on at least one clean minute.

No-go if:

- peak timing is essentially random.

### After Milestone 3

Go if:

- main model clearly improves over baseline on clean segments.

Fallback if:

- use baseline + DR-MUSIC preprocessing + careful NN cleaning.

### After Milestone 4

Main success if:

- five-minute metric agreement is clearly meaningful and reproducible.

Recommended targets for a strong course-project result:

- accepted five-minute windows should usually have `>= 80%` valid duration,
- retained accepted windows in a noisy prototype may reasonably land around `65% - 75%`,
- `HR` agreement target: correlation or CCC `>= 0.95`,
- `rMSSD` agreement target: correlation or CCC `>= 0.85`.

Stretch success if:

- beat-level median AE is in the same order as the cleaner `mmHRV` single-user results.

## Reporting language

Allowed claim:

- `In a controlled single-user stationary setting, our radar pipeline can recover a heartbeat-related waveform and estimate IBI/short-window HRV with meaningful agreement to ECG.`

Not allowed claim:

- `clinical-grade`
- `ECG-equivalent`
- `robust in general settings`
