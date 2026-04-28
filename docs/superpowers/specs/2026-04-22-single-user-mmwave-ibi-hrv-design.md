# Single-User mmWave IBI/HRV Design Spec

> Scope: offline signal-processing and evaluation only. Hardware bring-up is owned by teammates. This spec defines the software contract for converting `IWR1443 + DCA1000` captures into `IBI + 5-minute HRV` and validating them against ECG.

## Goal

Build a single-user, stationary, ECG-referenced, offline pipeline that takes raw mmWave ADC captures and produces:

- beat timestamps,
- IBI sequence,
- cleaned NN interval sequence,
- five-minute HRV metrics,
- comparison plots and error metrics against ECG.

## Default assumptions

These assumptions are intentionally strict because they maximize the chance of success:

- one subject only,
- subject is seated or semi-reclined,
- radar faces the chest frontally,
- distance is `0.5 m` to `1.0 m`, target `~1.0 m`,
- quiet indoor scene,
- five-minute recordings,
- ECG recorded simultaneously,
- offline processing only,
- raw ADC data available through `DCA1000`.

## Success definition

### Minimum success

- raw ADC capture is parsable into a radar cube;
- a stable chest phase signal is visible in clean segments;
- a heartbeat-related waveform is recoverable for at least some multi-minute segments;
- radar-derived IBI is meaningfully correlated with ECG-derived RR/NN intervals.

### Main success

- five-minute windows produce:
  - `mean IBI`,
  - `mean HR`,
  - `RMSSD`,
  - `SDNN/SDRR`;
- radar-derived metrics are directionally consistent with ECG;
- median beat-level IBI error is in the tens-of-milliseconds regime on clean segments.

### Stretch success

- radar IBI quality remains acceptable across several repeated trials;
- the main model clearly outperforms a simple band-pass baseline.

## Hard non-goals

- multi-user tracking,
- arbitrary pose robustness,
- real-time embedded implementation,
- clinical-grade claims,
- reconstructing ECG morphology.

## Core decision

The main pipeline will be:

1. `SWRA581B`-style raw ADC parsing
2. range FFT and chest-bin selection
3. phase extraction and phase cleaning
4. TI vital-signs-inspired motion and noise suppression
5. `mmHRV`-style heartbeat extraction
6. beat detection and IBI generation
7. NN interval cleaning
8. five-minute HRV estimation
9. ECG-based evaluation

The main engineering approximation is:

- paper concept: `mmHRV adaptive decomposition`
- practical implementation: `VMD-based or equivalent variational mode decomposition extractor`

This is the best balance of paper faithfulness and implementation feasibility.

## Method priority

Use a two-tier method strategy:

- **critical path**: TI preprocessing + `mmHRV`-style extractor + ECG validation
- **stretch path**: `RF-HRV`-style high-frequency harmonic extraction

Reason:

- `mmHRV` is the most reproducible direct IBI/HRV paper for this exact problem,
- `RF-HRV` appears stronger on harder datasets, but reproducing it on `IWR1443` is riskier and should not block delivery.

## Why this design is the highest-probability path

### Why not just use TI vital-signs output

TI's xWR1443 vital-signs lab is a strong preprocessing reference, but its default operating point is for breathing-rate and heart-rate estimation, not precise beat timing.

Key facts from the TI developer guide:

- the waveform is sampled along the slow-time axis at the frame rate,
- the reference design uses `Fs_slow = 20 Hz`,
- processing is over a running window of about `16 s`,
- the algorithm pipeline ends in spectral/rate decisions.

This makes it excellent for:

- range-bin selection,
- phase extraction,
- phase unwrapping,
- phase differencing,
- impulsive-noise removal,
- motion rejection.

But it is not enough by itself for:

- beat localization,
- IBI estimation,
- HRV estimation.

### Why mmHRV is the primary paper

`mmHRV` is the closest direct match to the project goal because it does not stop at average heart rate. It explicitly tries to recover heartbeat timing and HRV.

For this project, the most important conceptual contributions are:

- heartbeat timing must be recovered from a cleaned chest phase waveform,
- simple heartbeat-band filtering is not robust enough by itself,
- a decomposition-style heartbeat extractor is better aligned with IBI than a pure spectral estimator.

### Why RF-HRV is not the first target

`RF-HRV` is stronger on difficult far-field, population-scale scenarios, but it depends on:

- a more advanced signal-selection stack,
- higher-order harmonic exploitation,
- system assumptions that are harder to reproduce quickly on `IWR1443`.

It is useful as evidence and inspiration, but too risky as the first implementation target.

## Paper-to-pipeline mapping

### Stage A: raw ADC parsing

Primary source:

- TI `SWRA581B` raw ADC capture app note

What to take:

- xWR14xx + DCA1000 complex data format
- MATLAB reshape logic
- chirp-major organization assumptions

Implementation requirement:

- convert binary data into a complex array organized by receiver and chirp/frame

### Stage B: chest signal extraction

Primary sources:

- TI vital-signs xWR1443 developer guide
- `mmHRV`

What to take:

- range FFT
- target range-bin tracking inside a user-specified range window
- phase extraction from the selected range bin

Single-user simplification:

- skip the full multi-user CFAR + DBSCAN stack as a first pass,
- instead use a narrow physical distance gate and periodicity-based bin scoring.

### Stage C: phase cleaning

Primary sources:

- TI vital-signs xWR1443 developer guide
- Scientific Reports 2024 DR-MUSIC paper

What to take:

- phase unwrap
- phase difference
- impulsive-noise correction
- motion gating using short windows
- median-filter detrending
- optional respiratory-harmonic suppression

### Stage D: heartbeat extraction

Primary source:

- `mmHRV`

What to take:

- decomposition of the cleaned phase signal into several band-limited modes,
- choose the component most consistent with heartbeat dynamics,
- detect peaks on the recovered heartbeat waveform.

Practical approximation:

- use `VMD` with a small `K`, inspect center frequencies, then pick the heartbeat-like mode.

Implementation note:

- the `ICASSP 2021` paper is the cleanest statement of the extractor,
- the `JIOT 2021` paper is the better reproduction guide because it adds signal selection, normalization, and broader evaluation.

### Stage E: ECG evaluation

Primary sources:

- Oura ECG/IBI/HRV comparison paper
- HRV Task Force 1996 guidance

What to take:

- five-minute windows as the short-term HRV unit,
- artifact-aware interval cleaning before HRV,
- do not compute HRV on every raw interval blindly,
- prefer duration-based window validity rather than simple beat counts,
- report agreement with more than correlation alone.

## Recommended algorithm stack

### Baseline stack

- range FFT
- single-bin or small-bin-cluster selection
- `angle -> unwrap -> detrend -> band-pass(0.8-4.0 Hz) -> find_peaks`

Purpose:

- verify the signal chain,
- provide a fallback,
- create a comparison baseline in the report.

### Main stack

- range FFT
- gated chest-bin selection
- `angle -> unwrap -> detrend -> despike -> motion gating -> optional phase difference`
- `VMD / mmHRV-style decomposition`
- heartbeat-mode selection
- peak detection
- IBI -> NN -> HRV

### Preprocessing upgrade

- add RLS-style respiratory harmonic suppression only if baseline contamination is obvious.

### Fallback comparison

- cepstrum-based IBI estimation from the Sensors 2020 sleeping-scenarios paper.

### Optional stretch comparison

- `RF-HRV`-inspired high-frequency harmonic branch after the main stack is already working

## Design choices that maximize success probability

### Choice 1: work only on clean stationary segments first

Do not try to solve motion robustness before proving clean-segment IBI.

### Choice 2: make signal-selection quality a first-class module

Do not assume the strongest magnitude range bin is the best bin. Use a quality score that includes:

- distance gate membership,
- phase variance,
- heartbeat-band energy,
- autocorrelation periodicity.

### Choice 3: keep ECG evaluation window-based

The final claim should be about:

- IBI quality,
- five-minute HRV quality,

not exact sample-level waveform alignment to ECG.

Recommended evaluation protocol:

- use paired five-minute windows,
- accept a window only if both ECG and radar retain enough valid interval duration,
- target `>= 80%` valid duration per accepted five-minute window,
- report `Pearson r`, `CCC`, `Bland-Altman`, and absolute error.

### Choice 4: preserve a three-tier goal structure

- Tier 1: visible heartbeat waveform
- Tier 2: usable IBI on clean segments
- Tier 3: five-minute HRV against ECG

This keeps the project shippable even if the stretch goal slips.

## Risks and countermeasures

### Risk: slow-time resolution too low

Consequence:

- beat timing becomes quantized and coarse.

Countermeasure:

- insist on raw ADC and a chirp/frame configuration that supports dense slow-time sampling.

### Risk: bad chest-bin selection

Consequence:

- no stable phase,
- spurious peaks,
- poor IBI.

Countermeasure:

- explicitly compare several candidate bins and keep a quality score.

### Risk: respiratory leakage dominates heartbeat

Consequence:

- baseline BPF produces wrong peaks.

Countermeasure:

- upgrade to decomposition extractor,
- optionally add harmonic suppression from DR-MUSIC.

### Risk: ECG synchronization drift

Consequence:

- misleading beat-wise comparisons.

Countermeasure:

- compare both beat-level and window-level metrics,
- record explicit sync events where possible.

### Risk: evaluation rules hide failure or discard too much data

Consequence:

- the final numbers become either misleadingly optimistic or too sparse to trust.

Countermeasure:

- use `valid interval duration / 300 s` as the acceptance metric,
- report window retention explicitly,
- keep the primary claim on accepted clean stationary windows only.

## Recommended module boundaries

- `src/radar_io.py`
- `src/radar_config.py`
- `src/range_processing.py`
- `src/chest_bin_selection.py`
- `src/phase_processing.py`
- `src/heartbeat_extractors.py`
- `src/beat_detection.py`
- `src/ecg_reference.py`
- `src/hrv_metrics.py`
- `src/evaluation.py`
- `scripts/run_pipeline.py`
- `tests/...`

## Main sources

- `mmHRV: Contactless Heart Rate Variability Monitoring using Millimeter-Wave Radio`
- `Radio Frequency Based Heart Rate Variability Monitoring`
- TI xWR1443 vital-signs guides
- TI `SWRA581B`
- `Monitoring long-term cardiac activity with contactless radio frequency signals`
- `A high precision vital signs detection method based on millimeter wave radar`
- Oura ECG comparison paper
- ESC/NASPE 1996 HRV Task Force guidance
