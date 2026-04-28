# mmWave IBI/HRV Research Note for IWR1443 + DCA1000

## Goal

We need a practical path from raw mmWave signals to inter-beat intervals (IBI) and short-window HRV metrics in a controlled single-user stationary setting over five-minute recordings.

This note focuses on:

- what models/pipelines are actually usable for this project,
- how they work at the signal-processing level,
- which one is the best fit for `IWR1443 + DCA1000`,
- what to read from the local board notes before implementation.

## Bottom line

The best project-fit path is not "just use TI's vital-signs demo" and not "jump directly to the newest big paper."

The best fit is:

1. use TI's official raw capture path to get reliable ADC data;
2. use TI vital-signs style preprocessing for range-bin selection, phase extraction, unwrapping, impulsive-noise removal, and motion-segment rejection;
3. use an mmHRV-style heartbeat extractor on the cleaned phase signal to recover a beat waveform;
4. detect peaks on that beat waveform, turn them into IBI, then compute five-minute HRV metrics;
5. compare against ECG if possible, high-quality PPG if not.

My recommendation is a hybrid pipeline:

- TI front-end preprocessing
- mmHRV heartbeat extraction
- Oura-style "normal IBI only" quality control before HRV calculation

This gives the strongest balance of:

- direct relevance to your hardware,
- direct relevance to IBI rather than just average HR,
- realistic implementation scope for a class project.

## What the local PDF is telling the team to learn

From `Copy of 6.1820 Using IWR1443 + DCA1000 Notes-1.pdf`, the key links are:

- TI IWR1443BOOST user guide:
  `https://www.ti.com/lit/ug/swru518d/swru518d.pdf`
- TI mmWave SDK user guide:
  `https://dr-download.ti.com/software-development/software-development-kit-sdk/MD-PIrUeCYr3X/03.06.00.00-LTS/mmwave_sdk_user_guide.pdf`
- TI DCA1000 quick start:
  `https://www.ti.com/lit/ml/spruik7/spruik7.pdf`
- TI mmWave Studio user guide:
  `https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-h04ItoajtS/02.01.01.00/mmwave_studio_user_guide.pdf`
- mmWave demo visualizer:
  `https://dev.ti.com/gallery/view/mmwave/mmWave_Demo_Visualizer/ver/2.1.0/`
- Radar Toolbox:
  `https://dev.ti.com/tirex/explore/node?node=A__AXAenV2u4woV.FhTlAk68Q__radar_toolbox__1AslXXD__LATEST`
- UniFlash:
  `https://www.ti.com/tool/UNIFLASH`

Non-official but useful engineering references linked in the PDF:

- `https://github.com/ConnectedSystemsLab/xwr_raw_ros/tree/main/src/xwr_raw`
- `https://github.com/r-bt/MakeyMakey/tree/main/src/xwr`
- `https://github.com/r-bt/MakeyMakey/blob/main/scripts/1443_mmwavestudio_config_old.lua`

Takeaway from the local notes:

- the workflow is old and version-sensitive;
- everybody should understand `mmWave Studio`, `DCA1000`, `UniFlash`, and the raw capture path before coding algorithms;
- the hardware team and signal-processing team should agree early on the exact chirp/frame config, because your offline algorithm quality depends on the slow-time sampling rate and signal stability.

## Official TI capture path: what the board actually gives you

### Hardware/tool chain

The `IWR1443BOOST` board exposes `4 RX` and `3 TX` antennas and supports high-speed LVDS lanes through the 60-pin connector, which is what the capture board uses. The board is powered by `5 V`, and TI recommends pressing reset after power-on for reliable boot-up.

The `DCA1000EVM` receives LVDS data and sends raw ADC data to the PC over Ethernet as UDP packets.

Important details from TI docs:

- DCA1000 streams LVDS data as UDP datagrams to the host PC.
- The packets include sequence metadata so dropped or out-of-order packets can be detected.
- In newer mmWave Studio releases, reordered files are already produced and can be used directly for post-processing.
- Typical default addressing in TI examples is PC `192.168.33.30`, DCA1000 `192.168.33.180`, config port `4096`, data port `4098`.

### Why this matters for your project

This means your project should be framed as:

- raw ADC capture on Windows using TI tools,
- offline MATLAB/Python processing after capture,
- not real-time embedded IBI estimation on the board.

That matches your proposal exactly.

## What TI's own vital-signs lab already tells us

TI's `Vital Signs` lab for `IWR14xx` is very important, even if you do not use its output directly.

It states:

- adult respiration amplitude is about `1-12 mm` at `0.1-0.5 Hz`;
- adult heartbeat amplitude is about `0.1-0.5 mm` at `0.8-2 Hz`;
- the system measures phase change at the selected target range bin to recover these small chest displacements.

Its processing chain is:

1. range FFT
2. target range-bin tracking
3. phase extraction
4. phase unwrapping
5. phase difference
6. impulsive-noise removal
7. band-pass filtering for breathing and heartbeat
8. motion-corrupted segment rejection
9. spectral estimation / peak interval logic

This is extremely useful, but there is a critical limitation for your project:

- TI's lab is designed mainly for breathing rate and heart rate,
- not for precise beat-to-beat timing.

In TI's demo guide, the vital-sign waveform is sampled at the slow-time/frame rate, and the example implementation uses `20 Hz` slow-time sampling with `50 ms` frame periodicity and a `~16 s` running window for rate estimation.

Inference:

- that is good enough for rate estimation,
- but by itself it is not a convincing end-to-end design for millisecond-level IBI precision.

So TI's lab should be treated as:

- a preprocessing and engineering reference,
- not the final IBI algorithm.

## Model family 1: simple filter-bank + peak detection baseline

### Core idea

Take the selected chest phase signal, remove drift, band-pass the heartbeat band, suppress respiration leakage, then detect peaks directly.

### How it works

Typical steps:

1. select the target range bin or a small set of adjacent bins;
2. extract complex slow-time signal;
3. compute phase with `atan2(Q, I)`;
4. unwrap the phase;
5. detrend / remove baseline drift;
6. optionally apply phase difference to enhance the heartbeat component;
7. band-pass the heartbeat band, for example around `0.8-4 Hz`;
8. enforce physiologic peak spacing constraints;
9. convert peak times to IBI.

### Pros

- easiest first baseline;
- fast to implement;
- enough to prove that your pipeline is alive.

### Cons

- respiration harmonics leak badly into the heartbeat band;
- very sensitive to motion and range-bin mistakes;
- often good for average heart rate but not robust beat timing.

### Verdict

You should build this first as a debugging baseline, but do not make it your final main algorithm.

## Model family 2: mmHRV heartbeat extractor

### Why this is the most important paper for you

The ICASSP 2021 and IEEE IoT Journal 2021 `mmHRV` papers are the closest direct match to your research question:

- mmWave radar
- contactless HRV
- recover heartbeat timing
- derive IBI and short-window HRV metrics

### How mmHRV works

The mmHRV model starts from the selected phase signal at the chest range bin.

It models the signal as:

- body motion component
- respiration component
- heartbeat component
- noise

Then, instead of using one narrow band-pass filter, it decomposes the composite phase signal into multiple band-limited modes. The heartbeat component is the one whose amplitude/frequency behavior matches a plausible heartbeat signal.

The ICASSP 2021 paper describes:

- sparse/band-limited decomposition of the phase signal,
- ADMM-based optimization,
- adaptive tuning of the number of components `K` and penalty `alpha`,
- selection of the heartbeat mode,
- normalization,
- peak extraction on the recovered heartbeat waveform,
- IBI from the distance between adjacent peaks.

### Why this is better than simple BPF

The mmHRV papers explicitly argue that simple BPF approaches depend too much on knowing the heart-rate band cleanly and are vulnerable to respiratory leakage and noise. The decomposition approach tries to recover the heartbeat waveform itself rather than just a dominant frequency estimate.

That matters because:

- HR estimation only needs a dominant frequency;
- IBI estimation needs the exact time of individual beats.

### What performance they report

In the ICASSP 2021 paper, mmHRV reported:

- `3.68 ms` average error of mean IBI,
- `6.61 ms` average error of RMSSD,
- `7.09 ms` average error of SDRR,
- on `10` participants at `1 m`,
- outperforming a band-pass-filter-bank baseline.

The journal paper extends this line and shows the same design direction is not a one-off short-paper result.

### What makes it project-fit

- It directly solves your exact problem.
- It is phase-based and compatible with `IWR1443 + DCA1000`.
- It is still classical signal processing, so you do not need a big labeled dataset.
- It gives you a defensible paper baseline.

### Main risk

You likely need to implement the decomposition yourself because there does not seem to be an official public mmHRV implementation.

### Verdict

This should be your main algorithmic baseline.

## Model family 3: TI vital-signs style preprocessing + mmHRV back-end

### This is my recommended final project pipeline

Use TI's field-proven frontend to stabilize the signal, then use mmHRV to recover beat timing.

Recommended chain:

1. range FFT on raw ADC
2. select chest range bin within a user-defined distance window
3. extract phase
4. unwrap phase
5. impulsive-noise correction
6. motion-segment rejection
7. optional phase difference for drift reduction / heartbeat enhancement
8. mmHRV-style adaptive decomposition
9. select heartbeat mode
10. normalize waveform
11. detect peaks
12. convert to IBI
13. clean IBI into normal-to-normal intervals
14. compute HRV over five-minute windows

### Why this hybrid is stronger than pure mmHRV

Because on your actual hardware, the hardest failure modes will likely be:

- wrong range-bin selection,
- phase wrapping spikes,
- short motion bursts,
- drift and respiration leakage.

TI's lab already has engineering logic for those issues.

mmHRV is stronger specifically at the last mile:

- turning a cleaned chest-motion signal into beat-level timing.

## Model family 4: DR-MUSIC / RLS + MUSIC preprocessing

The 2024 Scientific Reports paper is not a direct IBI paper, but it is still valuable.

Its pipeline is:

- target phase extraction,
- phase unwrapping,
- median filtering for baseline drift removal,
- RLS adaptive filter to suppress respiratory harmonics,
- phase differencing to enhance weak heartbeat components,
- MUSIC spectral estimation for accurate heart-rate frequency extraction.

### Why it matters

This paper is strong evidence for a preprocessing idea:

- remove baseline drift first;
- explicitly cancel respiratory harmonics;
- then enhance the weak heartbeat component.

### Why it is not enough alone

Its endpoint is mainly precise heart-rate frequency estimation, not beat-by-beat event timing.

MUSIC gives you a strong estimate of frequency content.
IBI/HRV needs beat timestamps.

### Best use in your project

Use this as inspiration for:

- median-filter detrending,
- respiration-harmonic suppression,
- not as your final IBI estimator.

## Model family 5: cepstrum-based IBI extraction

The `Vital Sign Monitoring Using FMCW Radar in Various Sleeping Scenarios` paper provides another useful model family.

### How it works

Instead of extracting an explicit beat waveform and doing time-domain peak picking, it:

- takes multiple selected radar range signals,
- applies several FFT window lengths in parallel,
- computes the log-spectrum,
- takes inverse FFT to form cepstra,
- combines them into a summary cepstrum,
- then takes peaks in cepstrum space as IBI estimates.

The paper notes that cepstral analysis can emphasize weak heartbeat-induced motions through the logarithm of the spectrum and multi-window summary approach.

### Why this is interesting

- It is a legitimate IBI pipeline, not just HR estimation.
- It can work even when the beat waveform is not visually clean.
- It gives you a fundamentally different fallback from direct peak-detection pipelines.

### Limitation

That paper also reports that some metrics like `pNNI20/pNNI50` are more fragile, partly because the method can yield more than one estimate per actual heartbeat.

### Verdict

This is a strong fallback or comparison baseline.
It is not my first choice for your main pipeline, but it is very useful if direct beat-wave extraction is unstable.

## Model family 6: RF-HRV high-frequency harmonic extraction

The 2024 Nature Communications `RF-HRV` paper is the strongest recent result set, but it is not the easiest paper to reproduce on your exact stack.

### The key conceptual leap

Their claim is that trying to estimate HRV only inside the usual heart-rate frequency band is fundamentally limited by respiratory leakage.

So instead of looking only near the fundamental heart rate band, they search for heartbeat information in higher-frequency patterns/harmonics that are less contaminated by respiration.

The paper reports:

- a signal-selection stage over multiple candidate signals/voxels,
- beat-frequency pattern extraction,
- VMD-based extraction of a high-frequency component,
- better RT-IBI / RMSSD / SDRR / pNN50 than mmHRV and V2iFi baselines in their evaluation.

### Why this matters for your project

It gives a very important lesson:

- "heartbeat fundamental band" is not always the best place to recover beat timing.

### Why I do not recommend it as your first milestone

- different hardware generation and broader system design,
- much more sophisticated signal selection,
- likely too much to reproduce cleanly under class-project time pressure.

### Best use

Use it as:

- evidence that the problem is solvable,
- inspiration for a later ablation if mmHRV struggles,
- not the first implementation target.

## Model family 7: Oura-style HRV computation discipline

The Oura paper is not a radar paper, but it is useful for the evaluation side of your project.

Important takeaways:

- they compare PPG-derived IBI against ECG-derived RR intervals;
- they compute HR and HRV in `5-minute` windows;
- each IBI is labeled `normal` or `abnormal`;
- they only include windows using consecutive normal IBIs;
- they use `rMSSD` as a short-window HRV metric.

### Why this matters for your radar project

Your radar pipeline should not treat every raw detected interval as trustworthy.

You should have an explicit "NN interval" cleaning stage:

- detect likely artifact intervals,
- remove them or mark them invalid,
- compute HRV only on valid intervals.

This is especially important because radar errors will often appear as:

- one missed beat,
- one duplicated beat,
- one spurious peak after motion,
- one interval that is physiologically impossible.

### Verdict

Use the Oura paper as:

- evidence that five-minute windowing is standard,
- evidence that artifact rejection matters,
- not as the radar-side signal model.

## What is actually recommended for this specific project

### Best-fit stack

Primary path:

1. `IWR1443 + DCA1000` raw ADC capture
2. range FFT
3. chest range-bin / small-bin-cluster selection
4. phase extraction with `atan2`
5. phase unwrapping
6. detrending + impulsive spike cleanup
7. motion-corrupted segment rejection
8. mmHRV-style adaptive decomposition
9. heartbeat waveform peak detection
10. IBI sequence
11. outlier rejection to obtain NN intervals
12. five-minute HRV metrics

Comparison baselines:

- simple heartbeat-band BPF + peak detection
- cepstrum-based IBI extraction

Optional preprocessing upgrade:

- median filter drift removal and adaptive respiration-harmonic suppression inspired by DR-MUSIC

### Why this is the right balance

- more realistic than trying to reproduce RF-HRV in full;
- stronger than stopping at TI heart-rate estimation;
- better aligned to IBI than frequency-only estimators;
- feasible without deep-learning-scale data.

## Concrete implementation recipe

### Step 1: acquisition

For each five-minute recording:

- single subject;
- seated or semi-reclined;
- radar at chest height;
- short fixed distance, ideally around `0.8-1.2 m`;
- as frontal as possible;
- rigid mount for radar;
- quiet scene with no moving objects near the chest line.

If possible, record:

- radar raw ADC,
- ECG reference,
- start/stop synchronization event.

PPG can be a fallback reference, but ECG is better for millisecond IBI validation.

### Step 2: transform raw ADC into slow-time chest signals

For each chirp or frame:

1. apply range FFT;
2. pick the chest range bin inside a known distance gate;
3. optionally use adjacent bins and select the one with the best periodicity / variance / SNR;
4. keep the complex sample over time for that bin.

Important inference:

If you only operate at a slow-time rate close to the TI demo's `20 Hz`, your beat timing resolution will be limited. For IBI work, use the raw capture path and configure a sufficiently dense slow-time sampling rate so that beat peaks can be localized much more precisely than `50 ms`.

### Step 3: phase-domain preprocessing

Recommended order:

1. `phi(t) = atan2(Q(t), I(t))`
2. phase unwrap
3. drift removal
4. impulsive spike suppression
5. motion gating
6. optional phase difference

Practical choices:

- median filter for baseline drift
- Hampel / threshold spike removal for impulsive unwrap errors
- 1-second motion energy gating inspired by TI vital-signs lab

### Step 4: heartbeat extraction

Preferred method:

- adaptive decomposition of the cleaned phase signal into a small set of band-limited modes;
- pick the mode consistent with heartbeat amplitude/frequency behavior;
- normalize;
- detect peaks.

Fallback method:

- heartbeat band-pass filter;
- respiration harmonic suppression;
- detect peaks with physiological constraints.

Comparison method:

- cepstrum-based IBI extraction over multiple windows.

### Step 5: beat detection and IBI

After obtaining a heartbeat waveform:

- detect local maxima with a minimum inter-peak spacing based on plausible HR range;
- reject implausible short/long intervals;
- refine peak timing by local interpolation around the maximum if needed;
- compute `IBI_i = t_{i+1} - t_i`.

### Step 6: NN cleaning before HRV

Do not compute HRV directly from raw IBI.

Apply a cleaning pass:

- reject intervals in motion-corrupted segments;
- reject intervals with impossible physiology;
- reject intervals deviating strongly from local median or local trend;
- interpolate only if you need a continuous sequence for a downstream frequency-domain step.

For primary time-domain HRV metrics, it is acceptable to keep only valid intervals and compute metrics on the valid set.

### Step 7: five-minute HRV metrics

Primary metrics:

- mean IBI
- mean HR
- RMSSD
- SDRR / SDNN

Secondary metric:

- pNN50

Why pNN50 is secondary:

- some radar IBI pipelines are especially sensitive to duplicated / missed beats,
- pNN50 can become unstable even when RMSSD and SDRR still look reasonable.

## What I would implement first, second, third

### Milestone 1: prove the signal chain

Goal:

- show a stable chest phase signal from a fixed subject;
- show visible respiration;
- show at least a weak heartbeat-related waveform.

Implement:

- raw capture parsing
- range FFT
- target bin selection
- phase extraction / unwrap
- simple filtering

### Milestone 2: get a real IBI baseline

Goal:

- produce beat timestamps and IBI on clean segments.

Implement:

- motion gating
- simple BPF heartbeat baseline
- peak detection
- IBI sequence

### Milestone 3: upgrade to the real project algorithm

Goal:

- move from "some IBI" to "defensible IBI/HRV."

Implement:

- mmHRV-style decomposition extractor
- NN cleaning
- five-minute HRV metrics
- comparison against ECG/PPG

### Milestone 4: ablation and fallback

Compare:

- simple BPF baseline
- mmHRV main model
- cepstrum fallback

This will make your final report much stronger.

## What not to do

- Do not rely only on TI demo-level heart-rate outputs and claim HRV from that.
- Do not use only frequency-peak estimation and call it IBI recovery.
- Do not skip synchronization planning with the reference sensor.
- Do not wait to discover motion problems after full data collection.
- Do not assume the best range bin is always just the strongest-magnitude bin without checking periodicity quality.

## Suggested team decisions to make now

1. Decide the reference device now:
   ECG preferred, PPG fallback.
2. Decide the exact capture environment now:
   seated vs lying, distance, mounting, angle.
3. Decide the exact chirp/frame configuration now:
   enough slow-time resolution for beat timing, not just rate estimation.
4. Decide the first algorithm now:
   simple BPF baseline first, mmHRV hybrid as main path.
5. Decide the evaluation metrics now:
   mean absolute IBI error, median absolute IBI error, RMSSD error, SDRR/SDNN error, beat matching score.

## Final recommendation

If the question is "what model should we actually use first for this final project?", my answer is:

- main model: `TI preprocessing + mmHRV heartbeat extractor`
- first baseline: `simple heartbeat BPF + peak detection`
- fallback/comparison model: `cepstrum-based IBI extraction`
- preprocessing ideas to borrow: `DR-MUSIC median detrending + respiration harmonic suppression`
- evaluation discipline to borrow: `Oura-style normal-IBI filtering over five-minute windows`

That is the most defensible, implementable, and project-aligned answer I found.

## Sources

- TI IWR1443BOOST user guide:
  https://www.ti.com/lit/ug/swru518d/swru518d.pdf
- TI DCA1000EVM tool page and docs:
  https://www.ti.com/tool/DCA1000EVM
- TI mmWave Studio user guide:
  https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-oTIAkD3TFJ/04.03.01.00/mmwave_studio_user_guide.pdf
- TI Vital Signs xwr1443 user guide:
  https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/1023/5684.Vital_5F00_Signs_5F00_Lab_5F00_User_5F00_Guide_5F00_v1.1.pdf
- TI Vital Signs xwr1443 developer guide:
  https://e2e.ti.com/cfs-file/__key/communityserver-discussions-components-files/1023/Vital_5F00_Signs_5F00_xwr1443_5F00_Developers_5F00_Guide.pdf
- ICASSP 2021 mmHRV short paper:
  https://xiaolu1263.github.io/files/ICASSP2021HRV.pdf
- mmHRV journal paper:
  https://xiaolu1263.github.io/files/2021-mmHRV.pdf
- Scientific Reports 2024 preprocessing paper:
  https://www.nature.com/articles/s41598-024-77683-1
- Nature Communications 2024 RF-HRV:
  https://www.nature.com/articles/s41467-024-55061-9
- PMC mirror for RF-HRV:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11621424/
- FMCW radar in sleeping scenarios:
  https://www.mdpi.com/1424-8220/20/22/6505
- Oura HRV comparison paper:
  https://ouraring.com/blog/wp-content/uploads/2018/10/The-HRV-of-the-Ring-Comparison-of-OURA-Ring-to-ECG.pdf
