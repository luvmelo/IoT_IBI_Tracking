# 6.1820 Final Project Proposal (Revised)

## Working Title

Contactless Interbeat Interval Tracking with TI IWR1443 mmWave Radar

## 1. Problem Statement

Interbeat Interval (IBI) is the precise time, in milliseconds, between consecutive heartbeats. Accurate IBI estimation is the foundation of Heart Rate Variability (HRV), which is widely used as a non-invasive indicator of autonomic nervous system activity, stress, fatigue, and cardiovascular health.

Today, the most reliable IBI and HRV measurements come from contact sensors such as ECG or PPG. These methods are effective, but they are inconvenient for long-term and unobtrusive monitoring. Millimeter-wave (mmWave) radar is a promising alternative because it can sense tiny chest displacements without physical contact. However, IBI tracking is much harder than average heart-rate estimation: instead of estimating only a dominant frequency, the system must recover the timing of individual beats accurately enough to measure beat-to-beat variation.

This project proposes a proof-of-concept contactless IBI tracking system using the TI IWR1443 radar and DCA1000 capture board. The goal is not to build a product-level or clinical-grade monitor. The goal is to reproduce, in a controlled single-subject setting, whether commodity mmWave hardware can recover beat-level timing accurately enough to estimate IBI and derive short-window HRV metrics over five-minute recordings.

## 2. Research Question

Can a TI IWR1443 + DCA1000 mmWave setup recover heartbeat timing accurately enough, in a controlled single-user stationary setting, to estimate IBI and short-window HRV metrics over five-minute intervals?

## 3. Scope

### In scope

- One seated or lying subject at fixed short distance from the radar.
- Controlled indoor environment with minimal gross body motion.
- Raw ADC capture using TI tooling and DCA1000.
- Range-bin selection, phase extraction, heartbeat recovery, and IBI estimation.
- Comparison against a contact-based reference signal.
- HRV metrics such as mean IBI, RMSSD, and SDRR/SDNN.

### Out of scope

- Multi-user tracking.
- Through-wall sensing.
- Clinical validation.
- Real-time embedded deployment on the radar board.
- Product-level robustness across broad posture, clothing, and room variations.
- Human pose estimation as a primary task.

## 4. Motivation and Significance

This project is interesting because it sits at a harder layer of radar-based vital-sign sensing than standard heart-rate estimation. Many mmWave systems can recover an average heart-rate value, but IBI and HRV require substantially finer temporal precision. That makes this project a more meaningful test of whether the radar is truly resolving heartbeat micro-motion rather than just detecting coarse spectral peaks.

The project is also a good fit for a final course project because the hardware stack is available, TI provides a documented raw-data capture path, and prior research already suggests that mmWave can support HRV estimation in controlled conditions. That means the main contribution here is not inventing a new sensing modality, but engineering a reliable acquisition and processing pipeline and evaluating whether the claims from prior work hold up on our own setup.

## 5. Technical Approach

### 5.1 Hardware and data path

We will use:

- TI IWR1443BOOST radar board.
- TI DCA1000EVM capture board.
- Host laptop or PC for capture and offline processing.
- A contact-based reference sensor, ideally ECG; if unavailable, synchronized PPG as fallback.

The plan is to use the DCA1000 path to capture raw radar data for offline analysis. This avoids relying only on demo-level processed outputs and gives us direct control over the signal-processing pipeline.

### 5.2 Signal processing pipeline

The baseline pipeline is:

1. Configure the radar and collect raw data from a stationary subject.
2. Perform range FFT and select the chest range bin.
3. Extract slow-time complex phase from the selected range bin.
4. Unwrap phase and suppress static clutter.
5. Reduce baseline drift and motion artifacts.
6. Separate respiration and heartbeat components.
7. Detect heartbeat peaks from the recovered heartbeat waveform.
8. Convert beat-to-beat timing into IBI estimates.
9. Compute HRV metrics from the IBI sequence.
10. Compare against contact-based reference measurements.

### 5.3 Algorithmic baseline

The most relevant baseline is the mmHRV line of work. The ICASSP 2021 short paper introduces a heartbeat extractor that decomposes the phase signal into band-limited components and identifies the component consistent with heartbeat dynamics. Peaks in this recovered waveform are then used to estimate IBI. The later IEEE Internet of Things Journal version extends the same direction with broader validation.

For this project, we will focus only on the single-user controlled version of that idea:

- phase-based chest-motion extraction,
- respiration and heartbeat separation,
- peak-based beat timing recovery.

If the decomposition-style pipeline proves too brittle for our recordings, the fallback is a simpler but more robust baseline:

- narrow band-pass filtering around heartbeat,
- adaptive respiration suppression,
- heuristic peak picking with physiological timing constraints.

This fallback may reduce final accuracy, but it keeps the project feasible.

## 6. Evaluation Plan

### Dataset collection

- Record multiple five-minute trials from one subject in a quiet indoor setting.
- Keep subject distance and orientation fixed for the baseline recordings.
- If time permits, add a few recordings with mild posture variation or controlled disturbance.

### Metrics

- Mean absolute IBI error.
- Median absolute IBI error.
- Mean IBI error.
- RMSSD error.
- SDRR/SDNN error.
- Beat detection agreement within a fixed timing tolerance window.

### Success criterion

The project does not need clinical-grade performance to be successful. It is enough to show that:

- the radar consistently recovers a visible heartbeat-related waveform,
- beat-level peaks can be detected over multi-minute recordings,
- the resulting IBI sequence tracks the reference sequence reasonably well,
- short-window HRV metrics are directionally consistent with the reference.

## 7. Expected Challenges

### Motion contamination

IBI estimation is highly sensitive to body motion. Even small posture changes can overwhelm heartbeat-scale motion.

### Respiration leakage

Respiration amplitude is much larger than heartbeat amplitude, so leakage from the respiratory component may distort beat timing even when average heart rate looks plausible.

### Hardware and software friction

The IWR1443 is an older board and its tooling is less forgiving than newer TI platforms. The local course notes already indicate that some workflows are fragile and version-sensitive.

### Ground-truth synchronization

IBI comparison requires reliable timing alignment between radar recordings and the contact reference. Poor synchronization can degrade the evaluation even if the radar-side algorithm is reasonable.

## 8. Risk Mitigation and Fallback Plan

If raw-data capture with DCA1000 proves unstable, we will fall back to TI vital-sign demo outputs or saved binaries for offline analysis.

If precise IBI estimation is too noisy, we will narrow the project deliverable to:

- robust heartbeat waveform extraction,
- heart-rate estimation,
- exploratory IBI recovery on the cleanest segments.

If synchronization with the reference device is weak, we will still report waveform quality and average-rate agreement, but clearly state the limitation.

## 9. 4-Week Execution Plan

### Week 1: Hardware bring-up and capture verification

- Set up IWR1443 and DCA1000.
- Verify board flashing and raw capture path.
- Produce one short recording and parse it offline successfully.

### Week 2: Preprocessing and heartbeat extraction

- Implement range-bin selection, phase extraction, phase unwrapping, and clutter suppression.
- Recover a stable slow-time waveform from the chest target.
- Implement at least one heartbeat extraction pipeline.

### Week 3: IBI estimation and data collection

- Record multiple five-minute trials.
- Implement beat detection and IBI estimation.
- Compute HRV metrics from radar-derived IBIs.
- Align with the contact reference and quantify error.

### Week 4: Comparison, analysis, and report

- Compare radar estimates with the reference.
- Analyze failure modes such as motion, respiration leakage, and range-bin instability.
- Produce final plots, tables, and the written report.

## 10. Deliverables and Metrics

### Deliverables

- A working IWR1443 + DCA1000 capture setup.
- A reproducible offline parsing and signal-processing pipeline.
- Plots showing raw phase, cleaned phase, and recovered heartbeat waveform.
- A beat detector that outputs an IBI sequence.
- Comparison plots and tables against the contact-based reference.
- A final report describing methods, results, limitations, and lessons learned.

### Metrics

- Mean absolute heart-rate error in BPM.
- Mean absolute IBI error in milliseconds.
- Median absolute IBI error in milliseconds.
- RMSSD error.
- SDRR/SDNN error.
- Percentage of correctly recovered beats within a tolerance window.

### Minimum success criteria

The project will be considered successful if:

- raw radar capture works reliably enough for offline processing,
- heartbeat-related phase structure is visible on clean recordings,
- an estimated IBI sequence can be recovered for at least some five-minute trials,
- the recovered IBI sequence shows reasonable agreement with the contact reference.

## 11. Resources Needed

### Hardware

- TI IWR1443BOOST.
- TI DCA1000EVM.
- Stable power supply and cables for both boards.
- Host laptop or PC for capture and offline analysis.
- Contact reference sensor, ideally ECG, otherwise synchronized PPG.

### Software

- TI mmWave Studio and related drivers/tools.
- Python or MATLAB environment for offline parsing and analysis.
- Plotting and numerical processing libraries for evaluation.

### Data collection setup

- Quiet indoor environment.
- Fixed chair or bed setup for consistent subject placement.
- A simple procedure for synchronizing radar capture and reference recording.

## 12. Itemized Equipment List

### From course or lab inventory

- 1 x TI IWR1443BOOST.
- 1 x TI DCA1000EVM.

### Additional practical items

- 1 x 5V power adapter compatible with the radar setup.
- 1 x Ethernet cable for DCA1000 data path.
- 1 x micro-USB cable for board communication.
- 1 x host laptop or desktop.
- 1 x reference pulse or ECG device.
- 1 x tripod, stand, or stable mounting arrangement.

## 13. Literature Review and Relevance

### Directly relevant papers

1. **Fengyu Wang et al., "Radio Frequency Based Heart Rate Variability Monitoring," ICASSP 2021.**  
   This is the closest short-paper baseline for the class project. It directly addresses HRV from mmWave and recovers IBI through heartbeat waveform extraction and peak localization.

2. **Fengyu Wang et al., "mmHRV: Contactless Heart Rate Variability Monitoring Using Millimeter-Wave Radio," IEEE Internet of Things Journal, 2021.**  
   This is the stronger journal baseline and reports single-user and broader-condition evaluations for mmWave-based IBI estimation.

3. **Yuanchang Chen et al., "A high precision vital signs detection method based on millimeter wave radar," Scientific Reports, 2024.**  
   This paper is more relevant for preprocessing and respiration-suppression ideas than for direct IBI estimation, but it is still useful as a signal-processing reference.

4. **Bin-Bin Zhang et al., "Monitoring long-term cardiac activity with contactless radio frequency signals," Nature Communications, 2024.**  
   This is a much larger-scale system paper, but it is important evidence that contactless RF-based IBI and HRV tracking is now credible and active research.

### Less relevant papers

The pose-estimation papers listed earlier are not central to this project. They may inspire future work on motion rejection, but they should not sit on the critical path of an IBI tracking proposal.

## 14. Public Papers and Public Code

### Publicly accessible papers

- The ICASSP 2021 mmHRV paper PDF is publicly accessible.
- The 2024 Scientific Reports paper is open access.
- The 2024 Nature Communications paper is open access.

### Publicly accessible code and tooling

- TI provides official IWR1443 and DCA1000 documentation.
- TI engineers indicate that Vital Signs Lab source and MATLAB-side tooling are available through TI resources.
- Public projects such as `KylinC/mmVital-Signs` provide useful engineering references, although they are not direct reproductions of mmHRV-style IBI extraction.

### What appears to be missing

I did not find an official public implementation of the mmHRV papers. Therefore the likely workflow remains:

- use TI’s official hardware and capture path,
- implement the key signal-processing stages ourselves,
- borrow only engineering ideas from public repositories where useful.

## 15. Second-Pass Feasibility Check

This is a second-pass check after restoring the proposal to the original IWR1443 + DCA1000 + IBI scope.

### What looks solid

- The hardware path is coherent:
  IWR1443BOOST plus DCA1000 is the standard TI route for raw ADC capture.
- The research question is narrow enough for a class project:
  single user, controlled setting, offline processing, five-minute windows.
- The literature support is strong:
  mmHRV and later RF-HRV papers directly support the claim that IBI estimation is possible in controlled settings.

### What remains risky

- The IWR1443 tooling is old and may be brittle.
- Beat-level timing recovery is substantially harder than heart-rate estimation.
- Final quality depends heavily on motion control and reference synchronization.

### Recommendation after the second check

This proposal is still viable and stronger than the later sleep-monitoring variant because it has:

- tighter scope,
- clearer literature alignment,
- more defensible evaluation metrics,
- fewer weakly supported side tasks.

## 16. Revised Proposal Summary

This project proposes a focused reproduction study on contactless IBI tracking with mmWave radar. Using TI’s IWR1443 and DCA1000 hardware, we will build an offline signal-processing pipeline that extracts chest-motion phase, suppresses respiration and motion interference, detects heartbeat peaks, and estimates beat-to-beat intervals over five-minute recordings. We will validate the recovered IBI sequence and derived HRV metrics against a contact-based reference signal. The project is intentionally scoped to a single-user, mostly stationary setting and does not aim for product-level robustness. Its main contribution is to test whether commodity mmWave hardware, combined with a careful phase-based pipeline inspired by mmHRV, is sufficient for reproducible IBI estimation in a realistic course-project setting.

## 17. Sources

- TI IWR1443BOOST User Guide: https://www.ti.com/lit/ug/swru518d/swru518d.pdf
- TI DCA1000EVM Quick Start Guide: https://www.ti.com/lit/ml/spruik7/spruik7.pdf
- Local course notes: `Copy of 6.1820 Using IWR1443 + DCA1000 Notes-1.pdf`
- ICASSP 2021 mmHRV paper PDF: https://xiaolu1263.github.io/files/ICASSP2021HRV.pdf
- mmHRV journal metadata/abstract: https://pure.bit.edu.cn/en/publications/mmhrv-contactless-heart-rate-variability-monitoring-using-millime
- Scientific Reports 2024 vital-signs paper: https://www.nature.com/articles/s41598-024-77683-1
- Nature Communications 2024 RF-HRV paper: https://www.nature.com/articles/s41467-024-55061-9
- Open-source mmWave vital-signs repo: https://github.com/KylinC/mmVital-Signs
- TI forum on Vital Signs Lab source availability: https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/809402/ccs-iwr1443-lab0002-vital-signs-source-code
- TI forum on saved binary parsing script: https://e2e.ti.com/support/sensors-group/sensors/f/sensors-forum/732538/iwr1443boost-iwr1443-vital-signs-data
