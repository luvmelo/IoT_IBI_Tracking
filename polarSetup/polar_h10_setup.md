## 1. Software setup

###  macOS

1. Install Python 3.9 or newer (Anaconda, Homebrew, or python.org installer all work).
2. Install bleak:
   ```bash
   pip3 install --user bleak
   ```
3. The first time you run any BLE script, macOS will pop up a dialog asking
   for Bluetooth permission for the terminal application (Terminal.app,
   iTerm2, VSCode, etc). Approve it. If you skip it, scans return zero
   devices with no error message.
4. Verify NTP sync:
   System Settings → General → Date & Time → "Set time and date automatically"
   should be on.

### Windows

1. Install Python 3.10 or newer (64-bit) from python.org or the Microsoft
   Store.
2. From an Administrator PowerShell:
   ```powershell
   py -m pip install --upgrade pip
   py -m pip install bleak
   ```
3. Enable Bluetooth: Settings → Bluetooth & devices → Bluetooth = On.
4. Optional but recommended: pair the strap once via
   Settings → Bluetooth & devices → Add device → Bluetooth → choose
   "Polar H10 ...". After pairing, you can unpair if you prefer; bleak will
   rediscover it. Pre-pairing helps on some Intel/Realtek BT adapters that
   are slow to enumerate unpaired peripherals.
5. Verify NTP sync: Settings → Time & language → Date & time → "Sync now".

---

## 2. Recording a trial

### 5.1. macOS

```bash
cd /path/to/IoT_IBI_Tracking
python3 scripts/polar_h10_record_macos.py
```

### 5.2. Windows

```powershell
cd C:\path\to\IoT_IBI_Tracking
py scripts\polar_h10_record_windows.py
```

### 5.3. The interactive flow (identical on both)

1. The script prompts for a **trial label** (e.g. `subject01_d07_a00_rest`).
   Use the experiment-matrix convention: `<subject>_d<distance_cm>_a<angle_deg>_<state>`.
   Press Enter to skip the label.
2. The script scans for ~15 seconds. When it finds the strap, it connects.
3. The script collects ~2 stabilization packets, then prints `READY`.
4. **Synchronize start with the mmWave laptop.** When both operators are
   ready, press Enter on the Polar laptop and start the radar capture
   simultaneously. Sub-second alignment is good enough — the timestamp-based
   beat matching in `radar_analysis/` will refine the alignment using the
   per-beat `t_epoch_s` field.
5. The script records for `RECORDING_DURATION_S` seconds (default 60). Edit
   that constant at the top of the script for longer trials.
6. When recording finishes, the script writes:
   - `recordings/polar_h10_<timestamp>.csv` — the IBI data.
   - `recordings/polar_h10_<timestamp>.meta.txt` — sidecar with start/end ISO
     timestamps, epoch values, trial label, platform, and device info.
7. A summary HRV report is printed: beat count, mean HR, mean IBI, SDNN,
   RMSSD, pNN50.

---

## 3. CSV schema

Both scripts produce a single CSV per trial with the following columns:

| Column | Meaning |
|---|---|
| `t_perf_s`  | Monotonic seconds since recording start (`time.perf_counter`). Use for plotting within a single trial. |
| `t_epoch_s` | Unix epoch seconds (float, ms precision). Use for cross-device alignment with the radar. |
| `hr_bpm`    | Heart rate as reported by the strap in that BLE notification. |
| `ibi_ms`    | One RR / IBI interval in milliseconds. |

**One row per RR interval, not per BLE packet.** A single BLE notification can
carry 1–4 RR intervals (the strap buffers them between transmissions); each
RR is exploded into its own row with the same `t_perf_s` / `t_epoch_s` of the
notification arrival. This matches the Polar Sensor Logic SDK's CSV format
and is what downstream beat-pair matching code expects.

A representative row from a real trial in this project:

```
t_perf_s,t_epoch_s,hr_bpm,ibi_ms
0.872,1778008089.828,79,780.3
1.871,1778008090.827,79,779.3
1.871,1778008090.827,79,751.0
2.870,1778008091.826,79,682.6
```

---

## 4. Time synchronization with the radar

The validity of the IBI MAE comparison hinges on the radar laptop and the
Polar laptop sharing wall-clock time. Recommended procedure:

1. Both machines must have NTP sync turned on (see §3 above).
2. Verify before each session: open a terminal on each laptop and run
   `date` (macOS/Linux) or `Get-Date -Format o` (PowerShell). The two
   wall-clock outputs should agree within ~100 ms.
3. The mmWave pipeline writes per-frame epoch timestamps. The Polar pipeline
   writes per-beat epoch timestamps. Beat-pair matching in `radar_analysis/`
   does a search-window match around each Polar beat; a window of ±150 ms
   handles residual NTP drift and any radar-pipeline latency.

For a defensible sanity check — recommended once at the start of a session
and once at the end — have the subject perform a **deliberate breath-hold**
of ~15 seconds at a known time during the trial:

- The breath-hold appears as a flatline in the radar respiration channel.
- It produces a characteristic HRV depression and rebound in the Polar IBI
  series.
- Aligning these two features verifies sync to ~50 ms without any extra
  hardware.

---

## 5. Troubleshooting

### "Polar H10 not found"

- The strap is dry — wet the electrodes and re-seat the pod on the strap.
- Your phone is still connected to it — kill Polar Beat / Polar Flow.
- The pod battery is dead — pop it out and check (CR2025 coin cell).
- (Windows only) The OS Bluetooth stack is in a bad state — toggle Bluetooth
  off and on in Settings.

### Recording starts but no IBI rows appear

- Heart rate is being reported but RR is not, which means the flags byte in
  the notification has bit 4 cleared. This is rare. The strap firmware
  reports RR by default. Restart the strap by removing and re-snapping the
  pod.

### IBI values look wrong (e.g. consistently 1500+ ms)

- The strap is missing every other beat, usually because of poor electrode
  contact. Re-wet the electrodes generously, reposition the strap, and try
  again. Hairy chests may need a thin layer of conductive gel rather than
  just water.

### Recording works but timestamps drift relative to the radar

- Your two laptops are not actually NTP-synced. Re-check §3 and §7. On
  macOS in particular, "Set time and date automatically" can silently fail
  if the system was offline at the last sync attempt.

### Different signal between sessions on the same subject

- Strap position varies day to day. For longitudinal trials, mark the strap
  position once on the subject (a small pen mark on the skin is fine for
  same-day re-application).

---

## 6. Reference

> "RR intervals from a Polar H10 chest strap (Polar Electro Oy, Kempele,
> Finland) were used as the reference signal for validating radar-derived
> IBI."

The strap's own internal R-peak detection is well-validated against clinical
3-lead ECG (see Gilgen-Ammann et al., 2019, *Eur. J. Appl. Physiol.*) but is
not itself a regulated medical device.
