# IWR1443 + DCA1000 Python Interface — Setup Guide

This is a working guide for getting raw radar frames into Python, based on the
notes PDF and the recommended reference implementation at
[r-bt/MakeyMakey/src/xwr](https://github.com/r-bt/MakeyMakey/tree/main/src/xwr).

## How the pieces fit together

```
IWR1443 BOOST  --LVDS-->  DCA1000 EVM  --UDP/Ethernet-->  Host PC (Windows)
     ^                                                          |
     |                                                          v
  COM ports   <----  mmWave Studio (Lua scripts)  ----->  Python listener
                                                          (UDP socket)
```

- **mmWave Studio** is only used to configure and arm the radar over COM (it
  loads firmware, programs the chirp/frame parameters, tells the DCA1000 to
  start streaming).
- Once `StartFrame()` runs, the **DCA1000 streams raw ADC samples over UDP**
  on `192.168.33.180:4098` to the host (default `192.168.33.30:4098`).
- A Python script binds that UDP port, reassembles packets into frames, and
  reshapes them into `(n_chirps, n_samples_per_chirp, n_rx)`.
- mmWave Studio is **only required on Windows**. The Python side can run on
  any OS that can bind to the host static IP, but in practice the easiest
  path is to do everything on the same Windows machine.

## What the team's `.ini` tells us

The GuiVersion=2.1.1.0 file your groupmates produced is just **mmWave
Studio's saved GUI state**, not capture parameters. Useful facts confirmed
by it:

| Field | Value | Meaning |
|---|---|---|
| `GuiVersion` | `2.1.1.0` | mmWave Studio `02_01_01_00` installed (matches PDF) |
| `BoardType` | `1` | IWR1443 (xwr12xx/14xx firmware paths) |
| `ComPort` | `3` | UART control on **COM3** |
| `BaudRate` | `921600` | Standard mmWave Studio baud |
| `PhyFwFile` / `MacFwFile` | `C:\ti\mmwave_studio_02_01_01_00\...\xwr12xx_xwr14xx_*.bin` | Confirms install path |
| `TDA2xx_IPAddress` | `0.0.0.0` | **Not** using TDA capture board → using DCA1000 (good) |

**The radar/chirp/frame settings are NOT in this file.** They live in a Lua
script you run from mmWave Studio's *Shell* tab.

## Prerequisites (Windows, one-time)

Per the PDF notes:

1. mmWave SDK 2.1.0 LTS — installed already if `C:\ti\mmwave_studio_02_01_01_00\`
   exists.
2. mmWave Studio 2.1.1 — installed.
3. **32-bit MATLAB Runtime R2013a (8.5.1)** — required, easy to forget.
   Without it the Radar API tab is blank.
4. Microsoft Visual C++ 2013 redistributable.
5. Set the host's Ethernet adapter (the one wired to the DCA1000) to **static
   IP `192.168.33.30`, mask `255.255.255.0`**, no gateway.
6. Confirm jumpers on the IWR1443 are in **dev/raw-capture mode**: SOP0 + SOP1
   set, SOP2 cleared. (Flash mode is SOP0+SOP2; functional/Demo Visualizer
   mode is SOP0 only.)
7. Three USB cables to the IWR1443 (FTDI for RADAR\_FTDI, RADAR\_COM_PORT, and
   the third), one Ethernet from DCA1000 to host, 5 V/2 A barrel jack to DCA1000.

## Step 1 — Get the reference Python code

```bash
git clone https://github.com/r-bt/MakeyMakey.git
cd MakeyMakey
# Python 3.10+ required (per pyproject.toml)
pip install numpy numba pyserial
# Optional, only if you want their live FFT GUIs:
pip install pyqt6 pyqtgraph matplotlib opencv-python scikit-learn
```

Files that matter for the radar interface (under `src/xwr/`):

- `dca1000.py` — UDP socket client. Sends FPGA config commands (`0x5a 0xa5 ...`)
  to `192.168.33.180:4096`, receives frames on host port `4098`. Each UDP
  payload starts with `<I seqnum><I bytecount><2 bytes pad>` then raw int16
  IQ samples.
- `frame_buffer.py` — numba `@jitclass` ring buffer that zero-pads on dropped
  packets (detected via sequence-number gaps). Returns a complete frame as
  an int16 view when one is full.
- `radar_config.py` — parses the `.lua` file with regex to extract
  `ADC_SAMPLES`, `CHIRP_LOOPS`, `START_FREQ`, etc., and computes derived
  params (`frame_size`, `range_max`, `velocity_max`, ...).
- `dcapub.py` — wires the three pieces together; allocates a buffer of
  `2 × frame_size` so a frame is always contiguous.
- `dsp.py` — single function `reshape_frame()` that turns the int16 stream
  into `(n_chirps, n_samples, n_rx)` complex128. **This is the IWR1443-specific
  bit:** it does `data.reshape(-1, 8); data[:, :4] + 1j*data[:, 4:]`, i.e.
  4 RX antennas, I-then-Q layout per LVDS lane group. Don't blindly reuse
  the upstream `xwr_raw_ros` reshape — it's for AWR2243/cascade boards.
- `src/radar.py` — top-level `Radar(cfg_path)` class. Use this; don't talk to
  `DCAPub` directly.

## Step 2 — Pick / adapt a Lua config

Save this as `1443_capture.lua`. It is the MakeyMakey
[`1443_mmwavestudio_config_old.lua`](https://github.com/r-bt/MakeyMakey/blob/main/scripts/1443_mmwavestudio_config_old.lua)
with the **two fields you must change**:

```lua
COM_PORT = 3                          -- was 9; matches your .ini
SAVE_DATA_PATH = "C:\\Users\\<you>\\data\\adc_data.bin"
DUMP_DATA_PATH = "C:\\Users\\<you>\\data\\adc_data_RAW_0.bin"
PKT_LOG_PATH   = "C:\\Users\\<you>\\data\\pktlogfile.txt"
```

Capture params worth knowing (these are what `RadarConfig.get_params()` will
read):

| Var | Value | Effect |
|---|---|---|
| `START_FREQ` | 77 GHz | |
| `IDLE_TIME` + `RAMP_END_TIME` | 138 + 62 = 200 µs | chirp period |
| `FREQ_SLOPE` | 60.012 MHz/µs | |
| `ADC_SAMPLES` | 512 | samples per chirp |
| `SAMPLE_RATE` | 10000 ksps | 10 MHz |
| `CHIRP_LOOPS` | 255 | chirps per frame |
| `START_CHIRP_TX` / `END_CHIRP_TX` | 0 / 0 | **single TX** (not TDM) |
| `NUM_FRAMES` | 0 | continuous stream — set non-zero to capture a fixed count |
| `PERIODICITY` | 100 ms | one frame every 100 ms |

With these defaults you get: range res ≈ 4.9 cm, max range ≈ 25.6 m, max
velocity ≈ ±9.7 m/s, frame size = 512 × 4 × 255 × 2 × 2 = 2,088,960 bytes.

> The script uses `END_CHIRP_TX = 0` so only TX1 fires — even though the Lua
> notes mention "Tx1→Tx3→Tx2" ordering, that ordering only kicks in if you
> change the chirp/frame config. Stay with `END_CHIRP_TX = 0` unless you're
> ready to deinterleave TDM in `dsp.py`.

The bottom of the script calls `ar1.StartFrame()` automatically, so as soon
as you `dofile()` it, the radar is streaming. To stop, run
`ar1.StopFrame()` from the Shell tab (or power-cycle).

## Step 3 — Run the capture

1. Power up the IWR1443 + DCA1000.
2. Launch **mmWave Studio as Administrator** (admin is required for it to
   bind raw sockets to the DCA1000).
3. Open the **Shell** tab, run `dofile("C:\\path\\to\\1443_capture.lua")`.
   Watch for "Sensor Start" / "FRAME\_START\_ASYNC\_EVENT". If you see
   *"MSS Power Up async event was not received"* → re-flash the
   `xwr14xx_mmw_demo.bin` with Uniflash (PDF page 4) and try again.
4. In a separate terminal (same machine), run the Python side:

   ```bash
   cd MakeyMakey
   python record.py --cfg scripts/1443_capture.lua          # writes data/radar_data_*.npz
   # or for a live distance plot:
   python fft_live.py --cfg scripts/1443_capture.lua
   ```

5. `Ctrl-C` to stop. `record.py` then saves an `.npz` with shape
   `(n_frames, 255, 512, 4)` complex128.

### Minimal "just give me one frame" snippet

```python
from src.radar import Radar

radar = Radar("scripts/1443_capture.lua")
print(radar.params)        # range_res, velocity_max, frame_size, ...

frame = radar.read()        # one (255, 512, 4) complex frame
radar.close()
```

## Known gotchas (from the PDF "Side notes" + repo)

1. **Studio must be Admin.** Otherwise capture commands silently fail.
2. **Order matters.** Start the Lua script *before* the Python listener, but
   `radar.run_polling()` calls `flush_data_socket()` first, so a few seconds
   of skew is fine.
3. **DCA1000 in continuous mode occasionally stalls.** Per the team that
   wrote MakeyMakey, mid-capture stops happen "sometimes" with no clean
   reason — restart the Lua script if the Python side stops receiving.
4. **MATLAB toolbox doesn't work for IWR1443.** Don't waste time trying TI's
   MATLAB scripts; only the Lua + Python path works.
5. **The TI Demo Visualizer path is unrelated** to this Python flow. Demo
   Visualizer talks to the IWR1443 directly over UART after the demo
   firmware is flashed; it's only useful for the "is the board alive?" sanity
   check on PDF pages 2–3. Once you switch to raw capture you stop using it.
6. **One TX, four RX.** `dsp.reshape_frame` assumes 8-int16 lanes per LVDS
   word (4 RX × I+Q). If you ever enable TX2/TX3 (TDM-MIMO), the function
   needs the commented-out deinterleave block re-enabled.

## Quick verification checklist

- [ ] `ping 192.168.33.180` from host succeeds → DCA1000 sees the host IP.
- [ ] mmWave Studio "Connect" lights green on the Radar API tab with COM3.
- [ ] Lua script prints non-zero `Chirps Per Frame` / `Range Resolution`.
- [ ] Python prints `[INFO] Radar connected` followed by params and starts
      receiving with no `Packet drop` spam.
- [ ] After ~1 s, frames arrive at PERIODICITY (10 Hz with default config).

## Where to look next

- IWR1443 raw 4-channel capture format → PDF section 24.6 (page 5 of the
  notes), and TI mmWave SDK doc `MmwaveCapture.pdf`.
- Lua API reference → `C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\Scripts\`
  has examples plus `RadarStudio_API.html`.
- Upstream of MakeyMakey:
  [ConnectedSystemsLab/xwr_raw_ros](https://github.com/ConnectedSystemsLab/xwr_raw_ros/tree/main/src/xwr_raw)
  — useful if you want to understand why the buffer/packet code looks the
  way it does, but its `dsp.py` is for different boards.
