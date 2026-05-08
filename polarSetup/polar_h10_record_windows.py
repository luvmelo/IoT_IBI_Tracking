"""
polar_h10_record_windows.py
===========================

Synchronized timestamped Polar H10 RR-interval recorder for Windows.

Functionally identical to polar_h10_record_macos.py, but adapted for the
Windows BLE stack (WinRT backend in bleak). Outputs CSVs in the same schema
so the same downstream analysis works on either platform.

Why a separate Windows script?
------------------------------
1. Address format. macOS / CoreBluetooth uses 128-bit UUIDs as device
   identifiers. Windows / WinRT uses 48-bit MAC addresses (e.g.
   "AA:BB:CC:DD:EE:FF"). The user-facing "address" string differs, so the
   meta file records that explicitly for traceability.
2. Permissions. Windows requires Bluetooth to be enabled at OS level. There
   is no terminal-permission prompt like macOS shows.
3. Pairing model. On Windows it sometimes helps to pair the H10 in
   Settings > Bluetooth & devices BEFORE first run; on macOS pairing is
   handled implicitly by bleak.
4. Loop policy. On Windows + Python 3.8+, asyncio defaults to
   ProactorEventLoop, which is fine for bleak's WinRT backend, but we
   request it explicitly so behaviour is predictable across Python versions.

Setup (one-time)
----------------
1. Install Python 3.10 or later (64-bit) from python.org or the Microsoft
   Store.
2. From an Administrator PowerShell:
       py -m pip install --upgrade pip
       py -m pip install bleak
3. Enable Bluetooth in Windows Settings.
4. Wear the Polar H10 chest strap with electrodes moistened.
5. Disconnect any phone/watch currently paired with the strap (the H10
   only streams to one client at a time).
6. Optional but recommended: pair the strap once via
   Settings > Bluetooth & devices > Add device > Bluetooth > "Polar H10 ...".
   After pairing, you can unpair if you prefer; bleak will rediscover it.
7. Verify your PC's clock is NTP-synchronized (Settings > Time & language >
   Date & time > "Sync now"). Time alignment with the mmWave laptop is what
   makes the IBI ground-truth comparison valid.

Usage
-----
    py polar_h10_record_windows.py

Same prompts as the macOS script. CSV and meta files are written to
.\\recordings\\.

CSV columns (identical to the macOS script)
-------------------------------------------
t_perf_s         Monotonic seconds since recording start (perf_counter).
t_epoch_s        Unix epoch seconds (float, ms precision) when the RR was logged.
hr_bpm           Heart rate as reported by the strap that packet.
ibi_ms           One RR interval in milliseconds.

Troubleshooting
---------------
"Polar H10 not found"
    - Make sure the strap is worn with WET electrodes. The strap doesn't
      advertise BLE until it detects skin contact.
    - Disconnect your phone from the H10 (Polar Beat / Flow apps).
    - Try removing the H10 pod from the strap, then snap it back on to
      force a fresh advertisement cycle.
    - Run "py -m bleak --discover" to confirm Windows can see BLE devices
      at all.

"Access denied" / connection drops
    - Run the terminal as Administrator on first run.
    - In Windows Settings, "Forget" the H10 if it's stuck in a paired-but-
      unreachable state, then re-run the script.

Stuttering / dropped packets
    - Move closer to the H10 (within ~2 m for first connect).
    - Disable other Bluetooth devices that might be saturating the radio.
    - Some Intel/Realtek BT adapters need a driver update; check the
      vendor's site rather than relying on Windows Update.
"""

import asyncio
import csv
import datetime as dt
import sys
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner

# -----------------------------------------------------------------------------
# BLE constants (identical to macOS script — they are device-side, not OS-side)
# -----------------------------------------------------------------------------

HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# -----------------------------------------------------------------------------
# Recording configuration
# -----------------------------------------------------------------------------

RECORDING_DURATION_S = 60
SCAN_TIMEOUT_S       = 15.0
STABILIZE_PACKETS    = 2
OUTPUT_DIR           = Path("recordings")


# -----------------------------------------------------------------------------
# Helpers (identical logic to macOS version — duplicated so each script is a
# single self-contained file students/collaborators can run without our
# package layout being correct)
# -----------------------------------------------------------------------------

def now_epoch() -> float:
    return time.time()


def epoch_to_iso_local(epoch: float) -> str:
    local = dt.datetime.fromtimestamp(epoch).astimezone()
    return local.isoformat(timespec="milliseconds")


def parse_hr_measurement(data: bytearray):
    flags = data[0]
    hr_uint16  = bool(flags & 0x01)
    rr_present = bool(flags & 0x10)

    idx = 1
    if hr_uint16:
        hr_bpm = int.from_bytes(data[idx:idx + 2], "little")
        idx += 2
    else:
        hr_bpm = data[idx]
        idx += 1

    if flags & 0x08:
        idx += 2  # skip energy expended

    rr_ms_list = []
    if rr_present:
        while idx + 1 < len(data):
            rr_raw = int.from_bytes(data[idx:idx + 2], "little")
            idx += 2
            rr_ms_list.append(rr_raw * 1000.0 / 1024.0)

    return hr_bpm, rr_ms_list


def compute_hrv_stats(rr_ms_list):
    if len(rr_ms_list) < 2:
        return None
    import statistics
    n = len(rr_ms_list)
    mean_rr = statistics.mean(rr_ms_list)
    sdnn = statistics.pstdev(rr_ms_list)
    diffs = [rr_ms_list[i + 1] - rr_ms_list[i] for i in range(n - 1)]
    rmssd = (sum(d * d for d in diffs) / len(diffs)) ** 0.5
    nn50 = sum(1 for d in diffs if abs(d) > 50)
    return {
        "n_beats":     n,
        "mean_hr_bpm": 60_000.0 / mean_rr,
        "mean_rr_ms":  mean_rr,
        "min_rr_ms":   min(rr_ms_list),
        "max_rr_ms":   max(rr_ms_list),
        "sdnn_ms":     sdnn,
        "rmssd_ms":    rmssd,
        "pnn50_pct":   100.0 * nn50 / len(diffs),
    }


async def wait_for_enter(prompt: str = ">>> Press ENTER to start... "):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input, prompt)


# -----------------------------------------------------------------------------
# Device discovery
# -----------------------------------------------------------------------------

async def find_polar_h10():
    """
    On Windows, BleakScanner returns BLEDevice objects whose .address attribute
    is a MAC string ("AA:BB:CC:DD:EE:FF"). We still match by the human-readable
    name field because Polar straps embed the serial number there.
    """
    print("Scanning for Polar H10...")
    devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT_S)
    for d in devices:
        if d.name and "Polar H10" in d.name:
            print(f"Found: {d.name}  [MAC: {d.address}]")
            return d
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

async def main():
    trial_label = input("Trial label (e.g. subject01_rest, or blank): ").strip()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = OUTPUT_DIR / f"polar_h10_{ts}.csv"
    meta_path = OUTPUT_DIR / f"polar_h10_{ts}.meta.txt"

    device = await find_polar_h10()
    if device is None:
        print("Polar H10 not found. Check strap, electrodes, phone Bluetooth.")
        sys.exit(1)

    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["t_perf_s", "t_epoch_s", "hr_bpm", "ibi_ms"])

    all_rr = []
    packet_count = 0
    recording = False
    t0_perf = None
    t0_epoch = None
    start_iso = None

    def on_hr_notify(_sender, data):
        nonlocal packet_count
        packet_count += 1
        if not recording:
            return
        try:
            hr_bpm, rr_list = parse_hr_measurement(data)
        except Exception as e:
            print(f"  parse error: {e}")
            return
        t_perf  = time.perf_counter() - t0_perf
        t_epoch = now_epoch()
        for rr in rr_list:
            writer.writerow([f"{t_perf:.6f}", f"{t_epoch:.6f}",
                             hr_bpm, f"{rr:.1f}"])
            all_rr.append(rr)
            print(f"[t={t_perf:5.2f}s] HR={hr_bpm:3d} BPM | IBI={rr:7.1f} ms")

    async with BleakClient(device) as client:
        print(f"\nConnected to {device.name}\n")
        await client.start_notify(HR_MEASUREMENT_UUID, on_hr_notify)

        print("Stabilizing connection...")
        while packet_count < STABILIZE_PACKETS:
            await asyncio.sleep(0.1)
        print(f"H10 streaming OK ({packet_count} packets)\n")

        print("=" * 60)
        print("READY. Press ENTER to start the "
              f"{RECORDING_DURATION_S:.0f}-second recording.")
        print("Both this PC and mmWave computer should be NTP-synced.")
        print("=" * 60)
        await wait_for_enter("\n>>> Press ENTER to start... ")

        recording = True
        t0_perf  = time.perf_counter()
        t0_epoch = now_epoch()
        start_iso = epoch_to_iso_local(t0_epoch)

        print(f"\n*** RECORDING STARTED ***")
        print(f"  Local ISO:  {start_iso}")
        print(f"  Unix epoch: {t0_epoch:.3f}")
        print(f"  Duration:   {RECORDING_DURATION_S:.0f} seconds\n")

        while True:
            elapsed   = time.perf_counter() - t0_perf
            remaining = RECORDING_DURATION_S - elapsed
            if remaining <= 0:
                break
            await asyncio.sleep(min(remaining, 0.5))

        recording = False
        end_epoch = now_epoch()
        end_iso   = epoch_to_iso_local(end_epoch)
        print(f"\n=== RECORDING COMPLETE ===")
        print(f"  Local ISO:  {end_iso}")
        print(f"  Unix epoch: {end_epoch:.3f}\n")
        await client.stop_notify(HR_MEASUREMENT_UUID)

    csv_file.close()

    with open(meta_path, "w") as f:
        f.write(f"trial_label: {trial_label or '(none)'}\n")
        f.write(f"device: {device.name}\n")
        f.write(f"device_address: {device.address}\n")
        f.write(f"platform: Windows\n")
        f.write(f"start_iso: {start_iso}\n")
        f.write(f"end_iso:   {end_iso}\n")
        f.write(f"start_epoch: {t0_epoch:.3f}\n")
        f.write(f"end_epoch:   {end_epoch:.3f}\n")
        f.write(f"duration_s: {RECORDING_DURATION_S}\n")
        f.write(f"csv_file:   {csv_path.name}\n")

    stats = compute_hrv_stats(all_rr)
    print(f"CSV saved:  {csv_path}")
    print(f"Meta saved: {meta_path}")
    print(f"\n--- Summary ---")
    if stats:
        print(f"  Beats recorded:  {stats['n_beats']}")
        print(f"  Mean HR:         {stats['mean_hr_bpm']:.1f} BPM")
        print(f"  Mean IBI:        {stats['mean_rr_ms']:.1f} ms")
        print(f"  IBI range:       {stats['min_rr_ms']:.1f}"
              f" - {stats['max_rr_ms']:.1f} ms")
        print(f"  SDNN:            {stats['sdnn_ms']:.1f} ms")
        print(f"  RMSSD:           {stats['rmssd_ms']:.1f} ms")
        print(f"  pNN50:           {stats['pnn50_pct']:.1f} %")
    print(f"\n  Recording: {start_iso}  ->  {end_iso}")


if __name__ == "__main__":
    # Windows: explicitly select the proactor loop policy for predictability
    # across Python versions (default since 3.8 but pin it anyway).
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCanceled.")
