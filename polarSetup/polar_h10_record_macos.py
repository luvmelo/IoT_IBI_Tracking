"""
polar_h10_record_macos.py
=========================

Synchronized timestamped Polar H10 RR-interval recorder for macOS.

Streams Heart Rate Measurement notifications (BLE characteristic 0x2A37) from a
Polar H10 chest strap, parses the RR-interval values out of each notification
packet, and writes them to a timestamped CSV alongside a meta sidecar file.

The CSV is the IBI ground-truth file used to validate mmWave-derived IBI in the
IoT_IBI_Tracking pipeline.

Platform
--------
macOS only. macOS requires bleak's CoreBluetooth backend, which addresses
peripherals by 128-bit UUID (not MAC address) and requires Bluetooth permission
for the terminal application. For Windows, see polar_h10_record_windows.py.

Requirements
------------
- Python 3.9+
- bleak >= 0.20

Install:
    pip3 install --user bleak

Usage
-----
1. Wear the Polar H10 chest strap with electrodes moistened.
2. Disconnect any phone or watch currently paired with the strap (the H10
   only streams to one client at a time).
3. Run:
       python3 polar_h10_record_macos.py
4. Optionally enter a trial label (e.g. "subject01_d07_a00_rest").
5. Wait for "READY" message, then press ENTER to start the timed recording.
6. The script writes:
       recordings/polar_h10_<timestamp>.csv
       recordings/polar_h10_<timestamp>.meta.txt

CSV columns
-----------
t_perf_s         Monotonic seconds since recording start (perf_counter).
t_epoch_s        Unix epoch seconds (float, ms precision) when the RR was logged.
hr_bpm           Heart rate as reported by the strap that packet.
ibi_ms           One RR interval in milliseconds.

Note: a single BLE notification can contain multiple RR intervals. Each RR is
written as its own CSV row, all sharing the same t_perf_s / t_epoch_s of the
notification arrival. This matches Polar's own data export format.

Time-sync convention
--------------------
Both the Polar laptop and the mmWave laptop must be NTP-synced before
recording. The CSV's t_epoch_s column is then directly comparable to the
mmWave radar's epoch timestamps for beat-pair matching downstream.
"""

import asyncio
import csv
import datetime as dt
import os
import sys
import time
from pathlib import Path

from bleak import BleakClient, BleakScanner

# -----------------------------------------------------------------------------
# BLE constants
# -----------------------------------------------------------------------------

# Standard Bluetooth SIG Heart Rate Measurement characteristic.
# Polar H10 advertises it on the standard Heart Rate service (0x180D).
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# -----------------------------------------------------------------------------
# Recording configuration
# -----------------------------------------------------------------------------

RECORDING_DURATION_S = 60          # change for longer/shorter trials
SCAN_TIMEOUT_S       = 15.0
STABILIZE_PACKETS    = 2           # require N notifications before unblocking
OUTPUT_DIR           = Path("recordings")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def now_epoch() -> float:
    """Wall-clock Unix timestamp with sub-millisecond resolution."""
    return time.time()


def epoch_to_iso_local(epoch: float) -> str:
    """Convert epoch seconds to local ISO-8601 with ms precision."""
    local = dt.datetime.fromtimestamp(epoch).astimezone()
    return local.isoformat(timespec="milliseconds")


def parse_hr_measurement(data: bytearray):
    """
    Parse the Heart Rate Measurement notification payload.

    Bluetooth GATT spec: byte 0 is a flags byte. Bit 0 controls HR field width
    (0 = 1 byte, 1 = 2 bytes). Bit 4 controls whether RR intervals follow.
    RR intervals are uint16 little-endian, scaled at 1/1024 second per LSB.

    Returns
    -------
    hr_bpm : int
    rr_ms_list : list[float]
    """
    flags = data[0]
    hr_uint16 = bool(flags & 0x01)
    rr_present = bool(flags & 0x10)

    idx = 1
    if hr_uint16:
        hr_bpm = int.from_bytes(data[idx:idx + 2], "little")
        idx += 2
    else:
        hr_bpm = data[idx]
        idx += 1

    # Energy expended field (flags bit 3) — skip if present.
    if flags & 0x08:
        idx += 2

    rr_ms_list = []
    if rr_present:
        while idx + 1 < len(data):
            rr_raw = int.from_bytes(data[idx:idx + 2], "little")
            idx += 2
            # 1024 Hz tick -> milliseconds.
            rr_ms_list.append(rr_raw * 1000.0 / 1024.0)

    return hr_bpm, rr_ms_list


def compute_hrv_stats(rr_ms_list):
    """Quick post-recording HRV summary. Returns None if too few beats."""
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
    """Non-blocking ENTER prompt that yields back to the BLE event loop."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, input, prompt)


# -----------------------------------------------------------------------------
# Device discovery
# -----------------------------------------------------------------------------

async def find_polar_h10():
    """Scan for an advertising Polar H10. Returns the BLEDevice or None."""
    print("Scanning for Polar H10...")
    devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT_S)
    for d in devices:
        if d.name and "Polar H10" in d.name:
            print(f"Found: {d.name}")
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
        print("Both this Mac and mmWave computer should be NTP-synced.")
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
        f.write(f"platform: macOS\n")
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
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nCanceled.")
