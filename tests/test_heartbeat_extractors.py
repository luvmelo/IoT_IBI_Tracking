import numpy as np
import pytest

from radar_analysis.heartbeat_extractors import (
    bandpass,
    extract_respiration,
    extract_heartbeat,
)


def _make_signal(fs, duration_s, freqs_amps, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(fs * duration_s)) / fs
    x = np.zeros_like(t)
    for f, a in freqs_amps:
        x += a * np.sin(2 * np.pi * f * t)
    if noise:
        x += noise * rng.standard_normal(t.size)
    return t, x


def test_bandpass_passes_inband_tone():
    fs = 25.0
    t, x = _make_signal(fs, 60.0, [(1.2, 1.0)])
    y = bandpass(x, fs, 0.8, 4.0)
    # Steady-state amplitude preserved within 5 % (filtfilt has ~unit gain in passband)
    middle = slice(int(10 * fs), int(50 * fs))
    assert 0.95 < y[middle].std() / x[middle].std() < 1.05


def test_bandpass_suppresses_outofband_tone():
    fs = 25.0
    _, x = _make_signal(fs, 60.0, [(0.25, 1.0)])  # respiration tone
    y = bandpass(x, fs, 0.8, 4.0)
    middle = slice(int(10 * fs), int(50 * fs))
    # Should be heavily attenuated (>20× reduction)
    assert y[middle].std() < 0.05 * x[middle].std()


def test_extract_heartbeat_isolates_hr_from_resp():
    fs = 25.0
    t, x = _make_signal(fs, 60.0, [(0.25, 4.0), (1.2, 1.0)])
    hb = extract_heartbeat(x, fs)
    middle = slice(int(10 * fs), int(50 * fs))
    # Power spectrum: dominant peak should land near 1.2 Hz, not 0.25 Hz
    seg = hb[middle]
    freqs = np.fft.rfftfreq(seg.size, d=1 / fs)
    spec = np.abs(np.fft.rfft(seg - seg.mean()))
    peak_hz = freqs[int(np.argmax(spec))]
    assert abs(peak_hz - 1.2) < 0.1


def test_extract_respiration_isolates_resp_from_hr():
    fs = 25.0
    t, x = _make_signal(fs, 60.0, [(0.25, 1.0), (1.2, 4.0)])
    resp = extract_respiration(x, fs)
    middle = slice(int(10 * fs), int(50 * fs))
    seg = resp[middle]
    freqs = np.fft.rfftfreq(seg.size, d=1 / fs)
    spec = np.abs(np.fft.rfft(seg - seg.mean()))
    peak_hz = freqs[int(np.argmax(spec))]
    assert abs(peak_hz - 0.25) < 0.05


def test_bandpass_validates_cutoffs():
    with pytest.raises(ValueError):
        bandpass(np.zeros(100), fs=10.0, low_hz=0.5, high_hz=6.0)  # high >= nyq
    with pytest.raises(ValueError):
        bandpass(np.zeros(100), fs=10.0, low_hz=2.0, high_hz=1.0)  # inverted


def test_bandpass_rejects_too_short_signal_with_clear_error():
    # filtfilt on a signal shorter than padlen would crash with a cryptic message.
    # We catch it and surface the actual length / requirement instead.
    short = np.random.default_rng(0).standard_normal(10)
    with pytest.raises(ValueError, match="at least"):
        bandpass(short, fs=25.0, low_hz=0.8, high_hz=4.0, order=4)
