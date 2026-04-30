import numpy as np
import pytest

from radar_analysis.hrv_metrics import (
    mean_ibi_ms,
    mean_hr_bpm,
    sdnn_ms,
    rmssd_ms,
    pnn50,
)


def test_constant_nn_yields_zero_variability():
    nn = np.full(100, 1000.0)
    assert mean_ibi_ms(nn) == pytest.approx(1000.0)
    assert mean_hr_bpm(nn) == pytest.approx(60.0)
    assert sdnn_ms(nn) == pytest.approx(0.0)
    assert rmssd_ms(nn) == pytest.approx(0.0)
    assert pnn50(nn) == pytest.approx(0.0)


def test_alternating_pattern_known_values():
    # NN = [800, 1000, 800, 1000, 800] (length 5)
    # mean = 880, diffs = [200, -200, 200, -200]
    nn = np.array([800.0, 1000.0, 800.0, 1000.0, 800.0])
    # SDNN: variance with ddof=1 = (3*80^2 + 2*120^2)/4 = (19200 + 28800)/4 = 12000
    assert mean_ibi_ms(nn) == pytest.approx(880.0)
    assert sdnn_ms(nn) == pytest.approx(np.sqrt(12000.0))
    # RMSSD: sum(d^2) = 4 * 40000 = 160000; /4 = 40000; sqrt = 200
    assert rmssd_ms(nn) == pytest.approx(200.0)
    # All 4 diffs are |200| > 50 ms → pNN50 = 100%
    assert pnn50(nn) == pytest.approx(100.0)


def test_linearly_increasing_nn():
    nn = np.array([800.0, 850.0, 900.0, 950.0, 1000.0])
    assert mean_ibi_ms(nn) == pytest.approx(900.0)
    # diffs all = 50 ms → strictly > 50? No → pNN50 = 0
    assert pnn50(nn) == pytest.approx(0.0)
    # RMSSD: sum(2500)*4 / 4 = 2500; sqrt = 50
    assert rmssd_ms(nn) == pytest.approx(50.0)


def test_pnn50_strict_inequality():
    # diffs of exactly 50 should NOT count
    nn = np.array([1000.0, 1050.0, 1000.0, 1050.0])
    assert pnn50(nn) == pytest.approx(0.0)
    # diffs of 51 should count
    nn = np.array([1000.0, 1051.0, 1000.0, 1051.0])
    assert pnn50(nn) == pytest.approx(100.0)


def test_mean_hr_bpm_inverse_of_mean_ibi():
    nn = np.array([800.0, 900.0, 1000.0, 1100.0])
    assert mean_hr_bpm(nn) * mean_ibi_ms(nn) == pytest.approx(60_000.0)


def test_validation_errors():
    with pytest.raises(ValueError):
        sdnn_ms(np.array([1000.0]))           # < 2 intervals
    with pytest.raises(ValueError):
        rmssd_ms(np.array([1000.0]))
    with pytest.raises(ValueError):
        sdnn_ms(np.zeros((2, 2)))             # not 1-D


def test_sdnn_uses_n_minus_1():
    # Population std would give /N; sample std /N-1. Confirm we use N-1.
    nn = np.array([950.0, 1000.0, 1050.0])
    expected = np.sqrt(((950 - 1000)**2 + 0 + (1050 - 1000)**2) / 2)
    assert sdnn_ms(nn) == pytest.approx(expected)
