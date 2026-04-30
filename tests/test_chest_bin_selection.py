import numpy as np
import pytest

from radar_analysis.chest_bin_selection import select_chest_bin
from radar_analysis.synthetic import synthetic_range_cube


def test_picks_target_bin_in_clean_scene():
    range_res = 0.04
    target = 0.7  # m → bin 17 (round(0.7/0.04))
    cube = synthetic_range_cube(
        range_res_m=range_res,
        target_range_m=target,
        target_amp=1000.0,
        noise_std=10.0,
        seed=0,
    )
    bin_idx, score = select_chest_bin(cube, range_res_m=range_res)
    assert bin_idx == int(round(target / range_res))
    assert score > 0


def test_motion_variance_beats_stationary_clutter():
    # Equal-amplitude moving target vs stationary clutter; motion variance must win.
    range_res = 0.04
    cube = synthetic_range_cube(
        range_res_m=range_res,
        target_range_m=0.7,
        target_amp=1000.0,
        target_motion_amp_mm=1.0,
        target_motion_hz=1.2,
        clutter_range_m=1.2,
        clutter_amp=1000.0,
        noise_std=5.0,
        seed=1,
    )
    bin_idx, _ = select_chest_bin(cube, range_res_m=range_res, use_motion_variance=True)
    assert bin_idx == int(round(0.7 / range_res))


def test_pure_power_picks_clutter_when_amplitudes_are_close():
    # Sanity: with use_motion_variance=False and a louder clutter, clutter wins.
    range_res = 0.04
    cube = synthetic_range_cube(
        range_res_m=range_res,
        target_range_m=0.7,
        target_amp=500.0,
        clutter_range_m=1.2,
        clutter_amp=2000.0,
        noise_std=5.0,
        seed=2,
    )
    bin_idx, _ = select_chest_bin(cube, range_res_m=range_res, use_motion_variance=False)
    assert bin_idx == int(round(1.2 / range_res))


def test_search_window_excludes_out_of_band_targets():
    range_res = 0.04
    cube = synthetic_range_cube(
        range_res_m=range_res,
        n_range_bins=128,
        target_range_m=2.5,            # far outside default 0.3–1.5 m window
        target_amp=2000.0,
        clutter_range_m=0.6,           # weak inside-window candidate
        clutter_amp=200.0,
        noise_std=5.0,
        seed=3,
    )
    bin_idx, _ = select_chest_bin(cube, range_res_m=range_res)
    # Must land in [0.3, 1.5] m
    assert 0.3 / range_res <= bin_idx <= 1.5 / range_res


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        select_chest_bin(np.zeros((4, 4)), range_res_m=0.04)


def test_invalid_search_window_raises():
    cube = synthetic_range_cube(seed=4)
    with pytest.raises(ValueError):
        select_chest_bin(cube, range_res_m=0.04, search_window_m=(1.0, 0.5))


def test_target_at_lower_window_boundary_is_selected():
    # The default search window is (0.3, 1.5) m. With range_res=0.04, ceil(0.3/0.04)=8,
    # so bin 8 (= 0.32 m) is the first admissible bin. A subject sitting close to
    # the radar in real life lands here and must not be excluded by an off-by-one.
    range_res = 0.04
    cube = synthetic_range_cube(
        range_res_m=range_res,
        target_range_m=0.32,
        target_amp=2000.0,
        noise_std=5.0,
        seed=7,
    )
    bin_idx, score = select_chest_bin(cube, range_res_m=range_res)
    assert bin_idx == int(round(0.32 / range_res))
    assert score > 0
