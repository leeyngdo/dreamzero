"""Unit tests for groot.vla.eval.action_smoothness."""

from __future__ import annotations

import numpy as np
import pytest

from groot.vla.eval.action_smoothness import (
    action_total_variation,
    compute_action_smoothness,
    jerk_rms,
)


# ---------------------------------------------------------------------------
# Constant sequence
# ---------------------------------------------------------------------------

def test_constant_sequence_zero_atv_and_jerk():
    T, M = 20, 4
    actions = np.full((T, M), 0.42, dtype=np.float64)

    assert action_total_variation(actions, dt=1.0 / 30) == pytest.approx(0.0)
    assert action_total_variation(
        actions, dt=1.0 / 30, convert_per_dt=False
    ) == pytest.approx(0.0)
    assert jerk_rms(actions, dt=1.0 / 30) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Linear ramp:  a_t = c * t   -> diff = c   -> ATV = c / dt   (per dim)
#                                Jerk = 0
# ---------------------------------------------------------------------------

def test_linear_ramp_atv_equals_c_over_dt_and_zero_jerk():
    T = 10
    c = 0.25
    dt = 1.0 / 30
    t = np.arange(T, dtype=np.float64)
    actions = (c * t).reshape(T, 1)

    # Raw per-step diff
    atv_raw = action_total_variation(actions, dt=dt, convert_per_dt=False)
    assert atv_raw == pytest.approx(c)

    # Per dt -> rad/s
    atv = action_total_variation(actions, dt=dt, convert_per_dt=True)
    assert atv == pytest.approx(c / dt)

    # No higher-order motion -> jerk == 0
    assert jerk_rms(actions, dt=dt) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Quadratic ramp:  a_t = c * t^2
#                  diff_t = c * (2t + 1)   -> ATV grows linearly
#                  3rd derivative = 0 -> jerk == 0
# ---------------------------------------------------------------------------

def test_quadratic_ramp_jerk_is_zero_and_atv_grows():
    T = 12
    c = 0.5
    dt = 0.1
    t = np.arange(T, dtype=np.float64)
    actions = (c * t ** 2).reshape(T, 1)

    # JerkRMS for any polynomial of degree < 3 is zero (exact for finite diff).
    assert jerk_rms(actions, dt=dt) == pytest.approx(0.0, abs=1e-9)

    # ATV (raw) equals mean over t=0..T-2 of c*(2t+1)
    # = c * (1 + 3 + ... + (2(T-2)+1)) / (T-1)
    # = c * (T-1)^2 / (T-1) = c * (T - 1)
    # Wait: sum_{t=0}^{T-2} (2t+1) = (T-1)^2, mean = (T-1)^2 / (T-1) = T-1
    expected_atv_raw = c * (T - 1)
    atv_raw = action_total_variation(actions, dt=dt, convert_per_dt=False)
    assert atv_raw == pytest.approx(expected_atv_raw)

    # Longer sequence -> larger ATV (grows linearly with T)
    T2 = 24
    t2 = np.arange(T2, dtype=np.float64)
    actions2 = (c * t2 ** 2).reshape(T2, 1)
    atv2_raw = action_total_variation(actions2, dt=dt, convert_per_dt=False)
    assert atv2_raw == pytest.approx(c * (T2 - 1))
    assert atv2_raw > atv_raw


# ---------------------------------------------------------------------------
# Cubic ramp:  a_t = c * t^3
# Third forward finite difference of a cubic is exactly 3! * c = 6c
# So jerk_t = 6c / dt^3 (constant), JerkRMS = 6c / dt^3
# ---------------------------------------------------------------------------

def test_cubic_ramp_jerk_rms_matches_analytical():
    T = 16
    c = 1.0
    dt = 1.0
    t = np.arange(T, dtype=np.float64)
    actions = (c * t ** 3).reshape(T, 1)

    expected_jerk = 6.0 * c / (dt ** 3)
    jr = jerk_rms(actions, dt=dt)
    assert jr == pytest.approx(expected_jerk, rel=1e-9)


def test_cubic_ramp_jerk_rms_scales_with_dt():
    T = 16
    c = 2.5
    dt = 1.0 / 30
    t = np.arange(T, dtype=np.float64)
    actions = (c * t ** 3).reshape(T, 1)

    expected_jerk = 6.0 * c / (dt ** 3)
    jr = jerk_rms(actions, dt=dt)
    assert jr == pytest.approx(expected_jerk, rel=1e-9)


def test_cubic_ramp_multi_dim_jerk_rms():
    """For M motors all following cubics with coeffs c_j, the per-step
    squared jerk-norm is sum_j (6 c_j / dt^3)^2; RMS is its sqrt."""
    T = 20
    dt = 1.0 / 30
    coeffs = np.array([1.0, -0.5, 2.0, 0.0])
    t = np.arange(T, dtype=np.float64)[:, None]  # (T, 1)
    actions = coeffs[None, :] * (t ** 3)  # (T, M)

    expected = np.sqrt(np.sum((6.0 * coeffs / dt ** 3) ** 2))
    jr = jerk_rms(actions, dt=dt)
    assert jr == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# Batched input == mean of per-element results
# ---------------------------------------------------------------------------

def test_batched_input_matches_per_element_mean():
    rng = np.random.default_rng(0)
    B, T, M = 5, 32, 7
    dt = 1.0 / 30
    batch = rng.standard_normal((B, T, M))

    atv_each = np.array(
        [action_total_variation(batch[b], dt=dt) for b in range(B)]
    )
    jr_each = np.array([jerk_rms(batch[b], dt=dt) for b in range(B)])

    atv_batched = action_total_variation(batch, dt=dt)
    jr_batched = jerk_rms(batch, dt=dt)

    assert atv_batched == pytest.approx(atv_each.mean(), rel=1e-12)
    assert jr_batched == pytest.approx(jr_each.mean(), rel=1e-12)


# ---------------------------------------------------------------------------
# Shape checks
# ---------------------------------------------------------------------------

def test_T_less_than_4_raises_for_jerk():
    actions = np.zeros((3, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        jerk_rms(actions)


def test_atv_needs_at_least_two_timesteps():
    actions = np.zeros((1, 2), dtype=np.float64)
    with pytest.raises(ValueError):
        action_total_variation(actions)


def test_wrong_dimensionality_raises():
    with pytest.raises(ValueError):
        action_total_variation(np.zeros(10))  # 1-D
    with pytest.raises(ValueError):
        jerk_rms(np.zeros((2, 3, 4, 5)))  # 4-D


def test_T_M_one_and_T_M_many_both_work():
    T = 10
    dt = 1.0 / 30
    actions_1d = np.arange(T, dtype=np.float64).reshape(T, 1)
    actions_md = np.tile(actions_1d, (1, 5))  # (T, 5)

    assert action_total_variation(actions_1d, dt=dt) == pytest.approx(1.0 / dt)
    assert action_total_variation(actions_md, dt=dt) == pytest.approx(1.0 / dt)
    assert jerk_rms(actions_1d, dt=dt) == pytest.approx(0.0, abs=1e-9)
    assert jerk_rms(actions_md, dt=dt) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Spec-sheet numerical check from the task brief:
#   dt=1, actions = arange(10).reshape(-1,1).astype(float)  -> ATV == 1.0
# ---------------------------------------------------------------------------

def test_spec_numerical_check_arange_dt1():
    actions = np.arange(10).reshape(-1, 1).astype(float)
    # convert_per_dt=False
    assert action_total_variation(
        actions, dt=1.0, convert_per_dt=False
    ) == pytest.approx(1.0)
    # convert_per_dt=True with dt=1.0 still yields 1.0
    assert action_total_variation(
        actions, dt=1.0, convert_per_dt=True
    ) == pytest.approx(1.0)


def test_spec_numerical_check_cubic_dt1():
    actions = np.arange(10).reshape(-1, 1).astype(float) ** 3
    # Constant jerk = 6 for c=1, dt=1
    assert jerk_rms(actions, dt=1.0) == pytest.approx(6.0, rel=1e-9)


# ---------------------------------------------------------------------------
# compute_action_smoothness wrapper
# ---------------------------------------------------------------------------

def test_compute_action_smoothness_wrapper_keys_and_scaling():
    rng = np.random.default_rng(42)
    actions = rng.standard_normal((40, 7))
    out = compute_action_smoothness(actions, dt=1.0 / 30)

    assert set(out.keys()) == {"atv", "jerk_rms", "jerk_rms_x1k"}
    assert out["jerk_rms_x1k"] == pytest.approx(out["jerk_rms"] * 1e-3)
    # Sanity: matches the individual functions
    assert out["atv"] == pytest.approx(
        action_total_variation(actions, dt=1.0 / 30)
    )
    assert out["jerk_rms"] == pytest.approx(jerk_rms(actions, dt=1.0 / 30))
