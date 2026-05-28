"""Action-smoothness metrics from the ACG paper (arXiv 2510.22201, Table II).

Two open-loop metrics on a predicted action sequence:

    ATV (Action Total Variation, rad/s, lower is better)
        ATV = (1 / (M * (T - 1))) * sum_{t=1..T-1} sum_{j=1..M}
                |a_{t+1}^j - a_t^j|
        With ``convert_per_dt=True`` the per-step difference is divided by
        ``dt`` so the reported unit is rad/s (matching the paper).

    JerkRMS (rad/s^3, lower is better; paper reports x1e3)
        JerkRMS = sqrt(
            (1 / (T - 3)) * sum_{t=1..T-3} || jerk_t ||_2^2
        )
    where the third time-derivative is approximated by the standard forward
    third finite difference:
        jerk_t ~= (s_{t+3} - 3*s_{t+2} + 3*s_{t+1} - s_t) / dt**3

The paper computes JerkRMS on the observed joint angle (state). For
open-loop eval we compute it on the predicted action sequence, since
``action.joint_position`` is the target joint angle for genie_sim.
"""

from __future__ import annotations

import numpy as np


def _validate_actions(actions: np.ndarray, min_T: int) -> np.ndarray:
    """Validate and normalize ``actions`` to shape ``(B, T, M)``.

    Accepts ``(T, M)`` (treated as a single batch element) or ``(B, T, M)``.
    Raises ``ValueError`` if ``T < min_T``.
    """
    arr = np.asarray(actions)
    if arr.ndim == 2:
        arr = arr[None, ...]  # (1, T, M)
    elif arr.ndim != 3:
        raise ValueError(
            f"actions must have shape (T, M) or (B, T, M); got {arr.shape}"
        )

    T = arr.shape[1]
    if T < min_T:
        raise ValueError(
            f"Need at least {min_T} timesteps; got T={T} (shape={arr.shape})."
        )

    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def action_total_variation(
    actions: np.ndarray,
    dt: float = 1.0 / 30,
    convert_per_dt: bool = True,
) -> float:
    """Action Total Variation (ATV) from ACG paper Table II.

    Formula:
        ATV = (1 / (M * (T - 1))) * sum_{t=1..T-1} sum_{j=1..M}
                |a_{t+1}^j - a_t^j|

    With ``convert_per_dt=True`` (default) the result is divided by ``dt`` so
    its unit is rad/s, matching the paper. Set to ``False`` for raw per-step
    absolute differences in rad.

    Args:
        actions: float array of shape ``(T, M)`` or ``(B, T, M)``. ``T`` is
            the number of timesteps, ``M`` the number of motors / action dims.
        dt: control timestep in seconds. Default ``1/30`` (30 Hz, genie_sim
            G1).
        convert_per_dt: if True, divide by ``dt`` so the unit is rad/s.

    Returns:
        Scalar (Python float). If a batch is supplied, the per-element ATV
        is averaged across the batch.

    Raises:
        ValueError: if ``T < 2`` or the input shape is wrong.
    """
    arr = _validate_actions(actions, min_T=2)  # diff needs >= 2 timesteps
    # |a_{t+1} - a_t| -> shape (B, T-1, M)
    diffs = np.abs(np.diff(arr, axis=1))
    # mean over (T-1) and M for each batch element
    per_elem = diffs.mean(axis=(1, 2))  # shape (B,)
    atv = float(per_elem.mean())
    if convert_per_dt:
        atv = atv / dt
    return atv


def jerk_rms(
    actions: np.ndarray,
    dt: float = 1.0 / 30,
) -> float:
    """Jerk RMS from ACG paper Table II (open-loop, on predicted actions).

    Formula:
        JerkRMS = sqrt(
            (1 / (T - 3)) * sum_{t=1..T-3} || jerk_t ||_2^2
        )

    Third derivative via the standard forward third finite difference:
        jerk_t ~= (s_{t+3} - 3*s_{t+2} + 3*s_{t+1} - s_t) / dt**3

    Unit is rad/s^3 (not x1e3 -- caller can multiply by 1e-3 to report on
    the paper's scale).

    Args:
        actions: float array of shape ``(T, M)`` or ``(B, T, M)``.
        dt: control timestep in seconds. Default ``1/30``.

    Returns:
        Scalar (Python float). If a batch is supplied, the per-element
        JerkRMS is averaged across the batch.

    Raises:
        ValueError: if ``T < 4`` or the input shape is wrong.
    """
    arr = _validate_actions(actions, min_T=4)
    # Forward third finite difference:
    #   jerk_t = (s_{t+3} - 3 s_{t+2} + 3 s_{t+1} - s_t) / dt^3
    # for t = 0 .. T-4, so output length is T-3.
    s = arr  # (B, T, M)
    jerks = (s[:, 3:] - 3.0 * s[:, 2:-1] + 3.0 * s[:, 1:-2] - s[:, :-3]) / (
        dt ** 3
    )  # (B, T-3, M)

    # ||jerk_t||_2^2 sums across motor dims
    sq_norms = (jerks ** 2).sum(axis=2)  # (B, T-3)
    # mean over time gives (1/(T-3)) * sum
    ms = sq_norms.mean(axis=1)  # (B,)
    per_elem = np.sqrt(ms)  # (B,)
    return float(per_elem.mean())


def compute_action_smoothness(
    actions: np.ndarray,
    dt: float = 1.0 / 30,
) -> dict:
    """Convenience wrapper returning both metrics.

    Returns:
        dict with keys:
            - ``"atv"``: ATV in rad/s (``convert_per_dt=True``).
            - ``"jerk_rms"``: JerkRMS in rad/s^3.
            - ``"jerk_rms_x1k"``: JerkRMS in 1e3 * rad/s^3, the unit used in
              the ACG paper table.
    """
    atv = action_total_variation(actions, dt=dt, convert_per_dt=True)
    jr = jerk_rms(actions, dt=dt)
    return {
        "atv": atv,
        "jerk_rms": jr,
        "jerk_rms_x1k": jr * 1e-3,
    }
