"""Aggregator for DreamZero in-training open-loop eval metrics.

Consumes parallel lists of ``predictions`` (from ``EvalPolicyRunner.predict``)
and ``samples`` (from ``GenieSimEvalLoader``) and produces a flat
``dict[str, float]`` of scalar metrics suitable for merging into a HuggingFace
``TrainerCallback`` ``logs`` dict (wandb + loss_log).

Action dimension layout (normalized-relative, per ConcatTransform):
    dims [0:14]  = joint_position (14 joints, 7 per arm)
    dim  [14]    = left_effector_position
    dim  [15]    = right_effector_position
    dims [16:32] = padding zeros (ground-truth only; predictions are width 16)

Only the first ``action_horizon`` (default 24) timesteps of each sample's
``action_gt`` are compared to the policy's prediction chunk.
"""

from __future__ import annotations

import numpy as np

from groot.vla.eval.action_smoothness import (
    action_total_variation,
    jerk_rms,
)


def compute_eval_metrics(
    predictions: list[dict],
    samples: list[dict],
    action_horizon: int = 24,
    joint_slice: slice = slice(0, 14),
    left_eff_idx: int = 14,
    right_eff_idx: int = 15,
    fps: int = 30,
) -> dict[str, float]:
    """Aggregate per-episode predictions/GT into a flat scalar metrics dict.

    Args:
        predictions: list of dicts with key ``"action_pred"`` of shape
            ``(action_horizon, 16)``.
        samples: list of dicts with key ``"action_gt"`` of shape
            ``(max_chunk_size * action_horizon, max_action_dim)``. Only the
            first ``action_horizon`` rows are used.
        action_horizon: number of timesteps in one inference chunk.
        joint_slice: slice into action dim for the joint positions.
        left_eff_idx: action dim of the left end-effector.
        right_eff_idx: action dim of the right end-effector.
        fps: control rate in Hz; used as ``dt = 1/fps`` for smoothness metrics.

    Returns:
        Flat ``dict[str, float]`` with keys ``eval/joint_l1_norm``,
        ``eval/effector_l1_norm``, ``eval/atv_pred``, ``eval/atv_gt``,
        ``eval/jerk_rms_pred``, ``eval/jerk_rms_pred_x1k``,
        ``eval/jerk_rms_gt``, ``eval/jerk_rms_gt_x1k`` and
        ``eval/n_episodes``.
    """
    if len(predictions) != len(samples):
        raise ValueError(
            f"len(predictions)={len(predictions)} != len(samples)={len(samples)}"
        )

    n_episodes = len(predictions)
    if n_episodes == 0:
        # Return zeros to avoid breaking the training callback on empty eval.
        return {
            "eval_joint_l1_norm": 0.0,
            "eval_effector_l1_norm": 0.0,
            "eval_atv_pred": 0.0,
            "eval_atv_gt": 0.0,
            "eval_jerk_rms_pred": 0.0,
            "eval_jerk_rms_pred_x1k": 0.0,
            "eval_jerk_rms_gt": 0.0,
            "eval_jerk_rms_gt_x1k": 0.0,
            "eval_n_episodes": 0.0,
        }

    # Stack predictions and the first action_horizon timesteps of each GT.
    # action_pred: (action_horizon, 16); action_gt: (max_T, max_dim).
    pred_stack = np.stack(
        [np.asarray(p["action_pred"])[:action_horizon] for p in predictions],
        axis=0,
    ).astype(np.float64)  # (B, T, 16)

    pred_dim = pred_stack.shape[-1]  # 16 (joints + 2 effectors)

    gt_stack = np.stack(
        [
            np.asarray(s["action_gt"])[:action_horizon, :pred_dim]
            for s in samples
        ],
        axis=0,
    ).astype(np.float64)  # (B, T, 16)

    if pred_stack.shape != gt_stack.shape:
        raise ValueError(
            f"shape mismatch after slicing: pred {pred_stack.shape} "
            f"vs gt {gt_stack.shape}"
        )

    # --- L1 errors --------------------------------------------------------
    # Joint L1: mean over (episode, timestep, joint_dim).
    joint_diff = np.abs(
        pred_stack[..., joint_slice] - gt_stack[..., joint_slice]
    )
    joint_l1 = float(joint_diff.mean())

    # Effector L1: mean over (episode, timestep, {left, right}).
    eff_idx = np.array([left_eff_idx, right_eff_idx])
    eff_diff = np.abs(pred_stack[..., eff_idx] - gt_stack[..., eff_idx])
    eff_l1 = float(eff_diff.mean())

    # --- Smoothness metrics ----------------------------------------------
    dt = 1.0 / float(fps)
    atv_pred = action_total_variation(
        pred_stack, dt=dt, convert_per_dt=True
    )
    atv_gt = action_total_variation(
        gt_stack, dt=dt, convert_per_dt=True
    )
    jerk_pred = jerk_rms(pred_stack, dt=dt)
    jerk_gt = jerk_rms(gt_stack, dt=dt)

    return {
        "eval_joint_l1_norm": joint_l1,
        "eval_effector_l1_norm": eff_l1,
        "eval_atv_pred": float(atv_pred),
        "eval_atv_gt": float(atv_gt),
        "eval_jerk_rms_pred": float(jerk_pred),
        "eval_jerk_rms_pred_x1k": float(jerk_pred * 1e-3),
        "eval_jerk_rms_gt": float(jerk_gt),
        "eval_jerk_rms_gt_x1k": float(jerk_gt * 1e-3),
        "eval_n_episodes": float(n_episodes),
    }


if __name__ == "__main__":
    # Smoke test: 3 episodes; random preds vs. linear (zero-jerk) GT.
    rng = np.random.default_rng(0)

    action_horizon = 24
    max_chunk_size = 4
    max_T = max_chunk_size * action_horizon  # 96
    max_action_dim = 32
    pred_dim = 16

    predictions = []
    samples = []
    for ep in range(3):
        # Random predicted chunk in [-1, 1].
        action_pred = rng.uniform(-1.0, 1.0, size=(action_horizon, pred_dim))

        # Linear GT in normalized space. Use exactly-representable floats
        # (small integers / powers of 2) so the third finite difference of a
        # strictly linear signal is binary-exactly zero in float64.
        start = (rng.integers(-8, 9, size=(pred_dim,)).astype(np.float64)
                 / 16.0)  # k/16, exactly representable
        slope = (rng.integers(-4, 5, size=(pred_dim,)).astype(np.float64)
                 / 1024.0)  # k/1024, exactly representable, small
        t = np.arange(max_T, dtype=np.float64)[:, None]
        gt_linear = start[None, :] + slope[None, :] * t  # (max_T, 16)

        action_gt = np.zeros((max_T, max_action_dim), dtype=np.float64)
        action_gt[:, :pred_dim] = gt_linear

        predictions.append(
            {
                "action_pred": action_pred,
                "episode_index": ep,
                "start_frame": 0,
            }
        )
        samples.append({"action_gt": action_gt})

    metrics = compute_eval_metrics(predictions, samples)
    print("eval metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v!r}")

    # Sanity checks.
    assert metrics["eval_joint_l1_norm"] > 0, "random vs linear should diff"
    assert metrics["eval_effector_l1_norm"] > 0
    assert metrics["eval_atv_pred"] > 0, "random has nonzero ATV"
    assert metrics["eval_atv_gt"] > 0, "linear with nonzero slope has ATV>0"
    assert metrics["eval_jerk_rms_gt"] == 0.0, (
        f"linear GT should have exactly zero jerk_rms; "
        f"got {metrics['eval/jerk_rms_gt']!r}"
    )
    assert metrics["eval_n_episodes"] == 3.0
    print("smoke test OK; jerk_rms_gt is exactly 0")
