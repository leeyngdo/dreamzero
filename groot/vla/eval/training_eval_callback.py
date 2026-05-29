# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""In-training eval callback for DreamZero.

Wires together Agent 4's :class:`GenieSimEvalLoader`, Agent 5's
:class:`EvalPolicyRunner`, and Agent 6's :func:`compute_eval_metrics` into a
single :class:`transformers.TrainerCallback` that runs every ``eval_every``
steps. Metrics produced by the eval are stashed and re-injected into the next
``on_log`` event so they propagate to wandb and to the JSONL written by
:class:`groot.vla.experiment.base.LossLoggerCallback`.

Distributed semantics:
    * The model forward MUST be invoked on every rank — DeepSpeed/DDP will
      deadlock if only rank 0 enters the model. We therefore call the runner on
      every rank.
    * Only rank 0 mutates the ``logs`` dict (wandb / JSONL are also rank-0
      only), so the injected metrics only land in rank 0's log stream.
    * Any exception during eval is caught and printed (rank 0 only) — eval
      MUST NOT crash training.

Performance note:
    With ``eval_every=100`` and a 4-episode eval set of 4 denoising steps the
    overhead is on the order of 1-2 percent of wall time. Setting
    ``eval_every`` lower (e.g. 50) will be visible in step time.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from transformers import TrainerCallback

if TYPE_CHECKING:  # avoid import-time circular dependency
    from groot.vla.eval.episode_loader import GenieSimEvalLoader


def _to_python_float(v: Any) -> Any:
    """Coerce numpy / 0-d torch scalars to native Python float for wandb."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.ndarray) and v.ndim == 0:
        return float(v)
    try:
        import torch

        if isinstance(v, torch.Tensor) and v.ndim == 0:
            return float(v.item())
    except ImportError:
        pass
    return v


class TrainingEvalCallback(TrainerCallback):
    """Run open-loop eval on a small held-out subset every N training steps.

    Injects metrics (``eval/joint_l1_norm``, ``eval/effector_l1_norm``,
    ``eval/atv_pred``, ``eval/atv_gt``, ``eval/jerk_rms_pred``,
    ``eval/jerk_rms_gt``, ...) into the next ``logs`` dict so wandb and the
    LossLoggerCallback both pick them up.

    Runs the model forward on ALL distributed ranks (otherwise DDP/DeepSpeed
    hangs on partial-rank forward), but only logs / writes from rank 0.

    Args:
        eval_loader: Materialized :class:`GenieSimEvalLoader` instance. The
            callback pre-iterates it once in :meth:`on_train_begin` to avoid
            re-decoding video on every eval call.
        eval_every: Trigger eval every this many global steps.
        num_inference_steps: Number of denoising steps for the flow matching
            policy. Lower = faster but noisier.
        skip_step_zero: If True, the eval at ``global_step == 0`` is skipped
            (the first eval happens at ``eval_every``). This mirrors the
            typical convention of not evaluating before the first optimizer
            step.
    """

    def __init__(
        self,
        eval_loader: "GenieSimEvalLoader",
        eval_every: int = 100,
        num_inference_steps: int = 4,
        skip_step_zero: bool = True,
    ):
        self.eval_loader = eval_loader
        self.eval_every = int(eval_every)
        self.num_inference_steps = int(num_inference_steps)
        self.skip_step_zero = bool(skip_step_zero)

        # Filled in on_train_begin (model + tokenizer are not available until
        # then).
        self.runner: Optional[Any] = None
        self.eval_samples: Optional[list] = None
        self._pending: Optional[dict] = None
        self._is_initialized: bool = False

    # -- lazy init ---------------------------------------------------------

    def _build_runner(self, model, tokenizer):
        """Construct the :class:`EvalPolicyRunner` lazily.

        We defer the import so this module is importable without the runner /
        metrics modules being on the path (e.g. for unit tests).
        """
        from groot.vla.eval.policy_runner import EvalPolicyRunner  # local import

        return EvalPolicyRunner(
            model=model,
            tokenizer=tokenizer,
            num_inference_steps=self.num_inference_steps,
        )

    def _materialize_samples(self) -> list:
        """Pre-decode all eval episodes into a list of sample dicts."""
        return list(self.eval_loader)

    def _maybe_get_tokenizer(self, kwargs: dict) -> Any:
        """Best-effort tokenizer extraction.

        Transformers >= 4.46 passes ``processing_class`` in callback kwargs.
        Our :class:`DefaultDataCollator` exposes a ``tokenizer`` attribute.
        We try both, in order of specificity.
        """
        # Trainer's data_collator is the most reliable source for our
        # HuggingfaceTokenizer wrapper.
        train_dl = kwargs.get("train_dataloader", None)
        if train_dl is not None:
            collate_fn = getattr(train_dl, "collate_fn", None)
            tok = getattr(collate_fn, "tokenizer", None)
            if tok is not None:
                return tok
        # Fallback: processing_class (HF >= 4.46)
        proc = kwargs.get("processing_class", None)
        if proc is not None:
            return proc
        return None

    # -- transformers TrainerCallback hooks --------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self._is_initialized:
            return
        try:
            tokenizer = self._maybe_get_tokenizer(kwargs)
            if model is None:
                # transformers always passes model via kwargs in call_event;
                # treat absence as a hard failure but don't kill training.
                if state.is_world_process_zero:
                    print(
                        "[TrainingEvalCallback] model not provided to "
                        "on_train_begin; eval will be skipped."
                    )
                return
            self.runner = self._build_runner(model, tokenizer)
            self.eval_samples = self._materialize_samples()
            self._is_initialized = True
            if state.is_world_process_zero:
                print(
                    f"[TrainingEvalCallback] initialized with "
                    f"{len(self.eval_samples)} eval samples, "
                    f"eval_every={self.eval_every}, "
                    f"num_inference_steps={self.num_inference_steps}"
                )
        except Exception as e:
            if state.is_world_process_zero:
                print(f"[TrainingEvalCallback] init failed: {e}; eval disabled.")
            self._is_initialized = False
            self.runner = None
            self.eval_samples = None

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self._is_initialized or self.runner is None or self.eval_samples is None:
            return
        if state.global_step == 0 and self.skip_step_zero:
            return
        if self.eval_every <= 0 or state.global_step % self.eval_every != 0:
            return

        # Run eval on EVERY rank to avoid DDP/DeepSpeed hangs. The runner
        # itself is responsible for the no_grad / autocast / eval-mode plumbing.
        try:
            preds = self.runner.predict(self.eval_samples)
            from groot.vla.eval.metrics import compute_eval_metrics  # local import

            metrics = compute_eval_metrics(preds, self.eval_samples)
        except Exception as e:
            if state.is_world_process_zero:
                import traceback
                tb = traceback.format_exc()
                print(
                    f"[TrainingEvalCallback] eval at step "
                    f"{state.global_step} failed: {type(e).__name__}: {e}\n"
                    f"--- TRACEBACK ---\n{tb}--- END TRACEBACK ---"
                )
            return

        if not isinstance(metrics, dict):
            if state.is_world_process_zero:
                print(
                    f"[TrainingEvalCallback] compute_eval_metrics returned "
                    f"{type(metrics).__name__}, expected dict; skipping."
                )
            return

        # Stash for the next on_log. Only rank 0 ever writes them out, but we
        # keep the data on every rank so the next-log-event injection logic
        # stays symmetric.
        self._pending = {
            # Use the "eval_" prefix (underscore) at source so transformers'
            # WandbCallback.rewrite_logs converts to "eval/" in wandb.
            # Without this, wandb sees "eval/X" as a non-eval key and slaps
            # an extra "train/" prefix, producing "train/eval/X" in the UI.
            (k if k.startswith("eval_") else f"eval_{k}"): _to_python_float(v)
            for k, v in metrics.items()
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._pending or logs is None:
            return
        # Only rank 0 actually writes / streams logs; the others can drop the
        # stash without polluting their (no-op) logs dict.
        if state.is_world_process_zero:
            for k, v in self._pending.items():
                logs[k] = v
        self._pending = None


if __name__ == "__main__":
    # Smoke test: construct the callback with a dummy loader (no model).
    # Verifies the class is importable and basic init wiring works.
    class _DummyLoader:
        def __iter__(self):
            return iter([])

        def __len__(self):
            return 0

    cb = TrainingEvalCallback(
        eval_loader=_DummyLoader(),
        eval_every=100,
        num_inference_steps=4,
        skip_step_zero=True,
    )
    assert cb.eval_every == 100
    assert cb.num_inference_steps == 4
    assert cb.skip_step_zero is True
    assert cb.runner is None
    assert cb.eval_samples is None
    assert cb._pending is None
    assert cb._is_initialized is False
    print("TrainingEvalCallback constructed OK")
