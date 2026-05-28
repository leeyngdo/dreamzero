"""Thin inference wrapper around the live in-training DreamZero model.

This module exposes :class:`EvalPolicyRunner`, which the in-training eval
callback uses to roll the live policy forward on a small set of validation
samples from :class:`GenieSimEvalLoader` and return predicted action
sequences. It is intentionally minimal:

  * No checkpoint loading. The model passed in IS the live training model.
  * No metric computation. Agent 6 (metrics) consumes the returned numpy
    arrays.
  * No denormalization. The model was trained with ``relative_action=True``
    on q99-normalized relative actions (see
    ``configs/data/dreamzero/genie_sim_relative_wan22.yaml`` +
    ``meta/relative_stats_dreamzero.json``). The returned ``action_pred``
    therefore lives in q99-normalized RELATIVE units. To recover absolute
    joint angles in radians, the caller must (a) inverse-q99 with the same
    per-subkey stats and (b) add the reference state at the anchor frame.
    Agents 6/9 handle that.

Distributed safety
------------------
All ranks must call :meth:`predict` with identical inputs. The underlying
flow-matching action head runs a multi-step diffusion loop and (when
DeepSpeed is engaged) any rank that does not call the forward will hang
the collective.

Batch assembly
--------------
The dreamzero collator (``DefaultDataCollator`` /
``DreamTransform.apply_single``) does a lot of per-embodiment text
decoration (see ``collate`` in ``dreamzero_cotrain.py``). That decoration
branches on ``EmbodimentTag`` and does NOT have a ``GENIE_SIM`` arm — so
calling the collator directly on a ``GenieSimEvalLoader`` sample would
crash. Instead, this runner mirrors the *post-transform* shape contract
of ``apply_single`` and tokenizes text manually with the same
:class:`HuggingfaceTokenizer` the collator uses. The keys we emit match
exactly what the action head's ``lazy_joint_video_action`` reads off of
``action_input`` (``images``, ``state``, ``action``, ``text``,
``text_attention_mask``, ``text_negative``,
``text_attention_mask_negative``, ``embodiment_id``).

Returned slice ordering
-----------------------
The model predicts in the 32-dim padded action space. Only the first
16 dims are "real" for genie_sim; they are laid out per the
``ConcatTransform`` order in ``modality_config_genie_sim``:

  - dims  0..13 : action.joint_position    (14 dims)
  - dim   14    : action.left_effector_position
  - dim   15    : action.right_effector_position
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

# The default DreamTransform.text_negative string (copied verbatim from
# groot/vla/model/dreamzero/transform/dreamzero_cotrain.py::apply_single).
# Kept inline to avoid a heavyweight import of DreamTransform (which pulls
# in the entire data transform stack).
_DEFAULT_TEXT_NEGATIVE = (
    "Vibrant colors, overexposed, static, blurry details, text, subtitles, "
    "style, artwork, painting, image, still, grayscale, dull, worst quality, "
    "low quality, JPEG artifacts, ugly, mutilated, extra fingers, bad hands, "
    "bad face, deformed, disfigured, mutated limbs, fused fingers, stagnant "
    "image, cluttered background, three legs, many people in the background, "
    "walking backwards."
)

# Number of "real" action dims for genie_sim. Matches the ConcatTransform
# ordering in modality_config_genie_sim:
#   action.joint_position (14) + left_effector (1) + right_effector (1).
_GENIE_SIM_REAL_ACTION_DIMS = 16


class EvalPolicyRunner:
    """Thin wrapper that runs the live DreamZero model on eval samples.

    Parameters
    ----------
    model:
        The unwrapped DreamZero model. The caller is responsible for
        unwrapping DDP / DeepSpeed (i.e. pass
        ``model.module if hasattr(model, "module") else model``).
    tokenizer:
        The same :class:`HuggingfaceTokenizer` instance the
        :class:`DefaultDataCollator` uses at training time. It must
        accept ``(text, return_mask=True, add_special_tokens=True)`` and
        return ``(ids, attention_mask)`` torch tensors.
    num_inference_steps:
        Number of flow-matching denoising steps at eval. Defaults to 4
        (vs. the head's default 16) to keep the eval cheap. Restored
        after each :meth:`predict` call.
    device:
        Device to materialize the batch on. Defaults to ``"cuda"``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        num_inference_steps: int = 4,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_inference_steps = int(num_inference_steps)
        self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, samples: list[dict]) -> list[dict]:
        """Run the live model on a list of per-episode samples.

        Parameters
        ----------
        samples:
            List of sample dicts produced by
            :class:`GenieSimEvalLoader`. Each dict must have keys:
            ``images`` (T, H, W, C) uint8, ``state``
            (max_chunk_size*state_horizon, max_state_dim) float,
            ``action_gt`` (max_chunk_size*action_horizon, max_action_dim)
            float, ``text`` str, ``embodiment_id`` int,
            ``episode_index`` int, ``start_frame`` int.

        Returns
        -------
        list of dict, one per input sample, with keys:

          - ``action_pred``: np.ndarray of shape
            ``(action_horizon, 16)`` in q99-normalized RELATIVE units
            (see module docstring).
          - ``episode_index``: int
          - ``start_frame``: int
        """
        if len(samples) == 0:
            return []

        action_head = self.model.action_head

        # ------------------------------------------------------------------
        # 1) Save state so we can restore after the eval call.
        # ------------------------------------------------------------------
        prev_training = self.model.training
        prev_num_inference_steps = action_head.num_inference_steps
        prev_current_start_frame = action_head.current_start_frame
        prev_language = action_head.language

        self.model.eval()
        action_head.num_inference_steps = self.num_inference_steps
        action_head.current_start_frame = 0
        action_head.language = None

        try:
            # --------------------------------------------------------------
            # 2) Assemble the batch dict in the same shape contract as
            #    DreamTransform.apply_single + collate.
            # --------------------------------------------------------------
            batch = self._build_batch(samples)

            # --------------------------------------------------------------
            # 3) Forward through the live model under bf16 autocast.
            # --------------------------------------------------------------
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = self.model.lazy_joint_video_action_causal(batch)

            # outputs["action_pred"] has shape (B, action_horizon, action_dim)
            action_pred = outputs["action_pred"].detach().float().cpu().numpy()
        finally:
            # --------------------------------------------------------------
            # 4) Restore state so the caller's training loop is untouched.
            # --------------------------------------------------------------
            action_head.num_inference_steps = prev_num_inference_steps
            action_head.current_start_frame = prev_current_start_frame
            action_head.language = prev_language
            if prev_training:
                self.model.train()
            else:
                self.model.eval()

        # ------------------------------------------------------------------
        # 5) Slice to the 16 real dims and pack the per-sample result list.
        # ------------------------------------------------------------------
        real_dims = _GENIE_SIM_REAL_ACTION_DIMS
        out = []
        for i, sample in enumerate(samples):
            pred_i = action_pred[i, :, :real_dims]  # (action_horizon, 16)
            out.append(
                {
                    "action_pred": np.asarray(pred_i, dtype=np.float32),
                    "episode_index": int(sample["episode_index"]),
                    "start_frame": int(sample["start_frame"]),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_batch(self, samples: list[dict]) -> dict:
        """Stack per-sample dicts into a batch dict on ``self.device``.

        Mirrors the post-transform keyset produced by
        ``DreamTransform.apply_single`` followed by ``collate`` — minus
        the embodiment-specific text decoration, which we replace with
        plain tokenization of the raw text (the model only needs token
        ids + mask, not the decoration).
        """
        B = len(samples)

        # --- Images: (B, T, H, W, C) uint8 ---------------------------------
        images_np = np.stack([s["images"] for s in samples], axis=0)
        # Action head accepts uint8 directly and converts internally; we keep
        # uint8 to avoid wasting bandwidth on a needless float copy.
        images = torch.from_numpy(images_np)

        # --- State: (B, max_chunk_size*state_horizon, max_state_dim) -------
        state_np = np.stack(
            [np.asarray(s["state"], dtype=np.float32) for s in samples], axis=0
        )
        state = torch.from_numpy(state_np)
        state_mask = torch.zeros_like(state, dtype=torch.bool)

        # --- Action placeholders -------------------------------------------
        # The action head's inference path samples noise of its own and does
        # not condition on the input ``action`` tensor; we only need a
        # correctly-shaped placeholder so ``validate_inputs`` (which only
        # checks shape when "action" is present) is happy.
        action_shape = (
            B,
            samples[0]["action_gt"].shape[0],
            samples[0]["action_gt"].shape[1],
        )
        action = torch.zeros(action_shape, dtype=torch.float32)
        action_mask = torch.zeros(action_shape, dtype=torch.bool)

        # --- Tokenize text manually ----------------------------------------
        texts = [str(s["text"]) for s in samples]
        text_ids, text_mask = self.tokenizer(
            texts, return_mask=True, add_special_tokens=True
        )

        neg_texts = [_DEFAULT_TEXT_NEGATIVE] * B
        text_neg_ids, text_neg_mask = self.tokenizer(
            neg_texts, return_mask=True, add_special_tokens=True
        )

        # --- Misc per-sample scalars / placeholders ------------------------
        embodiment_id = torch.tensor(
            [int(s["embodiment_id"]) for s in samples], dtype=torch.long
        )

        has_real_action = torch.zeros((B,), dtype=torch.bool)
        has_lapa_action = torch.zeros((B,), dtype=torch.bool)

        lapa_action = torch.zeros(action_shape, dtype=torch.float32)
        lapa_action_mask = torch.zeros(action_shape, dtype=torch.bool)

        segmentation_target = torch.zeros((B, 2), dtype=torch.float32)
        segmentation_target_mask = torch.zeros((B, 1), dtype=torch.float32)

        is_cotrain_instance = torch.ones((B,), dtype=torch.bool)

        batch: dict = {
            "images": images,
            "state": state,
            "state_mask": state_mask,
            "action": action,
            "action_mask": action_mask,
            "lapa_action": lapa_action,
            "lapa_action_mask": lapa_action_mask,
            "has_real_action": has_real_action,
            "has_lapa_action": has_lapa_action,
            "segmentation_target": segmentation_target,
            "segmentation_target_mask": segmentation_target_mask,
            "is_cotrain_instance": is_cotrain_instance,
            "embodiment_id": embodiment_id,
            "text": text_ids,
            "text_attention_mask": text_mask,
            "text_negative": text_neg_ids,
            "text_attention_mask_negative": text_neg_mask,
        }

        # Move to device. dtype casts are deferred to the model's
        # ``prepare_input``, which casts floats to the action_head's compute
        # dtype and keeps integer / bool tensors as-is.
        for k, v in batch.items():
            batch[k] = v.to(self.device)

        return batch


if __name__ == "__main__":
    # Sanity check: instantiate the class with a dummy nn.Module +
    # dummy tokenizer. We DO NOT run forward — Agent 9 owns the smoke test.
    class _DummyTokenizer:
        def __call__(self, texts, return_mask=False, add_special_tokens=True):
            n = len(texts) if isinstance(texts, list) else 1
            ids = torch.zeros((n, 8), dtype=torch.long)
            mask = torch.ones((n, 8), dtype=torch.long)
            if return_mask:
                return ids, mask
            return ids

    dummy_model = torch.nn.Module()
    runner = EvalPolicyRunner(
        model=dummy_model,
        tokenizer=_DummyTokenizer(),
        num_inference_steps=4,
        device="cpu",
    )
    assert runner.num_inference_steps == 4
    assert runner.device == "cpu"
    # predict([]) should return [] without touching the dummy model.
    assert runner.predict([]) == []
    print("EvalPolicyRunner imports clean and basic sanity passes.")
