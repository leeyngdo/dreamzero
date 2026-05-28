"""Deterministic validation episode loader for the genie_sim_gear dataset.

This loader is used by the in-training eval callback to feed N validation episodes
to the live policy every M training steps. It is self-contained on purpose — it
must NOT depend on the iteration order of the training mixture, and it must
apply the SAME state/action normalization the training transforms apply, so the
live model sees the same input distribution at eval time as at train time.

Reference implementations the shapes/transforms below mirror:
    - ``groot/vla/model/dreamzero/transform/dreamzero_cotrain.py``
        ``DreamTransform._prepare_video / _prepare_state / _prepare_action``
    - ``groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml``
        ``modality_config_genie_sim`` (joint / left_effector / right_effector
        modality slices and ``q99`` normalization modes)
    - ``groot/vla/data/transform/state_action.py``
        ``Normalizer.forward`` (q99 formula)

NOTE on normalization choices:
    * State (joint_position, left/right_effector_position) is normalized using
      q99 against the per-modality stats sliced out of ``meta/stats.json`` (the
      raw 190-dim LeRobot stats), exactly like the dataset's ``_get_metadata``
      does at training time.
    * Action uses RELATIVE actions: every action in the chunk is offset by the
      reference state taken at the chunk's anchor frame, then normalized with
      q99 against ``meta/relative_stats_dreamzero.json``. This matches
      ``ShardedLeRobotSubLangSingleActionChunkDatasetDROID._convert_to_relative_action``
      followed by ``StateActionTransform`` with relative stats.
    * Effector dims in the relative stats have q99==q01==1.0 on this dataset
      (no variation), so the q99 normalizer falls through to identity for that
      slot — same behavior as training.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import cv2
import decord
import numpy as np
import pandas as pd

# Mirror DreamTransform shape configuration for genie_sim @ Wan 5B (Wan2.2):
#   max_state_dim=64, max_action_dim=32, num_views=3 (top_head, hand_left, hand_right).
# These constants are LOCAL defaults — callers override via __init__ args.
_DEFAULT_MAX_STATE_DIM = 64
_DEFAULT_MAX_ACTION_DIM = 32
_DEFAULT_STATE_HORIZON = 1
# embodiment_tag_to_projector_index["genie_sim"] from
# groot/vla/configs/model/dreamzero/transform/base.yaml.
_GENIE_SIM_EMBODIMENT_ID = 26

# Modality slice indices (must match meta/modality.json):
#   state.joint_position  = observation.state[54:68]   (14 dims)
#   state.left_effector_position  = observation.state[0:1]
#   state.right_effector_position = observation.state[1:2]
#   action.joint_position = action[16:30]              (14 dims)
#   action.left_effector_position  = action[0:1]
#   action.right_effector_position = action[1:2]
_STATE_SLICES = [
    ("joint_position", slice(54, 68)),
    ("left_effector_position", slice(0, 1)),
    ("right_effector_position", slice(1, 2)),
]
_ACTION_SLICES = [
    ("joint_position", slice(16, 30)),
    ("left_effector_position", slice(0, 1)),
    ("right_effector_position", slice(1, 2)),
]

# Native resolutions on genie_sim differ per view, so we mirror
# VideoPerViewResize(480x848) followed by VideoResize(image_resolution_height x
# image_resolution_width). For Wan 5B (Wan2.2) the data YAML uses
# image_resolution_width=320, image_resolution_height=160 — but the spec asks
# the OUTPUT of the 2x2 concat to be 320x640. So per-view target = 160x320 and
# concat doubles each axis (head row 160x320 + black row).
#
# The spec phrasing says "after running through VideoPerViewResize(480x848) and
# the cotrain's 2x2 grid concat (so final H=320, W=640, C=3 uint8)". H=320,
# W=640 = 2 * (160, 320), so each view is 160x320. We resize to 160x320 (skip
# the intermediate 480x848 — VideoPerViewResize's only purpose is to make views
# uniform BEFORE the random crop, but for eval we go straight to the final per-
# view size since we deterministically skip augmentation).
_PER_VIEW_H = 160
_PER_VIEW_W = 320


class _Normalizer:
    """q99 normalizer (mirrors groot.vla.data.transform.state_action.Normalizer).

    Implemented in numpy because the eval loader's outputs are numpy arrays —
    no need to round-trip through torch like the training transform does.
    """

    def __init__(self, q01: np.ndarray, q99: np.ndarray):
        self.q01 = q01.astype(np.float32)
        self.q99 = q99.astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        out = np.zeros_like(x)
        mask = self.q01 != self.q99
        # Broadcast q01/q99 across leading time dim.
        rng = (self.q99 - self.q01)
        # Avoid div-by-zero where mask is False; output will be replaced.
        rng_safe = np.where(mask, rng, np.ones_like(rng))
        normalized = (x - self.q01) / rng_safe
        normalized = 2.0 * normalized - 1.0
        out[..., mask] = normalized[..., mask]
        out[..., ~mask] = x[..., ~mask]
        out = np.clip(out, -1.0, 1.0)
        return out


def _build_state_normalizers(raw_stats: dict) -> dict[str, _Normalizer]:
    """Slice the raw 190-dim observation.state stats by modality.json indices.

    Mirrors lerobot.py::_get_metadata which does the same slicing per subkey.
    """
    raw = raw_stats["observation.state"]
    q01_full = np.asarray(raw["q01"], dtype=np.float32)
    q99_full = np.asarray(raw["q99"], dtype=np.float32)
    out = {}
    for subkey, sl in _STATE_SLICES:
        out[subkey] = _Normalizer(q01_full[sl], q99_full[sl])
    return out


def _build_action_normalizers(relative_stats: dict) -> dict[str, _Normalizer]:
    """Build per-subkey q99 normalizers from relative_stats_dreamzero.json.

    Training uses relative_action=True with relative_action_keys covering all
    three subkeys (joint / left_eff / right_eff) — see
    ``configs/data/dreamzero/genie_sim_relative_wan22.yaml`` — so we follow
    the same.
    """
    out = {}
    for subkey, _ in _ACTION_SLICES:
        s = relative_stats[subkey]
        out[subkey] = _Normalizer(np.asarray(s["q01"]), np.asarray(s["q99"]))
    return out


def _resize_video_frames(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize a (T, H, W, C) uint8 video to (T, h, w, C) using cv2 linear interp."""
    t, _, _, c = frames.shape
    out = np.empty((t, h, w, c), dtype=frames.dtype)
    for i in range(t):
        out[i] = cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _concat_2x2(views: list[np.ndarray]) -> np.ndarray:
    """Concatenate up to 3 views into a 2x2 grid (bottom-right black).

    Layout matches DreamTransform._prepare_video for the non-OXE_DROID branch
    (num_views >= 3 case used by genie_sim):
        top-left  = top_head    (view 0)
        bottom-left = hand_left  (view 1)
        top-right = hand_right  (view 2)
        bottom-right = zeros
    Inputs: list of (T, h, w, C) arrays, all same shape.
    Output: (T, 2h, 2w, C) uint8 array.
    """
    assert len(views) >= 1
    t, h, w, c = views[0].shape
    out = np.zeros((t, 2 * h, 2 * w, c), dtype=views[0].dtype)
    if len(views) > 0:
        out[:, :h, :w] = views[0]   # head (top-left)
    if len(views) > 1:
        out[:, h:, :w] = views[1]   # hand_left (bottom-left)
    if len(views) > 2:
        out[:, :h, w:] = views[2]   # hand_right (top-right)
    # bottom-right stays zero (matches DreamTransform black-pad behavior)
    return out


class GenieSimEvalLoader:
    """Deterministic validation episode iterator for genie_sim_gear.

    Yields one dict per episode, with the same keyset/shape contract as a
    single train sample (post-transform), so Agent 5's policy runner can feed
    samples in without a separate code path.
    """

    def __init__(
        self,
        dataset_root: str,
        split: str = "val",
        num_episodes: int = 4,
        num_frames: int = 33,
        action_horizon: int = 24,
        max_chunk_size: int = 4,
        fps: int = 30,
        seed: int = 1234,
    ):
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.num_episodes = int(num_episodes)
        self.num_frames = int(num_frames)
        self.action_horizon = int(action_horizon)
        self.max_chunk_size = int(max_chunk_size)
        self.state_horizon = _DEFAULT_STATE_HORIZON
        self.max_state_dim = _DEFAULT_MAX_STATE_DIM
        self.max_action_dim = _DEFAULT_MAX_ACTION_DIM
        self.fps = int(fps)
        self.seed = int(seed)
        self.embodiment_id = _GENIE_SIM_EMBODIMENT_ID

        # Load meta files (episodes index, task map, raw + relative stats).
        self._info = self._load_json("meta/info.json")
        self._tasks = self._load_tasks("meta/tasks.jsonl")
        self._episodes = self._load_episodes("meta/episodes.jsonl")
        self._raw_stats = self._load_json("meta/stats.json")
        self._rel_stats = self._load_json("meta/relative_stats_dreamzero.json")

        # Pre-build normalizers (state from raw observation.state slices; action
        # from per-subkey relative stats).
        self._state_normalizers = _build_state_normalizers(self._raw_stats)
        self._action_normalizers = _build_action_normalizers(self._rel_stats)

        # chunks_size used to map episode_index -> chunk subdirectory.
        self._chunks_size: int = int(self._info.get("chunks_size", 1000))

        # Deterministic episode selection: val = last N, train = first N.
        total = len(self._episodes)
        all_idx = sorted(ep["episode_index"] for ep in self._episodes)
        if self.split == "val":
            self._selected = all_idx[-self.num_episodes :]
        elif self.split == "train":
            self._selected = all_idx[: self.num_episodes]
        else:
            raise ValueError(f"split must be 'val' or 'train', got {split!r}")
        # Index episode metadata by episode_index for fast lookup.
        self._episode_meta = {e["episode_index"]: e for e in self._episodes}
        assert len(self._selected) == self.num_episodes, (
            f"Asked for {self.num_episodes} eps but only {total} available"
        )

    # -- meta loaders --------------------------------------------------------

    def _load_json(self, rel: str) -> dict:
        with open(self.dataset_root / rel, "r") as f:
            return json.load(f)

    def _load_tasks(self, rel: str) -> dict[int, str]:
        out = {}
        with open(self.dataset_root / rel, "r") as f:
            for line in f:
                row = json.loads(line)
                out[int(row["task_index"])] = row["task"]
        return out

    def _load_episodes(self, rel: str) -> list[dict]:
        eps = []
        with open(self.dataset_root / rel, "r") as f:
            for line in f:
                eps.append(json.loads(line))
        return eps

    # -- per-episode path helpers -------------------------------------------

    def _parquet_path(self, ep_idx: int) -> Path:
        chunk = ep_idx // self._chunks_size
        return (
            self.dataset_root
            / f"data/chunk-{chunk:03d}/episode_{ep_idx:06d}.parquet"
        )

    def _video_path(self, ep_idx: int, view_key: str) -> Path:
        chunk = ep_idx // self._chunks_size
        return (
            self.dataset_root
            / f"videos/chunk-{chunk:03d}/observation.images.{view_key}/episode_{ep_idx:06d}.mp4"
        )

    # -- per-sample materialization -----------------------------------------

    def _pick_start_frame(self, episode_length: int) -> int:
        """Deterministic start frame: middle of the episode if it leaves room.

        Falls back to 0 if the episode is short. The window must hold
        ``max(num_frames, action_horizon)`` frames going forward.
        """
        window = max(self.num_frames, self.action_horizon)
        if episode_length <= window:
            return 0
        # Mid-episode, clamped so the window stays in bounds.
        mid = episode_length // 2
        return max(0, min(mid, episode_length - window))

    def _read_video_window(
        self, ep_idx: int, view_key: str, start: int, n_frames: int
    ) -> np.ndarray:
        """Read n_frames contiguous frames starting at ``start`` from one view."""
        path = self._video_path(ep_idx, view_key)
        vr = decord.VideoReader(str(path))
        total = len(vr)
        # Clamp; pad-by-repeat at the tail if the episode is shorter than the
        # window (DreamTransform does the same "repeat last frame" trick).
        idx = np.arange(start, start + n_frames)
        idx = np.minimum(idx, total - 1)
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, C) uint8
        return frames

    def _build_state(self, parquet_state: np.ndarray, anchor_idx: int) -> np.ndarray:
        """Slice + normalize + pad state at the single anchor frame.

        Returns shape (max_chunk_size * state_horizon, max_state_dim). Only the
        first state_horizon rows hold real data; the rest is zero-padded to
        match DreamTransform._prepare_state's time-padding behavior.
        """
        row = parquet_state[anchor_idx : anchor_idx + self.state_horizon]  # (S, 190)
        pieces = []
        for subkey, sl in _STATE_SLICES:
            piece = row[:, sl].astype(np.float32)
            piece = self._state_normalizers[subkey].forward(piece)
            pieces.append(piece)
        concat = np.concatenate(pieces, axis=-1)  # (S, 16)
        # Pad channel dim up to max_state_dim and time dim up to
        # max_chunk_size * state_horizon (latter rows stay zero).
        s, d = concat.shape
        out = np.zeros(
            (self.max_chunk_size * self.state_horizon, self.max_state_dim),
            dtype=np.float32,
        )
        out[:s, :d] = concat
        return out

    def _build_action_gt(
        self, parquet_state: np.ndarray, parquet_action: np.ndarray, anchor_idx: int
    ) -> np.ndarray:
        """Build the ground-truth action sequence (relative + normalized).

        Returns shape (max_chunk_size * action_horizon, max_action_dim). The
        first ``action_horizon`` rows hold the real action for the single chunk
        anchored at ``anchor_idx``; rows beyond are zero (action mask would be
        False there in training, which we don't emit since this is just GT).
        """
        # Reference state for the relative-action computation is the state at
        # the anchor index (state_horizon=1 => delta_indices=[0]). See
        # lerobot_sharded.py::_convert_to_relative_action.
        ref_state_row = parquet_state[anchor_idx]  # (190,)

        # Pull the action chunk. Clamp into bounds; absolute action keys would
        # be first/last-padded in training but the action chunk we pick is
        # always in-bounds because _pick_start_frame guarantees the window
        # fits.
        end = anchor_idx + self.action_horizon
        end = min(end, parquet_action.shape[0])
        chunk = parquet_action[anchor_idx:end]  # (<=action_horizon, 36)
        if chunk.shape[0] < self.action_horizon:
            # Pad action time dim by repeating the last row (matches
            # padding_strategy="first_last" used for absolute action keys).
            pad = self.action_horizon - chunk.shape[0]
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], pad, axis=0)], axis=0)

        pieces = []
        for subkey, action_sl in _ACTION_SLICES:
            # state slice for the same subkey to compute reference.
            state_sl = dict(_STATE_SLICES)[subkey]
            ref = ref_state_row[state_sl].astype(np.float32)  # (D,)
            act = chunk[:, action_sl].astype(np.float32)  # (T, D)
            rel = act - ref  # broadcast: (T, D)
            rel = self._action_normalizers[subkey].forward(rel)
            pieces.append(rel)
        concat = np.concatenate(pieces, axis=-1)  # (T, 16)

        t, d = concat.shape
        out = np.zeros(
            (self.max_chunk_size * self.action_horizon, self.max_action_dim),
            dtype=np.float32,
        )
        out[:t, :d] = concat
        return out

    def _build_sample(self, ep_idx: int) -> dict:
        # Episode length comes from meta — avoids loading the parquet just to
        # check size.
        ep_meta = self._episode_meta[ep_idx]
        length = int(ep_meta["length"])
        start = self._pick_start_frame(length)

        # Load parquet once per episode.
        parquet_path = self._parquet_path(ep_idx)
        df = pd.read_parquet(parquet_path)
        state_all = np.stack(df["observation.state"].values)  # (L, 190)
        action_all = np.stack(df["action"].values)  # (L, 36)
        task_idx = int(df["task_index"].iloc[start])
        text = self._tasks.get(task_idx, "")

        # Video frames per view. Match the cotrain transform stack on Wan 5B:
        #   raw -> VideoPerViewResize(480x848) -> VideoResize(160x320).
        # At eval we skip the intermediate (no augmentation needed); resize
        # directly to the final per-view 160x320 since cv2.resize from raw is
        # equivalent up to interpolation rounding and avoids one cv2 pass.
        view_keys = ["top_head", "hand_left", "hand_right"]
        per_view = []
        for vk in view_keys:
            frames = self._read_video_window(ep_idx, vk, start, self.num_frames)
            frames = _resize_video_frames(frames, _PER_VIEW_H, _PER_VIEW_W)
            per_view.append(frames)
        images = _concat_2x2(per_view).astype(np.uint8)  # (T, 320, 640, 3)

        state = self._build_state(state_all, start)
        action_gt = self._build_action_gt(state_all, action_all, start)

        return {
            "images": images,
            "state": state,
            "action_gt": action_gt,
            "text": text,
            "embodiment_id": self.embodiment_id,
            "episode_index": int(ep_idx),
            "start_frame": int(start),
        }

    # -- iterator interface --------------------------------------------------

    def __len__(self) -> int:
        return self.num_episodes

    def __iter__(self) -> Iterator[dict]:
        for ep_idx in self._selected:
            yield self._build_sample(ep_idx)


if __name__ == "__main__":
    loader = GenieSimEvalLoader(
        dataset_root="/mnt/robot/youngdo/dreamzero_genie/dataset/place_object_into_box_color_g1_gear",
        num_episodes=2,
    )
    print(f"loader has {len(loader)} episodes; selected={loader._selected}")
    for sample in loader:
        print({k: (v.shape if hasattr(v, "shape") else v) for k, v in sample.items()})
