#!/bin/bash
# DreamZero Genie Sim G1 Full Fine-Tuning Script with Wan2.2-TI2V-5B backbone
# (8x RTX PRO 6000 Blackwell, ZeRO-2)
#
# Usage:
#   # Defaults below assume the on-machine /mnt/robot layout.
#   bash scripts/train/genie_sim_training_wan22.sh
#
# Prerequisites:
#   - Genie Sim G1 dataset in LeRobot/GEAR format at GENIE_SIM_DATA_ROOT
#     (state 64, action with joint_position + left_effector_position + right_effector_position, 3 views)
#   - Wan2.2-TI2V-5B weights at WAN22_CKPT_DIR (DiT + T5 + VAE; no CLIP)
#   - Wan2.1-I2V-14B-480P weights at IMAGE_ENCODER_DIR (used ONLY for CLIP - Wan2.2 doesn't ship it)
#   - umt5-xxl tokenizer at TOKENIZER_DIR
#   - WANDB_API_KEY exported in the shell environment (for wandb logging)
#
# NOTE: Wan2.2-TI2V-5B has a different architecture (dim=3072 vs 5120, 30 layers vs 32,
# 48-channel VAE16x vs 16-channel VAE8x). The DreamZero-AgiBot pretrained checkpoint is
# Wan2.1-14B-compatible only, so we do NOT load it here. Training starts from the
# Wan2.2 backbone weights and learns the action head from scratch.
# See docs/WAN22_BACKBONE.md for full architecture differences.

export HYDRA_FULL_ERROR=1

# ============ CACHE / TMP DIRS (keep root disk free) ============
export HF_HOME=${HF_HOME:-/mnt/robot/youngdo/hf-cache}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/mnt/robot/youngdo/hf-cache/hub}
export WANDB_DIR=${WANDB_DIR:-/mnt/robot/youngdo/dreamzero_genie/output/wandb}
export TMPDIR=${TMPDIR:-/mnt/robot/youngdo/dreamzero_genie/tmp}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$WANDB_DIR" "$TMPDIR"
# ================================================================

# ============ USER CONFIGURATION ============
# Dataset path (Genie Sim G1 in LeRobot/GEAR format)
GENIE_SIM_DATA_ROOT=${GENIE_SIM_DATA_ROOT:-/mnt/robot/youngdo/dreamzero_genie/dataset/place_object_into_box_color_g1_gear}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/robot/youngdo/dreamzero_genie/output/dreamzero_genie_sim_full_20k_wan22}

# Wan2.2-TI2V-5B checkpoint (contains DiT, T5 text encoder, and 48ch VAE)
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-/mnt/robot/youngdo/dreamzero_genie/checkpoints/Wan2.2-TI2V-5B}

# Image encoder: Wan2.2-TI2V-5B does NOT include CLIP - reuse Wan2.1's
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-/mnt/robot/youngdo/dreamzero_genie/checkpoints/Wan2.1-I2V-14B-480P}

TOKENIZER_DIR=${TOKENIZER_DIR:-/mnt/robot/youngdo/dreamzero_genie/checkpoints/umt5-xxl}

# Number of GPUs (default: all visible GPUs; final fallback is 8)
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}
# =============================================

# ============ ENV CHECKS ============
if [ -z "${WANDB_API_KEY}" ]; then
    echo "WARNING: WANDB_API_KEY is not set in the environment. wandb logging will likely fail."
    echo "         Export WANDB_API_KEY=... before running, or pass report_to=none."
fi

if [ ! -d "$GENIE_SIM_DATA_ROOT" ]; then
    echo "ERROR: Genie Sim dataset not found at $GENIE_SIM_DATA_ROOT"
    exit 1
fi

if [ ! -d "$WAN22_CKPT_DIR" ]; then
    echo "ERROR: Wan2.2-TI2V-5B checkpoint not found at $WAN22_CKPT_DIR"
    echo "Download: huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir $WAN22_CKPT_DIR"
    exit 1
fi

if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "ERROR: CLIP image encoder not found at $IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"
    echo "Wan2.2 does not ship CLIP. Download Wan2.1-I2V-14B-480P to get it:"
    echo "  huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir $IMAGE_ENCODER_DIR"
    exit 1
fi

if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "ERROR: umt5-xxl tokenizer not found at $TOKENIZER_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
# =====================================

# Notes on the overrides below (vs. the Wan2.1-14B version):
#   - data=dreamzero/genie_sim_relative_wan22  (sets image_resolution_height=160 for even latent)
#   - model/dreamzero/action_head=wan_flow_matching_action_tf_wan22  (5B DiT + VAE38)
#   - deepspeed=zero2.json (no CPU offload; 5B fits comfortably on 97GB cards)
#   - NO frame_seqlen override (the wan22 action_head config sets it to 50 internally)
#   - NO pretrained_model_path (no DreamZero-Wan2.2 ckpt; fresh start)
torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/genie_sim_relative_wan22 \
    wandb_project=dreamzero \
    +wandb_run_name=dreamzero_genie_sim_full_20k_wan22 \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=2500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=32 \
    max_steps=20000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=160 \
    save_lora_only=false \
    max_chunk_size=4 \
    save_strategy=steps \
    genie_sim_data_root=$GENIE_SIM_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    eval_cfg.enable=true \
    eval_cfg.dataset_root=$GENIE_SIM_DATA_ROOT \
    eval_cfg.eval_every=100 \
    eval_cfg.num_episodes=4 \
    eval_cfg.num_inference_steps=4
