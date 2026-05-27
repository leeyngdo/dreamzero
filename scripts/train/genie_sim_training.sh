#!/bin/bash
# DreamZero Genie Sim G1 Full Fine-Tuning Script (8x RTX PRO 6000 Blackwell, ZeRO-2 + CPU Offload)
#
# Usage:
#   # Defaults below assume the on-machine /mnt/robot layout.
#   # Override env vars if your paths differ:
#   bash scripts/train/genie_sim_training.sh
#
# Prerequisites:
#   - Genie Sim G1 dataset in LeRobot/GEAR format at GENIE_SIM_DATA_ROOT
#     (state 64, action with joint_position + left_effector_position + right_effector_position, 3 views)
#   - Wan2.1-I2V-14B-480P weights at WAN_CKPT_DIR
#   - umt5-xxl tokenizer at TOKENIZER_DIR
#   - DreamZero-AgiBot pretrained checkpoint at PRETRAINED_DIR (used as warm-start; full FT from here)
#   - WANDB_API_KEY exported in the shell environment (for wandb logging)

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
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/robot/youngdo/dreamzero_genie/output/dreamzero_genie_sim_full_20k}

# Model weight paths
WAN_CKPT_DIR=${WAN_CKPT_DIR:-/mnt/robot/youngdo/dreamzero_genie/checkpoints/Wan2.1-I2V-14B-480P}
TOKENIZER_DIR=${TOKENIZER_DIR:-/mnt/robot/youngdo/dreamzero_genie/checkpoints/umt5-xxl}

# Pretrained DreamZero-AgiBot checkpoint to warm-start full fine-tune
PRETRAINED_DIR=${PRETRAINED_DIR:-/mnt/robot/youngdo/dreamzero_genie/checkpoints/DreamZero-AgiBot}

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
    echo "Set GENIE_SIM_DATA_ROOT to your LeRobot/GEAR-format Genie Sim G1 dataset."
    exit 1
fi

if [ ! -d "$WAN_CKPT_DIR" ]; then
    echo "ERROR: Wan2.1-I2V-14B-480P checkpoint not found at $WAN_CKPT_DIR"
    exit 1
fi

if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "ERROR: umt5-xxl tokenizer not found at $TOKENIZER_DIR"
    exit 1
fi

if [ ! -d "$PRETRAINED_DIR" ]; then
    echo "ERROR: DreamZero-AgiBot pretrained checkpoint not found at $PRETRAINED_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
# =====================================

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/genie_sim_relative \
    wandb_project=dreamzero \
    +wandb_run_name=dreamzero_genie_sim_full_20k \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2_offload.json" \
    save_steps=2500 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
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
    image_resolution_height=176 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    genie_sim_data_root=$GENIE_SIM_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=$PRETRAINED_DIR \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
