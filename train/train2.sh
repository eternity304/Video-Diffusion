#!/bin/bash

clear

export CONFIG_PATH=/scratch/ondemand28/harryscz/diffusion/train/acc_2.yaml
export MODEL_PATH=/scratch/ondemand28/harryscz/model/CogVideoX-2b
export MODEL_CONFIG=/scratch/ondemand28/harryscz/diffusion/train/model_config.yaml
export DATASET_PATH=/scratch/ondemand28/harryscz/head_audio/data/data256/uv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=8,9
export TRITON_CACHE_DIR=/scratch/ondemand28/harryscz/diffusion/triton_cache_dir

export ACCELERATE_LOG_LEVEL=DEBUG       # turn on DEBUG logs
export ACCELERATE_DEBUG_MODE=1         # also enable operational debug mode
export NCCL_DEBUG=INFO                  # if you’re using NCCL for collectives
export TORCH_DISTRIBUTED_DEBUG=DETAIL

echo "Starting training with 2 GPUs..."
echo "Config: $CONFIG_PATH"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"

accelerate launch --debug \
    --config_file $CONFIG_PATH \
    --use_deepspeed \
    -m train.train_video_diffusion \
    --dataset-path $DATASET_PATH \
    --pretrained-model-name-or-path $MODEL_PATH \
    --gradient-accumulation-steps 2 \
    --mixed-precision "no" \
    --report-to "wandb" \
    --logging-dir "/scratch/ondemand28/harryscz/head_audio/trainlog2" \
    --output-dir "/scratch/ondemand28/harryscz/head_audio/trainOutput2" \
    --seed 526 \
    --batch-size 1 \
    --use-text false \
    --model-config $MODEL_CONFIG \
    --enable-tiling true \
    --enable-slicing true \
    --num-train-epochs 100 \
    --lr-scheduler "cosine" \
    --lr-warmup-steps 500 \
    --learning-rate 0.0001 \
    --tracker-name "cogvideox" \
    --is-uncond false \
    --max-grad-norm 1.0 \
    --checkpointing-steps 500 \
    --sample-frames 29