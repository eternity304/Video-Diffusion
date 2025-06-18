#!/bin/bash

clear

export CONFIG_PATH=/scratch/ondemand28/harryscz/diffusion/train/accelerate_config_machine_single.yaml
export MODEL_PATH=/scratch/ondemand28/harryscz/model/CogVideoX-2b
export MODEL_CONFIG=/scratch/ondemand28/harryscz/diffusion/train/model_config.yaml
export DATASET_PATH=/scratch/ondemand28/harryscz/head_audio/data/data256/uv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/scratch/ondemand28/harryscz/diffusion/triton_cache_dir

accelerate launch --config_file $CONFIG_PATH /scratch/ondemand28/harryscz/diffusion/train/train_video_diffusion.py \
    --dataset-path $DATASET_PATH \
    --pretrained-model-name-or-path $MODEL_PATH \
    --gradient-accumulation-steps 2 \
    --mixed-precision "no" \
    --report-to "wandb" \
    --logging-dir "/scratch/ondemand28/harryscz/head_audio/trainlog" \
    --output-dir "/scratch/ondemand28/harryscz/head_audio/trainOutput" \
    --seed 42 \
    --batch-size 1 \
    --use-text false \
    --model-config $MODEL_CONFIG \
    --enable-tiling true \
    --enable-slicing true \
    --num-train-epochs 3000 \
    --lr-scheduler "cosine" \
    --lr-warmup-steps 500 \
    --learning-rate 0.0001 \
    --tracker-name "cogvideox" \
    --is-uncond false \
    --max-grad-norm 1.0 \
    --checkpointing-steps 200 \






