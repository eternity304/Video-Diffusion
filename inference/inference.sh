#!/bin/bash

export MODEL_PATH=/scratch/ondemand28/harryscz/model/CogVideoX-2b
export DATASET_PATH="/scratch/ondemand28/harryscz/head_audio/data/data256/uv"
export CHECKPOINT_PATH="/scratch/ondemand28/harryscz/head_audio/trainOutput/checkpoint-6000.pt"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/scratch/ondemand28/harryscz/diffusion/trition_cache_dir"

python -m inference.video_generation  \
    --model-config     inference/model_config.yaml \
    --pretrained-model-name-or-path $MODEL_PATH \
    --ckpt-path        $CHECKPOINT_PATH \
    --dataset-path $DATASET_PATH \
    --output-path      videoOut/5kStepOut.mp4 \
    --num-inference-steps  "50" \
    --sample-frames       "29" \
    --seed             "42" \
    --output-dir videoOut/
