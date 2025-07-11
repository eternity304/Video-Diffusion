#!/bin/bash

export MODEL_PATH=/scratch/ondemand28/harryscz/model/CogVideoX-2b
export DATASET_PATH="/scratch/ondemand28/harryscz/head_audio/data/data512/uv"
export CHECKPOINT_PATH=/scratch/ondemand28/harryscz/diffusion/modelOut/checkpoint-14000.pt
export TRITON_CACHE_DIR="/scratch/ondemand28/harryscz/diffusion/trition_cache_dir"

python -m inference.video_generation  \
    --model-config     train/model_config.yaml \
    --pretrained-model-name-or-path $MODEL_PATH \
    --ckpt-path        $CHECKPOINT_PATH \
    --dataset-path $DATASET_PATH \
    --output-path      videoOut/ref1kOut.mp4 \
    --num-inference-steps  "50" \
    --sample-frames       "29" \
    --seed             "42" \
    --output-dir videoOut/
