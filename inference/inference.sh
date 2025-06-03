#!/bin/bash

clear

export MODEL_PATH="/scratch/ondemand28/harryscz/model/CogVideoX-2b"
export MODEL_CONFIG="/scratch/ondemand28/harryscz/diffusion/model_config.yaml"
export DATASET_PATH="/scratch/ondemand28/harryscz/head_audio/data/data256/uv"
export CHECKPOINT_PATH="/scratch/ondemand28/harryscz/head_audio/trainOutput/checkpoint-1000.pt"
export OUTPUT_DIR="/scratch/ondemand28/harryscz/diffusion/videoOut"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="/scratch/ondemand28/harryscz/diffusion/trition_cache_dir"

# Path to your inference script
INFERENCE_SCRIPT="/scratch/ondemand28/harryscz/diffusion/inference/inference.py"

python "$INFERENCE_SCRIPT" \
    --model-config     "$MODEL_CONFIG" \
    --pretrained-model "$MODEL_PATH" \
    --checkpoint       "$CHECKPOINT_PATH" \
    --reference-frames "$DATASET_PATH" \
    --output           "$OUTPUT_DIR/generated.mp4" \
    --device           "cuda" \
    --num-inference-steps  "50" \
    --height           "256" \
    --width            "256" \
    --num-frames       "25" \
    --seed             "42"
