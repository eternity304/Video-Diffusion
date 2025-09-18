#!/bin/bash

export MODEL_PATH=/scratch/ondemand28/harryscz/model/CogVideoX-2b
export DATASET_PATH="/scratch/ondemand28/harryscz/head_audio/data/data512/uv"
export CHECKPOINT_PATH=/scratch/ondemand28/harryscz/diffusion/modelOut/diffusion/checkpoint-10000.pt
export TRITON_CACHE_DIR="/scratch/ondemand28/harryscz/diffusion/trition_cache_dir"
export VAE_PATH="/scratch/ondemand28/harryscz/diffusion/modelOut/vae/WFVAE-lr1.00e-05-bs1-rs256-sr1-fr25/checkpoint-44000.ckpt"

python -m inference.video_generation  \
    --model-config     train/model_config.yaml \
    --pretrained-model-name-or-path $MODEL_PATH \
    --audio-model-path /scratch/ondemand28/harryscz/model/wav2vec2-base \
    --ckpt-path        $CHECKPOINT_PATH \
    --dataset-path $DATASET_PATH \
    --vae-ckpt $VAE_PATH \
    --output-path      videoOut/ref1kOut.mp4 \
    --num-inference-steps  "50" \
    --sample-frames      50 \
    --seed             "42" \
    --output-dir videoOut/
