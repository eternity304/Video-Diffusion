#!/usr/bin/env bash

clear

#SBATCH --job-name=video_diffusion
#SBATCH --partition=gpunodes
#SBATCH --nodes=1
#SBATCH --nodelist=tyche
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1         # if you need a GPU
#SBATCH --time=40:00:00  

export CONFIG_PATH=/scratch/ondemand28/harryscz/diffusion/train/acc_1.yaml
export MODEL_PATH=/scratch/ondemand28/harryscz/model/CogVideoX-2b
export MODEL_CONFIG=/scratch/ondemand28/harryscz/diffusion/train/model_config.yaml
export DATASET_PATH=/scratch/ondemand28/harryscz/head_audio/data/data256/uv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_VISIBLE_DEVICES=0,1
export TRITON_CACHE_DIR=/scratch/ondemand28/harryscz/diffusion/triton_cache_dir

# export ACCELERATE_DEBUG=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL       # or “INFO”
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL

echo "Starting training with Single GPUs..."
echo "Config: $CONFIG_PATH"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"

accelerate launch \
    --config_file $CONFIG_PATH \
    -m train.train_video_diffusion \
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
    --num-train-epochs 1000 \
    --lr-scheduler "cosine" \
    --lr-warmup-steps 500 \
    --learning-rate 0.0001 \
    --tracker-name "cogvideox" \
    --is-uncond false \
    --max-grad-norm 1.0 \
    --checkpointing-steps 1000 \
    --sample-frames 29 \
    # --chkpt_path /scratch/ondemand28/harryscz/head_audio/trainOutput/checkpoint-4000.pt
