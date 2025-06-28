export MODEL_PATH="/scratch/ondemand28/harryscz/model/CogVideoX-2b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# export NCCL_SOCKET_IFNAME=^docker0,lo
# export CUDA_LAUNCH_BLOCKING=1

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file train/cap/train_config.yaml --multi_gpu \
  -m train.cap.train \
  --tracker_name "cogvideox" \
  --model_config_path train/model_config.yaml \
  --dataset_config_path "" \
  --load_checkpoint_if_exists 1 \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --seed 42 \
  --mixed_precision no \
  --output_dir modelOut/ \
  --stride_min 1 \
  --stride_max 3 \
  --downscale_coef 8 \
  --train_batch_size 2 \
  --dataloader_num_workers 4 \
  --num_train_epochs 20 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --report_to wandb \
  --sample_frames 29 