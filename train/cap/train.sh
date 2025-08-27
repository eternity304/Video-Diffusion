export MODEL_PATH="/scratch/ondemand28/harryscz/model/CogVideoX-2b"
export VAE_PATH="/scratch/ondemand28/harryscz/diffusion/modelOut/vae/WFVAE-lr1.00e-05-bs1-rs256-sr1-fr25/checkpoint-44000.ckpt"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NCCL_TOPO_DUMP_FILE=./topo.xml
export NCCL_TOPO_DUMP_FILE_MODE=GROUP
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=

# export CUDA_LAUNCH_BLOCKING=1 
# export TORCH_SHOW_CPP_STACKTRACES=1


# For Debug
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export NCCL_ALGO=Ring
export NCCL_COLLNET_DISABLE=1

unset NCCL_SHM_DISABLE       # disable /dev/shm transport  
export NCCL_P2P_DISABLE=1       # disable direct P2P, force host staging  

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file train/cap/train_config.yaml --multi_gpu \
  -m train.cap.train \
  --tracker_name "cogvideox" \
  --model_config_path train/model_config.yaml \
  --audio_model_path /scratch/ondemand28/harryscz/model/wav2vec2-base \
  --dataset_config_path "" \
  --load_checkpoint_if_exists 0 \
  --pretrained_model_name_or_path $MODEL_PATH \
  --vae_ckpt $VAE_PATH \
  --enable_tiling \
  --enable_slicing \
  --seed 42 \
  --mixed_precision no \
  --output_dir modelOut/diffusion \
  --stride_min 1 \
  --stride_max 3 \
  --downscale_coef 8 \
  --train_batch_size 1 \
  --dataloader_num_workers 1 \
  --num_train_epochs 30 \
  --checkpointing_steps 500 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --report_to wandb \
  --sample_frames 50 