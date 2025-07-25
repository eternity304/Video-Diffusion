unset https_proxy
export WANDB_PROJECT=WFVAE
export CUDA_VISIBLE_DEVICES=0,1
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_10:1,mlx5_11:1,mlx5_12:1,mlx5_13:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_IB_TC=162
# export NCCL_IB_TIMEOUT=22
# export NCCL_PXN_DISABLE=0
# export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_ALGO=Ring
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

EXP_NAME=WFVAE

torchrun \
    --nnodes=1 --nproc_per_node=2 \
    --master_addr=localhost \
    --master_port=12135 \
    -m train.vae.train_ddp \
    --exp_name ${EXP_NAME} \
    --pretrained_model_name_or_path /scratch/ondemand28/harryscz/other/WF-VAE/weight \
    --video_path /scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit \
    --eval_video_path /scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit \
    --ckpt_dir /scratch/ondemand28/harryscz/diffusion/modelOut/vae \
    --model_name WFVAE \
    --model_config train/vae/wfvae-large-16chn.json \
    --flame_path /scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz \
    --mix_precision fp32 \
    --resolution 256 \
    --num_frames 25 \
    --batch_size 1 \
    --lr 0.00001 \
    --epochs 4 \
    --disc_start 0 \
    --log_steps 1 \
    --save_ckpt_step 5000 \
    --eval_steps 1000 \
    --eval_batch_size 1 \
    --eval_num_frames 25 \
    --eval_sample_rate 1 \
    --eval_subset_size 100 \
    --eval_lpips \
    --ema \
    --ema_decay 0.999 \
    --perceptual_weight 1.0 \
    --loss_type l1 \
    --sample_rate 1 \
    --disc_cls model.causalvideovae.model.losses.LPIPSWithDiscriminator3D \
    --wavelet_loss \
    --wavelet_weight 0.1