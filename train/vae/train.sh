
accelerate launch --config_file train/cap/train_config.yaml --multi_gpu \
    -m train.vae.train \
    --pretrained-model-name-or-path /scratch/ondemand28/harryscz/model/CogVideoX-2b \
    --dataset-path /scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit \
    --output-dir /scratch/ondemand28/harryscz/diffusion/videoOut \
    --batch-size 2 \
    --sample-frames 5 \
    --lr 0.0001



