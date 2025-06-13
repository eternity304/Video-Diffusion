from data.VideoDataset import VideoDataset
from model.cap_transformer import CAPVideoXTransformer3DModel
from train.trainUtils import *

from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from diffusers.optimization import get_scheduler

import torch
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger

import logging
import yaml
import wandb
import numpy as np
import random
import math
from tqdm.auto import tqdm
import os
import argparse
import sys

os.chdir("..")

def main():
    # Load Model Config
    with open(args.model_config) as f: model_config_yaml = yaml.safe_load(f)

    # Setup Accelerator
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[kwargs],
        )

    # Setup Seeds
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        print("Seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup Logger
    logger = get_logger(__name__)
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_warning()
        hf_logging.set_verbosity_info()
    else:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        hf_logging.set_verbosity_error()

    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_dataset = VideoDataset(
        videos_dir=args.dataset_path,
        num_ref_frames=1,
        num_target_frames=49
    )
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=lambda x: x[0],
        worker_init_fn=worker_init_fn,
        generator=g
    )

    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        subfolder="transformer",
        torch_dtype=torch.float32,
        cond_in_channels=1,  # only one channel (the ref_mask)
        sample_width=model_config_yaml["width"] // 8,
        sample_height=model_config_yaml["height"] // 8,
        max_text_seq_length=1,
        max_n_references=model_config_yaml["max_n_references"],
        apply_attention_scaling=model_config_yaml["use_growth_scaling"],
        use_rotary_positional_embeddings=False,
    )

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.float32
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        prediction_type="v_prediction"
    )

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    transformer.requires_grad_(True)
    vae.requires_grad_(False)

    if accelerator.state.deepspeed_plugin:
        ds_cfg = accelerator.state.deepspeed_plugin.deepspeed_config
        if "fp16" in ds_cfg and ds_cfg["fp16"]["enabled"]:
            weight_dtype = torch.float16
        elif "bf16" in ds_cfg and ds_cfg["bf16"]["enabled"]:
            weight_dtype = torch.float16
        else:
            weight_dtype = torch.float32
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            weight_dtype = torch.float32

    def unwrap_model(m):
        m = accelerator.unwrap_model(m)
        return m._orig_mod if hasattr(m, "_orig_mod") else m

    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    trainable_parameters =  list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": trainable_parameters, "lr": args.learning_rate}]
    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    optimizer = get_optimizer(
        learning_rate=args.learning_rate,
        adam_beta1=0.9, 
        adam_beta2=0.95, 
        adam_epsilon=1e-8, 
        adam_weight_decay=1e-4, 
        params_to_optimize=params_to_optimize, 
        use_deepspeed=use_deepspeed_optimizer
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Build lr_scheduler
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    if use_deepspeed_scheduler:
        # Let DeepSpeed handle scheduling
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,  # handled by deepspeed
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )
    else:
        # Normal HF scheduler
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,  # placeholder, will be replaced after prepare()
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
        )

    transformer, optimizer, lr_scheduler, train_dataloader = accelerator.prepare(transformer, optimizer, lr_scheduler, train_dataloader)
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "video-diffusion"
        accelerator.init_trackers(tracker_name, config={"dropout": 0.0, "learning_rate": args.learning_rate})

    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters    = {num_trainable_parameters}")
    logger.info(f"  Num examples                = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch      = {len(train_dataloader)}")
    logger.info(f"  Num epochs                  = {args.num_train_epochs}")
    logger.info(f"  Batch size per device       = {args.batch_size}")
    logger.info(f"  Total train batch size      = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps    = {args.max_train_steps}")

    first_epoch = 0
    initial_global_step = 0
    global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    torch.cuda.empty_cache()
    model_conf = transformer.module.config if hasattr(transformer, "module") else transformer.config

    def encode_video(vae, video):
        with torch.no_grad():
            dist = vae.encode(video).latent_dist.sample()
        latent = dist * vae.config.scaling_factor
        return latent.permute(0,2,1,3,4).contiguous()

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate =  [transformer]
            with accelerator.accumulate(models_to_accumulate): 
                latent_chunks = []
                ref_mask_chunks = []

                # Initialize necessary data for diffusion
                for i, video in enumerate(batch["video_chunks"]):
                    # Initiate the Remainder Frames As Gaussian Noise
                    # Initial video has shape [B, C, F, H, W]
                    # video[:, :, 1:, :, :] = torch.randn(video[:, :, 1:, :, :].shape)
                    video = video.to(accelerator.device).to(weight_dtype)

                    # Encode Video
                    latent = encode_video(vae, video) # [B, F, C_z, H, W]
                    latent_chunks.append(latent)

                    # Ref Mask Chunk, Mask of shape [B, F, H, W, C]
                    B, F_z, C, H, W = latent.shape
                    rm = torch.zeros((B, F_z, 1, H, W), device=accelerator.device, dtype=weight_dtype)
                    rm[:, 0] = 1.0
                    ref_mask_chunks.append(rm)

                # Sequence Info, Sequence of Bool suggesting which chunk is used as reference
                # Here, all are not reference
                sequence_infos = [[False, torch.arange(chunk.shape[1])]for chunk in latent_chunks]
                
                # Sample Random Noise
                B, F_z, C_z, H_z, W_z = latent_chunks[0].shape
                timesteps = torch.randint(
                    1,
                    scheduler.config.num_train_timesteps,
                    (B,),
                    device=accelerator.device
                ).long()

                # Noise Latent
                # noised_latents = []
                # for i, video in enumerate(batch["video_chunks"]):
                #     video = video.to(accelerator.device).to(weight_dtype)
                #     noise = torch.randn_like(video)
                #     noise[:, :, 0, :, :] = 0
                #     video[:, :, 1:, :, :] = scheduler.add_noise(video[:, :, 1:, :, :], noise[:, :, 1:, :, :], timesteps).to(weight_dtype)
                #     noised_latents.append(encode_video(vae, video))
                noised_latents = []
                for idx, latent in enumerate(latent_chunks):
                    noise = torch.randn_like(latent, device=accelerator.device, dtype=weight_dtype)
                    noisy_latent = scheduler.add_noise(latent, noise, timesteps)
                    noised_latents.append(noisy_latent)

                # Trivial Audio, Text, and Condition
                audio_embeds = torch.zeros((B, F_z, 768), dtype=weight_dtype, device=accelerator.device)
                text_embeds  = torch.zeros((B, 1,
                    unwrap_model(transformer).config.attention_head_dim * unwrap_model(transformer).config.num_attention_heads
                ), dtype=weight_dtype, device=accelerator.device)
                B, F_z, C_z, H_z, W_z = noised_latents[0].shape
                zero_cond = [torch.zeros((B, F_z, 1, H_z, W_z), dtype=weight_dtype, device=accelerator.device)] * len(noised_latents)

                # Predict Noise
                model_outputs = transformer(
                    hidden_states=noised_latents,
                    encoder_hidden_states=text_embeds,
                    audio_embeds=audio_embeds,
                    condition=zero_cond,
                    timestep=timesteps,
                    sequence_infos=sequence_infos,
                    image_rotary_emb=None,
                    return_dict=False
                )[0]

                ref_mask = torch.cat(ref_mask_chunks, dim=1)
                non_ref_mask = 1. - ref_mask

                model_output = torch.cat(model_outputs, dim=1)
                model_input = torch.cat(latent_chunks, dim=1)
                noisy_input = torch.cat(noised_latents, dim=1)

                # print("model_output", model_output.min(), model_output.max())
                model_pred = scheduler.get_velocity(model_output, noisy_input, timesteps)

                alpha_bar = scheduler.alphas_cumprod[timesteps].to(weight_dtype)
                sigma_bar = (1 - alpha_bar).sqrt()
                eps = (model_input - alpha_bar.sqrt() * noisy_input) / sigma_bar
                v_true = alpha_bar.sqrt() * eps - sigma_bar * model_input
                loss = F.mse_loss(model_pred, v_true)

                # alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                # weights = 1 / (1 - alphas_cumprod)
                # while len(weights.shape) < len(model_pred.shape):
                #     weights = weights.unsqueeze(-1)

                # target = model_input

                # loss = (weights * (model_pred - target) ** 2)
                # loss = torch.mean(loss.reshape(B, -1), dim=1)
                # loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if accelerator.scaler is not None: accelerator.scaler.unscale_(optimizer)
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step(global_step)
                    global_step += 1
                    progress_bar.update(1)

                    # Save checkpoint every args.checkpointing_steps
                    if accelerator.is_main_process and (global_step % args.checkpointing_steps == 0):
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        save_dict = {
                            "state_dict": unwrap_model(transformer).state_dict(),
                            "optimizer":  optimizer.state_dict(),
                            "global_step": global_step,
                            "epoch":      epoch,
                        }
                        torch.save(save_dict, save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                    accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

                    if global_step >= args.max_train_steps:
                        break

def get_args():
    parser = argparse.ArgumentParser(description="CogVideoX Training Script")

    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--pretrained-model-name-or-path", type=str, required=True)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", required=True)
    parser.add_argument("--report-to", type=str, default="wandb", required=True)
    parser.add_argument("--logging-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42, required=True)
    parser.add_argument("--batch-size", type=int, default=1, required=True)
    parser.add_argument("--use-text", type=bool, default=False, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--enable-slicing", type=bool, default=False, required=True)
    parser.add_argument("--enable-tiling", type=bool, default=False, required=True)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=1, required=True)
    parser.add_argument("--lr-scheduler", required=True)
    parser.add_argument("--lr-warmup-steps", type=int, default=500, required=True)
    parser.add_argument("--lr-num-cycles", type=int, default=1)
    parser.add_argument("--lr-power", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4, required=True)
    parser.add_argument("--tracker-name", type=str, default=None, required=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0, required=True)
    parser.add_argument("--checkpointing-steps", type=int, default=500, required=True)
    parser.add_argument("--is-uncond", type=bool, default=False, required=True)


    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main()