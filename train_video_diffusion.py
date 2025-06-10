from CustomDataset import VideoDataset
from cap_transformer import CAPVideoXTransformer3DModel
from trainUtils import *

from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import T5EncoderModel, T5Tokenizer

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from accelerate import Accelerator
from accelerate.logging import get_logger
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

def main():
    with open(args.model_config) as f: model_config_yaml = yaml.safe_load(f)
    torch.set_default_dtype(torch.float32)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    kwargs = DistributedDataParallelKwargs()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        print("Seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        import diffusers
        hf_logging.set_verbosity_info()
    else:
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        import diffusers
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

    if args.use_text:
        tokenizer = T5Tokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )
        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        text_encoder.requires_grad_(False)
    else:
        tokenizer = None
        text_encoder = None

    if model_config_yaml["use_audio"]:
        audio_model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            torch_dtype=torch.float32,
            attn_implementation="flash_attention_2",
        )
        audio_model.freeze_feature_encoder()
        audio_model.encoder.config.layerdrop = 0.
        audio_model.requires_grad_(True)
    else:
        audio_model = None

    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        subfolder="transformer",
        torch_dtype=torch.float32,
        revision=args.revision,
        variant=args.variant,
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
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    
    transformer = transformer.float()
    vae         = vae.float()

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # for p in transformer.parameters():
    # p.data = p.data.to(torch.float32)
    # for b in transformer.buffers():
    # b.data = b.data.to(torch.float32)

    num_ref = model_config_yaml['num_reference']
    num_target = model_config_yaml['num_target']
    height = model_config_yaml['height']
    width = model_config_yaml['width']

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

    transformer.to(accelerator.device)
    vae.to(accelerator.device)

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

    overrode_max_train_steps = False
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
        tracker_name = args.tracker_name or "cogvideox-diffusion"
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

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        def encode_video(video):
            # video is originally CPU float32 or uint8 → move to GPU float32
            video = video.to(accelerator.device)  # default dtype remains float32
            video = video.to(accelerator.device)  # default dtype remains float32
            with torch.no_grad():
                # Under AMP autocast, this conv runs in bf16 for speed, then returns a float32 output.
                # with torch.amp.autocast(dtype=torch.bfloat16, device_type="cuda"):
                latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
            # latent_dist is returned as float32 (AMP always returns float32 for downstream use)
            return latent_dist.contiguous()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate =  [transformer]
            with accelerator.accumulate(models_to_accumulate): 
                latent_chunks = []
                permuted_latents = []
                ref_mask_chunks = []
                cond_mask_chunks = []

                for chunk_id in range(len(batch["video_chunks"])):
                    raw_video = batch["video_chunks"][chunk_id]

                    # 1. Encode to latent (→ [B, C_z, F_lat, h_lat, w_lat]):
                    pre = encode_video(raw_video)
                    if args.is_uncond:
                        pre = pre * 0.0
                    latent_chunks.append(pre)
                    permuted_latents.append(pre.permute(0,2,1,3,4))

                    # 2. downsample reference mask to latent resolution
                    raw_ref_mask = batch["cond_chunks"]["ref_mask"][chunk_id].to(device=accelerator.device, dtype=weight_dtype)
                    B, C_z, F_lat, h_lat, w_lat = pre.shape
                    mask_down = torch.nn.functional.interpolate(
                        raw_ref_mask.unsqueeze(1), size=(F_lat, h_lat, w_lat), mode="nearest"
                    )
                    if args.is_uncond:
                        mask_down = mask_down * 0.0
                    ref_mask_chunks.append(mask_down.to(dtype=weight_dtype))
                    mask_cond = mask_down.permute(0, 2, 1, 3, 4)
                    cond_mask_chunks.append(mask_cond.to(dtype=weight_dtype))

                # 3. build cond chunks of length 1 from ref_mask_chunks
                cond_chunks = [ rm for rm in cond_mask_chunks ]

                # Sample and forward noise
                B, C_z, F, h_z, w_z = latent_chunks[0].shape
                timesteps = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (B,),
                    device=accelerator.device,
                ).long()

                noisy_preperm_list = []
                for chunk_id, clean_latent in enumerate(latent_chunks):
                    # clean_latent = [B, C_z, F_lat, h, w]
                    noise = torch.randn_like(clean_latent)
                    noisy = scheduler.add_noise(clean_latent, noise, timesteps)
                    noisy = noisy.to(dtype=weight_dtype)
                    rm = ref_mask_chunks[chunk_id]  # [B, 1, F_lat, h, w]
                    one_bf16 = torch.ones_like(rm)
                    merged = noisy * (one_bf16 - rm) + clean_latent * rm
                    merged = merged.to(dtype=weight_dtype)  
                    noisy_preperm_list.append(merged.to(dtype=weight_dtype))

                noisy_latents = [m.permute(0, 2, 1, 3, 4) for m in noisy_preperm_list]

                # Get Sequence info
                sequence_infos = []
                for chunk_id, clean_latent in enumerate(latent_chunks):
                    F_lat = clean_latent.shape[2]
                    seq_idx = torch.arange(0, F_lat, device=accelerator.device)
                    sequence_infos.append((False, seq_idx))

                inner_dim = accelerator.unwrap_model(transformer).config.num_attention_heads * accelerator.unwrap_model(transformer).config.attention_head_dim  # 30×64=1920
                fake_text_embeds = torch.zeros((B, 1, inner_dim), dtype=weight_dtype, device=accelerator.device)

                # 5b) Fake audio embeddings:
                audio_feature_dim  = 768   # _must_ match what Wav2Vec would have produced
                fake_audio_embeds  = torch.zeros(
                    (B, F_lat, audio_feature_dim),
                    dtype=weight_dtype,
                    device=accelerator.device,
                )
                
                model_output = transformer(
                    hidden_states=noisy_latents,
                    condition=cond_chunks,
                    sequence_infos=sequence_infos,
                    timestep=timesteps,
                    audio_embeds=fake_audio_embeds,
                    encoder_hidden_states=fake_text_embeds,
                    image_rotary_emb=None,
                    return_dict=False,
                )[0]

                # compute loss on non-reference pixels
                ref_mask = torch.cat(ref_mask_chunks, dim=2)
                non_ref_mask = 1 - ref_mask
                non_ref_mask = non_ref_mask.permute(0, 2, 1, 3, 4)  

                model_output = torch.cat(model_output, dim=1)
                model_input = torch.cat(permuted_latents, dim=1)
                noisy_input = torch.cat(noisy_latents, dim=1)

                model_pred = scheduler.get_velocity(model_output, noisy_input, timesteps)
                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1.0 / (1.0 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                            weights = weights.unsqueeze(-1)

                target = torch.cat(permuted_latents, dim=1)
                loss = weights * ((model_pred - target) ** 2)
                denom = non_ref_mask.mean().clamp(min=1e-6)
                loss = (loss * non_ref_mask) / denom
                loss = loss.reshape(B, -1).mean(dim=1).mean()


                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    if accelerator.scaler is not None: accelerator.scaler.unscale_(optimizer)
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step(global_step)
                    global_step += 1
                    progress_bar.update(1)
                    accelerator.log({"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step)

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

                    if global_step >= args.max_train_steps:
                        break

    accelerator.wait_for_everyone()
    accelerator.end_training()

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