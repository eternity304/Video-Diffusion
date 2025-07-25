import argparse
import os

import math
import yaml
import logging
import random
import numpy as np
import sys
import imageio
import wandb
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger

from model.flameObj import *
from data.VideoDataset import *

from diffusers import AutoencoderKLCogVideoX

def main(args):

    dtype = torch.float32

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                        logging_dir=os.path.join(args.output_dir, "logs"))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision="no",
                                project_config=accelerator_project_config,
                                kwargs_handlers=[ddp_kwargs])

    # 2.4 Set random seed
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info("Accelerator state:", accelerator.state)

    dataset = VideoPathDataset(
        source_dir=args.dataset_path,
    )
    if args.shuffle:
        sampler = DistributedSampler(
            dataset,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            shuffle=True
        )
    else:
        sampler = None
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # sampler=sampler,
        collate_fn=lambda x: x,   
        num_workers=2,
        pin_memory=True,
    )
    logger.info(f"Number of test examples: {len(data_loader)}")

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    ).to(dtype)
    vae, data_loader = accelerator.prepare(vae, data_loader)
    
    logger.info(f"VAE Loaded")

    flamePath = "/scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz"
   
    head = Flame(flamePath, device="cuda")

    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-5)

    epoch = 20
    batch_per_epoch = len(data_loader)
    train_steps = epoch * batch_per_epoch
    save_every = 2

    wandb.init(
        project="VAE",  # change this to your actual project
        config={
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "Number of Data": len(data_loader),
        }
    )

    progress_bar = tqdm(
        range(0, train_steps),
        initial=1,
        desc="Steps",
    )

    for i in range(epoch):
        for step, batch in enumerate(data_loader):
            uvs = head.batch_uv(batch, resolution=256, rotation=False, sample_frames=29).permute(0,1,4,2,3) 

            uvs = uvs.to(accelerator.device, dtype=dtype)
            uvs = uvs.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
            z = vae.module.encode(uvs)
            latents = z.latent_dist.sample() * vae.config.scaling_factor
            latents = 1 / vae.config.scaling_factor * latents
            recon = vae.module.decode(latents).sample

            kl = compute_kl_loss(z).sum()
            l1 = F.l1_loss(uvs, recon, reduction='mean')

            loss = 0.0001 * kl + l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress_bar.update(1)
            progress_bar.set_postfix(**log)

            log = {
                "loss" : f"{loss.item():.4f}",
                "KL" : f"{kl.item():.4f}",
                "L1" : f"{l1.item():.4f}"
            }

            wandb.log({
                "loss": loss.item(),
                "l1": l1.item(),
                "kl": kl.item()
            })

            if (i * batch_per_epoch + step) % save_every == 0:
                torch.save(vae.state_dict(), f"modelOut/checkpoint_{i}.pt")
                logger.info(f"Models Saved at Step {i * batch_per_epoch + step}")


def compute_kl_loss(z):
    mu = z.latent_dist.mean
    logvar = z.latent_dist.logvar
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / mu.numel()

def encode_video(vae, video, grad=False):
    video = video.to(accelerator.device, dtype=vae.dtype)
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

    if grad:
        latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    else: 
        with torch.no_grad(): latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)

def decode_latents(vae, latents: torch.Tensor, grad=False) -> torch.Tensor:
    latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    latents = 1 / vae.config.scaling_factor * latents

    if grad:
        frames = vae.decode(latents).sample
    else:
        with torch.no_grad(): frames = vae.decode(latents).sample
    return frames.permute(0,2,1,3,4)

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(
        description="Unconditioned Video Diffusion Inference"
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Directory containing input reference videos."
    )
    parser.add_argument(
        "--pretrained-model-name-or-path", type=str, required=True,
        help="Path or HF ID where transformer/vae/scheduler are stored."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size per device (usually 1 for inference)."
    )
    parser.add_argument(
        "--sample-frames", type=int, default=50
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Where to write generated videos."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--shuffle", type=int, default=False,
        help="Whether to shuffle dataset. Usually False for inference."
    )

    # If arg_list is None, argparse picks up sys.argv; 
    # otherwise it treats arg_list as the full argv list.
    return parser.parse_args(arg_list)

if __name__ == "__main__":
    args = parse_args()
    main(args)

