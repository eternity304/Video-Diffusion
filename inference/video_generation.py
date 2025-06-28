
import argparse
import os

import math
import yaml
import logging
import random
import numpy as np
import sys
import imageio
import torch

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger

from data.VideoDataset import VideoDataset 
from torch.utils.data import DataLoader, DistributedSampler

from diffusers import AutoencoderKLCogVideoX, CogVideoXDDIMScheduler
from model.cap_transformer import CAPVideoXTransformer3DModel

from inference.inference_pipeline import *

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Unconditioned Video Diffusion Inference")

    parser.add_argument("--dataset-path", type=str, required=True, help="Directory containing input reference videos.")
    parser.add_argument("--pretrained-model-name-or-path", type=str, required=True, help="Path or HF ID where transformer/vae/scheduler are stored.")
    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to fine‐tuned checkpoint containing transformer state_dict.")
    parser.add_argument("--output-path", type=str, required=True, help="Where to write generated videos.")
    parser.add_argument("--model-config", type=str, required=True, help="YAML file describing model params (height, width, num_reference, num_target, etc.)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device (usually 1 for inference).")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of reverse diffusion steps to run.")
    parser.add_argument("--mixed-precision", type=str, default="no", help="Whether to run backbone in 'fp16', 'bf16', or 'fp32'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--shuffle", type=int, default=False, help="Whether to shuffle dataset. Usually False for inference.")
    parser.add_argument("--sample-frames", type=int, default=50)
    parser.add_argument("--output-dir", type=str)

    # If arg_list is None, argparse picks up sys.argv; 
    # otherwise it treats arg_list as the full argv list.
    return parser.parse_args(arg_list)

def save_video(video_tensor, path):
    video_tensor = video_tensor.squeeze(0)  # [3, 49, 256, 256]
    video_tensor = video_tensor.permute(1, 2, 3, 0)  # [49, 256, 256, 3]

    # Normalize to 0–255 and convert to uint8 if needed
    video_np = (video_tensor * 255).clamp(0, 255).byte().cpu().numpy()

    # Save as .mp4
    imageio.mimsave(path, video_np, fps=16)

def main(args):
    with open(args.model_config, "r") as f: model_config = yaml.safe_load(f)
    if args.mixed_precision.lower() == "fp16":
        dtype = torch.float16
    elif args.mixed_precision.lower() == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir,
                                                        logging_dir=os.path.join(args.output_dir, "logs"))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(mixed_precision=args.mixed_precision,
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

    dataset = VideoDataset(
        videos_dir=args.dataset_path,
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
        collate_fn=lambda x: x[0],   # since dataset returns already‐batched items
        num_workers=2,
        pin_memory=True,
    )
    logger.info(f"Number of test examples: {len(data_loader)}")

    device = "cuda"
    dtype = torch.float32

    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        subfolder="transformer",
        torch_dtype=torch.float32,
        cond_in_channels=1,  # only one channel (the ref_mask)
        sample_width=model_config["width"] // 8,
        sample_height=model_config["height"] // 8,
        sample_frames=args.sample_frames,
        max_text_seq_length=1,
        max_n_references=model_config["max_n_references"],
        apply_attention_scaling=model_config["use_growth_scaling"],
        use_rotary_positional_embeddings=False,
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    vae.eval().to(dtype)
    transformer.eval().to(dtype)

    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    print("------------------------")
    print("Loading Model Checkpoint")
    print("------------------------")

    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    transformer.load_state_dict(ckpt["state_dict"], strict=False)

    vae, transformer, scheduler, data_loader = accelerator.prepare(vae, transformer, scheduler, data_loader)

    print("--------------------------")
    print("Loading Inference Pipeline")
    print("--------------------------")

    pipe = VideoDiffusionPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler
    )
    batch = next(iter(data_loader))
    videos = pipe(batch, num_inference_steps=50, return_decode_latent=True, return_latent=False, sample_frames=args.sample_frames)
    
    save_video(videos[0] , args.output_path)
    print("Saved")

if __name__ == "__main__":
    args = parse_args()
    main(args)
