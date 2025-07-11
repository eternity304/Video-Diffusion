
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

from inference._inference_pipeline import *
from data.VideoDataset import *
from model.flameObj import *


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

def encode_video(vae, video):
    video = video.to(vae.device, dtype=vae.dtype)
    video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
    with torch.no_grad(): latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)

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
        batch_size=1,
        # sampler=sampler,
        collate_fn=lambda x: x,   
        num_workers=2,
        pin_memory=True,
    )
    logger.info(f"Number of test examples: {len(data_loader)}")
    device = "cuda"
    weight_dtype = torch.float32

    flamePath = "/scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz"
    sourcePath = "/scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit"
    dataPath = [os.path.join(os.path.join(sourcePath, data), "fit.npz") for data in os.listdir(sourcePath)]
    seqPath = "/scratch/ondemand28/harryscz/head/_-91nXXjrVo_00/fit.npz"

    head = Flame(flamePath, device="cuda")

    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        subfolder="transformer",
        torch_dtype=torch.float32,
        cond_in_channels=16,  # only one channel (the ref_mask)
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
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )

    vae.eval().to(weight_dtype)
    transformer.eval().to(weight_dtype)

    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, map_location="cpu")
    transformer.load_state_dict(ckpt["state_dict"], strict=False)

    vae, transformer, scheduler, data_loader = accelerator.prepare(vae, transformer, scheduler, data_loader)

    pipe = CAPVideoPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler
    )
    
    for i in range(0,10):
        batch = dataPath[i:i+1]
        print(batch)

        latent_chunks = []
        uncond_latent_chunks = []
        cond_chunks = []
        uncond_chunks = []
        cond_vises = []
        ref_mask_chunks = []

        uvs = head.batch_uv(batch, resolution=256, sample_frames=args.sample_frames).permute(0,1,4,2,3) # load UVs of shape B, F, C, H, W
        ref_frame = uvs[:, 0, :, :, :].unsqueeze(1) # B, 1, C, H, W

        latent_chunk = encode_video(vae, uvs).to(dtype=dtype)
        _ref_latent = encode_video(vae, ref_frame).to(dtype=dtype)
        ref_latent = torch.zeros(latent_chunk.shape).to(dtype=dtype, device=accelerator.device)
        ref_latent[:, 0, ...] = _ref_latent.squeeze(1)
        ref_mask = torch.zeros(latent_chunk.shape).to(accelerator.device)
        ref_mask[:, 0, ...] = 1.0

        # ref_mask_chunks.append(cond_chunk["ref_mask"].permute(0, 1, 4, 2, 3))
        # cond_vis = conditioning.get_vis(cond_chunk["condition"])
        # cond_chunk = einops.rearrange(cond_chunk["condition"], 'b f h w c -> b f c h w').to(device=device)
        # cond_chunk = cond_chunk.to(dtype=dtype)
        latent_chunks.append(latent_chunk)
        uncond_latent_chunks.append(latent_chunk * 0.)
        cond_chunks.append(ref_latent)
        # cond_vises.append(cond_vis)
        uncond_chunks.append(ref_latent * 0.)
        ref_mask_chunks.append(ref_mask)
        B, F_z, C, H_z, W_z = latent_chunks[0].shape
            
        # with torch.no_grad():
        #     audio_encoding = encode_audio(
        #         audio_model=audio_model,
        #         audio=batch["audio"].to(dtype=dtype, device=device),
        #         sample_positions=batch["sample_positions"].to(dtype=dtype, device=device),
        #     )
        # if not batch["has_audio"][0]:  # if there is no audio, set audio encoding to zero
        #     audio_encoding = audio_encoding * 0.
        audio_encoding = torch.zeros(
            (B, 3, 768), dtype=torch.float32, device=accelerator.device
        )
        uncond_audio_encoding = audio_encoding * 0.
        text_embeds = torch.zeros(1, 1, 1920, device=device, dtype=dtype)

        sequence_infos = []
        for chunk_id, latent in enumerate(latent_chunks):
            sequence_infos.append((False, torch.arange(0, latent.shape[1], device=accelerator.device)))

        out = pipe(
            height=256,
            width=256,
            num_frames=29,
            num_inference_steps=50,
            conditioning=cond_chunks,
            uncond_conditioning=uncond_chunks,
            latents=latent_chunks,
            uncond_latents=uncond_latent_chunks,
            ref_mask_chunks=ref_mask_chunks,
            audio_embeds=audio_encoding,
            uncond_audio_embeds=uncond_audio_encoding,
            text_embeds=text_embeds,
            uncond_text_embeds=text_embeds,
            sequence_infos=sequence_infos, 
            output_type="pt",
        )

        orig = uvs[0].permute(0,2,3,1)
        recon = out[0][0][0].permute(0,2,3,1)

        sampledUV = head.sampleFromUV(orig, savePath=f"diffOut/{i}_uv.mp4") # input has shape F, H, W, C
        head.sampleTo3D(sampledUV, savePath=f"diffOut/{i}_3d.mp4", resolution=512, dist=1.2)

        sampledUV = head.sampleFromUV(recon, savePath=f"diffOut/{i}_diff_uv.mp4") # input has shape F, H, W, C
        head.sampleTo3D(sampledUV, savePath=f"diffOut/{i}_diff_3d.mp4", resolution=512, dist=1.2)
    
    # latent_chunks = []
    # uncond_latent_chunks = []
    # cond_chunks = []
    # uncond_chunks = []
    # cond_vises = []
    # ref_mask_chunks = []
    # for chunk_id in range(len(batch["video_chunks"])):
        
    #     video = batch["video_chunks"][chunk_id].permute(0,2,1,3,4)
    #     if video.shape[1] > args.sample_frames: video = video[:,:args.sample_frames,...]   
    #     latent_chunk = encode_video(vae, video).to(dtype=weight_dtype)
 
    #     # cond_chunk = {key: batch["cond_chunks"][key][chunk_id] for key in batch["cond_chunks"]}
    #     # for key in cond_chunk:
    #     #     cond_chunk[key] = cond_chunk[key].to(dtype=torch.float32, device=device)
    #     ref_frame = video[:, 0, :, :, :].unsqueeze(1) # B, 1, C, H, W
    #     _ref_latent = encode_video(vae, ref_frame).to(dtype=weight_dtype)
    #     ref_latent = torch.zeros(latent_chunk.shape).to(dtype=weight_dtype, device=accelerator.device)
    #     ref_latent[:, 0, ...] = _ref_latent.squeeze(1)
    #     ref_mask = torch.zeros(latent_chunk.shape).to(accelerator.device)
    #     ref_mask[:, 0, ...] = 1.0

    #     # ref_mask_chunks.append(cond_chunk["ref_mask"].permute(0, 1, 4, 2, 3))
    #     # cond_vis = conditioning.get_vis(cond_chunk["condition"])
    #     # cond_chunk = einops.rearrange(cond_chunk["condition"], 'b f h w c -> b f c h w').to(device=device)
    #     # cond_chunk = cond_chunk.to(dtype=dtype)
    #     latent_chunks.append(latent_chunk)
    #     uncond_latent_chunks.append(latent_chunk * 0.)
    #     cond_chunks.append(ref_latent)
    #     # cond_vises.append(cond_vis)
    #     uncond_chunks.append(ref_latent * 0.)
    #     ref_mask_chunks.append(ref_mask)
    #     B, F_z, C, H_z, W_z = latent_chunks[0].shape
        
    # # with torch.no_grad():
    # #     audio_encoding = encode_audio(
    # #         audio_model=audio_model,
    # #         audio=batch["audio"].to(dtype=dtype, device=device),
    # #         sample_positions=batch["sample_positions"].to(dtype=dtype, device=device),
    # #     )
    # # if not batch["has_audio"][0]:  # if there is no audio, set audio encoding to zero
    # #     audio_encoding = audio_encoding * 0.
    # audio_encoding = torch.zeros(
    #     (B, 3, 768), dtype=weight_dtype, device=accelerator.device
    # )
    # uncond_audio_encoding = audio_encoding * 0.
    # text_embeds = torch.zeros(1, 1, 1920, device=device, dtype=weight_dtype)

    # sequence_infos = []
    # for chunk_id, latent in enumerate(latent_chunks):
    #     sequence_infos.append((False, torch.arange(0, latent.shape[1], device=accelerator.device)))

    # out = pipe(
    #     height=512,
    #     width=512,
    #     num_frames=29,
    #     num_inference_steps=50,
    #     conditioning=cond_chunks,
    #     uncond_conditioning=uncond_chunks,
    #     latents=latent_chunks,
    #     uncond_latents=uncond_latent_chunks,
    #     ref_mask_chunks=ref_mask_chunks,
    #     audio_embeds=audio_encoding,
    #     uncond_audio_embeds=uncond_audio_encoding,
    #     text_embeds=text_embeds,
    #     uncond_text_embeds=text_embeds,
    #     sequence_infos=sequence_infos, 
    #     output_type="pt",
    # )

    

if __name__ == "__main__":
    args = parse_args()
    main(args)
