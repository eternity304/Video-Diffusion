import torch
from typing import List, Optional, Tuple, Union
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch
from diffusers import CogVideoXDPMScheduler

from cap_pipeline import CAPVideoPipeline
from pathlib import Path
from typing import Dict, List
import torch
import einops
import cv2
import numpy as np

import os
# import torchaudio
from torchvision.utils import make_grid

def get_optimizer(learning_rate, adam_beta1, adam_beta2, adam_epsilon, adam_weight_decay, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=learning_rate,
            betas=(adam_beta1, adam_beta2),
            eps=adam_epsilon,
            weight_decay=adam_weight_decay,
        )

    optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        params_to_optimize,
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=adam_weight_decay,
    )

    return optimizer

def read_video(video_path, start_index=0, frames_count=49, stride=1):
    video_reader = VideoReader(video_path)
    end_index = min(start_index + frames_count * stride, len(video_reader)) - 1
    batch_index = np.linspace(start_index, end_index, frames_count, dtype=int)
    numpy_video = video_reader.get_batch(batch_index).asnumpy()
    return numpy_video

def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
        embed_dim=attention_head_dim,
        crops_coords=grid_crops_coords,
        grid_size=(grid_height, grid_width),
        temporal_size=num_frames,
    )

    freqs_cos = freqs_cos.to(device=device)
    freqs_sin = freqs_sin.to(device=device)
    return freqs_cos, freqs_sin

def init_pipeline(
    pretrained_model_name_or_path: str,
    transformer,
    vae,
    scheduler,
    dtype,
):
    pipeline = CAPVideoPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=dtype,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = CogVideoXDPMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", **scheduler_args)
    # pipeline = pipeline.to(accelerator.device)

    return pipeline

def init_pipeline(
    pretrained_model_name_or_path: str,
    transformer,
    vae,
    scheduler,
    dtype,
):
    pipeline = CAPVideoPipeline.from_pretrained(
        pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=dtype,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    pipeline.scheduler = CogVideoXDPMScheduler.from_config(pipeline.scheduler.config, timestep_spacing="trailing", **scheduler_args)
    # pipeline = pipeline.to(accelerator.device)

    return pipeline

def run_pipe(
    pipeline,
    height: int,
    width: int,
    num_frames: int,
    conditioning: List[torch.Tensor],
    latents: List[torch.Tensor],
    ref_mask_chunks: List[torch.Tensor],
    audio_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    sequence_infos: List[torch.Tensor],  # timestep ids
    uncond_latents: Optional[torch.FloatTensor] = None,
    uncond_conditioning: Optional[List[torch.FloatTensor]] = None,
    uncond_audio_embeds: Optional[torch.Tensor] = None,
    uncond_text_embeds: Optional[torch.Tensor] = None,
    guidance_scale: float = 1.,
    use_dynamic_cfg: bool = False,
    num_inference_steps: int = 50,
    seed: int = None,
    growth_factor: float = None,
):
    # run inference
    generator = torch.Generator(device=pipeline.device).manual_seed(seed) if seed else None

    videos = pipeline(
        guidance_scale=guidance_scale,
        use_dynamic_cfg=use_dynamic_cfg,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        conditioning=conditioning,
        uncond_conditioning=uncond_conditioning,
        latents=latents,
        uncond_latents=uncond_latents,
        ref_mask_chunks=ref_mask_chunks,
        audio_embeds=audio_embeds,
        uncond_audio_embeds=uncond_audio_embeds,
        text_embeds=text_embeds,
        uncond_text_embeds=uncond_text_embeds,
        sequence_infos=sequence_infos, 
        generator=generator, 
        output_type="pt",
        growth_factor=growth_factor,
    ).frames
    
    pred_videos = []
    for pred_video in videos:
        pred_videos.append(pred_video.float() * 2. - 1.)

    return pred_videos
    

def log_validation(
    file_name: str,
    output_dir: Path,
    sequence_infos: List[Tuple[bool, torch.Tensor]],
    pred_videos: List[torch.Tensor],
    input_videos: List[torch.Tensor],
    cond_vises: List[torch.Tensor],
    write_frames: bool = False,
    audio: torch.Tensor = None,
):
    img = visualize_batch(
        input_videos,
        cond_vises,
        pred_videos,
    )

    cv2.imwrite(str(output_dir / f"{file_name}.jpg"), img)

    vis_gt_videos = []
    vis_pred_videos = []
    vis_ref_videos = []
    for chunk_id, seq_info in enumerate(sequence_infos):
        if seq_info[0]:
            vis_ref_videos.append(input_videos[chunk_id])
        else:
            vis_gt_videos.append(input_videos[chunk_id])
            vis_pred_videos.append(pred_videos[chunk_id])

    write_videos(
        vis_ref_videos,
        vis_gt_videos,
        vis_pred_videos,
        str(output_dir / f"{file_name}.mp4"),
        audio,
    )

    if write_frames:
        visualize_batch_frames(
            str(output_dir / f"{file_name}"),
            input_videos,
            cond_vises,
            pred_videos,
        )

