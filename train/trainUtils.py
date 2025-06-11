import torch
from typing import List, Optional, Tuple, Union
from diffusers.pipelines.cogvideo.pipeline_cogvideox import get_resize_crop_region_for_grid
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import torch

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

