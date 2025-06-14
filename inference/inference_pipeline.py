from typing import Dict, List, Optional, Union, Tuple


import imageio
import numpy as np
from tqdm import tqdm
import inspect

import torch

from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.video_processor import VideoProcessor

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

def save_video(video : torch.tensor, save_path : str, fps : int = 16):
    video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # [49, 256, 256, 3]

    video_np = (video_np * 255).clip(0, 255).astype(np.uint8)

    imageio.mimsave(save_path, video_np, fps=fps)
    
    print("Saved !")

class VideoDiffusionPipeline(DiffusionPipeline):
    """
    A custom diffusion pipeline that mirrors your manual inference loop,
    but inherits from DiffusionPipeline to leverage no-grad, mixed-precision,
    and buffer reuse for maximum efficiency.
    """
    def __init__(
        self,
        vae,
        transformer,
        scheduler,
    ):
        super().__init__()
        self.register_modules(vae=vae, transformer=transformer, scheduler=scheduler)

        # Scale factors for spatial/temporal axes
        self.vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_scale_factor_temporal = getattr(self.vae.config, "temporal_compression_ratio", 1)

        # Video post-processor
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4).to(self.vae.dtype)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames
    
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    @torch.no_grad()
    def __call__(
        self,
        batch: Dict[str, Union[List[torch.FloatTensor], Dict[str, List[torch.FloatTensor]]]],
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int = 42,
        eta : float = 0.0,
        output_type: str = "pil",
        return_dict: bool = False,
        return_latent: bool = False,
        return_pil: bool = False,
        return_decode_latent : bool = True,
    ) -> Union[List, Dict]:
        device = self._execution_device
        dtype = self.transformer.dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(seed+1)

        # 1) Extract & encode
        latent_chunks: List[torch.Tensor] = []
        ref_mask_chunks: List[torch.Tensor] = []
        sequence_infos: List[tuple] = []
        ref_latent_chunks: List[torch.Tensor] = []

        # 4) timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device=device)

        for i, video in enumerate(batch["video_chunks"]):
            # video: [B, C, F, H, W]
            video = video.to(device=device, dtype=dtype)
            B = video.shape[0]
            dist = self.vae.encode(video).latent_dist.sample()
            latent = dist * self.vae.config.scaling_factor 
            latent = latent.permute(0,2,1,3,4).contiguous().to(dtype) # [B, F, C_z, h, w]
            latent_chunks.append(latent)

            # mask: batch["cond_chunks"]["ref_mask"][i] shape [B, F, H, W, C_mask]
            B, F, _, H, W = latent.shape
            rm = torch.zeros((B, F, 1, H, W), device=device, dtype=dtype)
            rm[:, 0, 0, :, :] = 1.0    # keep frame 0
            ref_mask_chunks.append(rm)
            ref_latent_chunks.append(latent)  # same shape [B, F, C_z, h, w]

            # sequence info
            is_ref = batch.get("chunk_is_ref", [False] * len(latent_chunks))[i]
            seq = torch.arange(0, latent.shape[1], device=device)
            sequence_infos.append((is_ref, seq))

            init_latents = []
            for idx, x0 in enumerate(latent_chunks):
                # sample a fresh ε noise
                noise = torch.randn_like(x0, dtype=dtype, device=device)
                zT = self.scheduler.add_noise(x0, noise, timesteps[0])
                init_latents.append(zT)
            latents = init_latents

                    # 3) dummy audio/text embeddings (adjust if you have real ones)
            B2 = latent_chunks[0].shape[0] 
            total_F = sum(z.shape[1] for z in latents)
            audio_embeds = torch.zeros((B2, total_F, 768), dtype=dtype, device=device)
            text_embeds  = torch.zeros((B2, 1,
                self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads
            ), dtype=dtype, device=device)

            # 6) denoising loop
            old_pred_original_samples = [None] * len(latents)
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            for i, t in enumerate(tqdm(timesteps, desc="Inference Progress")):
                # 1) prep the model input
                latent_in = [self.scheduler.scale_model_input(x, t).to(dtype) for x in latents]
                B = latent_in[0].shape[0]
                
                zero_cond = torch.zeros((latent_in[0].shape[0], F, 1, H, W), dtype=dtype, device=device)

                new_latents = []
                # 2) predict noise
                model_out = self.transformer(
                    hidden_states=latent_in,
                    encoder_hidden_states=text_embeds,
                    audio_embeds=audio_embeds,
                    condition=[zero_cond] * len(latent_in),
                    sequence_infos=[[False, torch.arange(chunk.shape[1])]for chunk in latents],
                    timestep=t.expand(B),
                    image_rotary_emb=None,
                    return_dict=False,
                )[0]

                for idx, noise_pred in enumerate(model_out):
                    _z, pred_original_sample = self.scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=latents[idx],
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                    new_latents.append(_z)
                latents = list(new_latents)

            # 7) decode to videos
            videos = []
            if return_latent:
                return latents
            
            if return_decode_latent:
                return [self.decode_latents(lat) for lat in latents]
            
            for latent in latents:
                dec = latent.permute(0, 2, 1, 3, 4) / self.vae.config.scaling_factor
                frames = self.vae.decode(dec).sample
                # frames = frames.permute(0, 2, 1, 3, 4)
                print(frames.shape)
                video = self.video_processor.postprocess_video(video=frames, output_type=output_type)
                videos.append(video)

            return {"frames": videos} if return_dict else videos
        

