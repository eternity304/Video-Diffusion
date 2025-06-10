from typing import Dict, List, Optional, Union, Tuple


import imageio
import numpy as np
from tqdm import tqdm
import inspect

import torch

from diffusers import DiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps
from diffusers.video_processor import VideoProcessor

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
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
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

        for i, video in enumerate(batch["video_chunks"]):
            # video: [B, C, F, H, W]
            video = video.to(device=device, dtype=dtype)
            # Initialize first frame and set rest as random noise
            video[:, :, 1:, :, :] = torch.randn(video[:, :, 1:, :, :].shape, generator=generator, device=device)
            with torch.no_grad(): dist = self.vae.encode(video).latent_dist.sample()
            latent = dist * self.vae.config.scaling_factor
            latent = latent.permute(0, 2, 1, 3, 4).contiguous()  # [B, F, C_z, h, w]
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

        # 2) Build 2× for classifier-free guidance
        latents = latent_chunks
        masks   = [torch.cat([m, torch.zeros_like(m)], dim=0) for m in ref_mask_chunks]
        # keep ref_latents for mixing
        ref_latents = [torch.cat([z, torch.zeros_like(z)], dim=0) for z in latent_chunks]

        # 3) dummy audio/text embeddings (adjust if you have real ones)
        B2 = latent_chunks[0].shape[0] * 2
        total_F = sum(z.shape[1] for z in latents)
        audio_embeds = torch.zeros((B2, total_F, 768), dtype=dtype, device=device)
        text_embeds  = torch.zeros((B2, 1,
            self.transformer.config.attention_head_dim * self.transformer.config.num_attention_heads
        ), dtype=dtype, device=device)

        # 4) timesteps
        timesteps, _ = retrieve_timesteps(self.scheduler, num_inference_steps, device=device)

        # 5) optional fuse QKV once
        # try:
        #     self.transformer.fuse_qkv_projections()
        # except Exception:
        #     pass

        # 6) denoising loop
        old_pred_original_samples = [None] * len(latents)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        for i, t in enumerate(tqdm(timesteps, desc="Inference Progress")):
            latent_model_inputs = [self.scheduler.scale_model_input(latent, t) for latent in latents]
            latent_model_inputs = [torch.cat([chunks] * 2, dim=0)for chunks in latent_model_inputs]
            B2, F, C, H, W = latent_model_inputs[0].shape
            # one zero condition tensor
            zero_cond = torch.zeros((B2, F, 1, H, W), dtype=dtype, device=device)
            # single forward
            noise_preds = self.transformer(
                hidden_states=latent_model_inputs,
                encoder_hidden_states=text_embeds,
                audio_embeds=audio_embeds,
                condition=[zero_cond] * len(latent_model_inputs),
                sequence_infos=[[False, torch.arange(chunk.shape[1])]for chunk in latents],
                timestep=t.expand(B2),
                image_rotary_emb=None,
                return_dict=False,
            )[0]

            # apply guidance, scheduler.step, then mixing
            new_latents = []
            new_old_preds = []

            for idx, noise_pred in enumerate(noise_preds):
                latent = latents[idx]
                old_pred = old_pred_original_samples[idx]
                noise_pred, noise_pred_uncond = noise_pred.chunk(2, dim=0)
                # noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

                prev_sample, pred_original_sample = self.scheduler.step(
                    model_output=noise_pred,
                    timestep=t,
                    sample=latent,
                    **extra_step_kwargs,
                    return_dict=False,
                )
                mixed = ref_mask_chunks[idx] * ref_latent_chunks[idx] + (1 - ref_mask_chunks[idx]) * prev_sample
                new_latents.append(mixed)
                new_old_preds.append(pred_original_sample)

            # update for next iteration
            latents = list(new_latents)
            old_pred_original_samples = new_old_preds

        # 7) decode to videos
        videos = []
        if return_latent:
            return latents
        
        if return_decode_latent:
            return [self.decode_latents(lat) for lat in latents]
        
        for latent in latents:
            dec = latent.permute(0, 2, 1, 3, 4) / self.vae.config.scaling_factor
            frames = self.vae.decode(dec).sample
            video = self.video_processor.postprocess_video(video=frames, output_type=output_type)
            videos.append(video)

        return {"frames": videos} if return_dict else videos