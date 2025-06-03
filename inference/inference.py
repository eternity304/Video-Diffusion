
import os
import argparse
import math
import torch
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms
from einops import rearrange
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderKLCogVideoX
from diffusers import CogVideoXDPMScheduler
from typing import Callable, Dict, List, Optional, Tuple, Union

from ..CustomDataset import VideoDataset                   # Your custom Dataset
from ..cap_transformer import CAPVideoXTransformer3DModel   # Your 3D transformer class
from ..trainUtils import *                                  # E.g. get_optimizer(), etc.

#
# ——— Helper functions from the “reference pipeline” ———
#

def resize_for_crop(image: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
    """
    Resize `image` so that its smaller dimension matches the target,
    but maintain aspect ratio, then we will center-crop to exactly (crop_h, crop_w).
    `image` is a Tensor shape [C,H,W], assumed in [-1,1] already.
    """
    img_h, img_w = image.shape[-2:]
    if img_h >= crop_h and img_w >= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    elif img_h <= crop_h and img_w <= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w 
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized

def prepare_frames(input_images: List[Image.Image], video_size: Tuple[int,int], do_resize=True, do_crop=True):
    """
    Given a list of PIL Images (RGB), stack them, normalize to [-1,1], 
    optionally resize-then-center-crop to `video_size`, 
    and return a Tensor shape [1, T, 3, H, W].
    """
    # 1) stack numpy arrays
    arrs = np.stack([np.array(x) for x in input_images], axis=0)  # [T, H, W, 3]
    tensor = torch.from_numpy(arrs).permute(0, 3, 1, 2) / 127.5 - 1.0  # [T,3,H,W], in [-1..1]

    if do_resize:
        resized_list = []
        for frame in tensor:
            resized_list.append(resize_for_crop(frame, crop_h=video_size[0], crop_w=video_size[1]))
        tensor = torch.stack(resized_list, dim=0)

    if do_crop:
        cropped_list = []
        for frame in tensor:
            cropped_list.append(transforms.functional.center_crop(frame, video_size))
        tensor = torch.stack(cropped_list, dim=0)

    return tensor.unsqueeze(0)  # [1, T, 3, H, W]

def get_resize_crop_region_for_grid(
    src: Tuple[int,int], tgt_width: int, tgt_height: int
) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Used for rotary embeddings. Identical to reference.
    """
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """
    Same “retrieve_timesteps” logic as in reference.
    """
    import inspect
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The scheduler {scheduler.__class__} does not support custom timestep schedules."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The scheduler {scheduler.__class__} does not support custom sigma schedules."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

#
# ——— The exact same CAPVideoPipeline class definition from training. ———
#
class CAPVideoPipeline(DiffusionPipeline):
    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        vae: AutoencoderKLCogVideoX,
        transformer: "CAPVideoXTransformer3DModel",
        scheduler: CogVideoXDPMScheduler,
    ):
        super().__init__()
        self.register_modules(vae=vae, transformer=transformer, scheduler=scheduler)

        # Make sure these two factors match how training computed them:
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio
            if hasattr(self, "vae") and self.vae is not None
            else 4
        )

        # VideoProcessor handles post‐processing from decoded latents → actual video frames (e.g. convert [-1,1] to uint8)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Union[torch.Generator, List[torch.Generator]],
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You passed a list of {len(generator)} generators but batch_size={batch_size}."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # latents: [B, F_lat, C_z, h_lat, w_lat] → permute to [B, C_z, F_lat, h_lat, w_lat]
        latents = latents.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latents = latents / self.vae.config.scaling_factor

        frames = self.vae.decode(latents).sample  # [B, 3, F, H, W]
        return frames

    def prepare_extra_step_kwargs(self, generator: Union[torch.Generator, List[torch.Generator]], eta: float):
        import inspect
        extra: Dict[str, object] = {}
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_eta:
            extra["eta"] = eta
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra["generator"] = generator
        return extra

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Exactly the same logic used in training for building 3D rotary embeddings.
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
        return freqs_cos.to(device), freqs_sin.to(device)

    @property
    def guidance_scale(self) -> float:
        return self._guidance_scale

    @property
    def num_timesteps(self) -> int:
        return self._num_timesteps

    @property
    def attention_kwargs(self) -> Dict:
        return self._attention_kwargs

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        num_frames: int,
        num_inference_steps: int,
        latents: List[torch.Tensor],
        uncond_latents: List[torch.Tensor],
        conditioning: List[torch.Tensor],
        uncond_conditioning: List[torch.Tensor],
        ref_mask_chunks: List[torch.Tensor],
        audio_embeds: torch.Tensor,
        uncond_audio_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        uncond_text_embeds: torch.Tensor,
        sequence_infos: List[Tuple[bool, torch.Tensor]],
        guidance_scale: float = 1.0,
        use_dynamic_cfg: bool = False,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 1,
        growth_factor: Optional[float] = None,
    ):
        """
        Matching the “reference” __call__ exactly. 
        We assume `latents` is a LIST of T‐chunks, each of shape [B, F_lat, C_z, h_lat, w_lat].
        - `conditioning` is a LIST of the per‐chunk “cond_mask” (permute to [B, F_lat, 1, h_lat, w_lat]).
        - `ref_mask_chunks` is the same shape as `conditioning` but before permutation (so [B,1,F_lat,h_lat,w_lat]).
        - `sequence_infos`: list of (is_from_previous_chunk: bool, index_tensor).
        """
        if num_frames > 49:
            raise ValueError("num_frames must be ≤ 49 (rotary embeddings limit).")

        # 1) Normalize height/width if not provided
        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2) Determine batch_size
        if sequence_infos is None:
            raise ValueError("sequence_infos must be passed.")
        # For classifier-free guidance, batch_size is half of conditioning[0].shape[0] if guidance>1.
        if guidance_scale > 1.0:
            batch_size = conditioning[-1].shape[0] // 2
        else:
            batch_size = conditioning[-1].shape[0]

        device = self._execution_device

        # Move audio/text to device
        audio_embeds = audio_embeds.to(device)
        text_embeds = text_embeds.to(device)

        do_classifier_free_guidance = guidance_scale > 1.0

        # 3) Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, None
        )
        self._num_timesteps = len(timesteps)

        # 4) Prepare latents
        initial_latents: List[torch.Tensor] = []
        ref_latents: List[torch.Tensor] = []

        for idx, latent_chunk in enumerate(latents):
            if sequence_infos[idx][0]:  # was a chunk coming from previous generation? Then we keep it.
                noisy_lat = latent_chunk
            else:
                noisy_lat = randn_tensor(
                    latent_chunk.shape,
                    generator,
                    device,
                    dtype=conditioning[-1].dtype,
                )
            initial_latents.append(noisy_lat)

            #── ref_latents: just the original latent (not noised) used to re‐inject at each step:
            ref_lat = latent_chunk
            if do_classifier_free_guidance:
                # we will concat unconditional chunk along batch‐dimension
                ref_lat = torch.cat([ref_lat, uncond_latents[idx]], dim=0)
            ref_latents.append(ref_lat)

        latents = initial_latents

        # 5) Prepare extra step kwargs (e.g. “eta” for DDIM)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6) Prepare rotary embeddings if needed
        if self.transformer.config.use_rotary_positional_embeddings:
            image_rotary_emb = self._prepare_rotary_positional_embeddings(
                height, width, latents[0].shape[1], device
            )
        else:
            image_rotary_emb = None

        # 7) Denoising loop
        old_pred_original_samples = [None] * len(latents)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    break

                # Build latent_model_inputs for each chunk:
                latent_model_inputs = []
                for chunk_idx, latent_chunk in enumerate(latents):
                    if do_classifier_free_guidance:
                        latent_in = torch.cat([latent_chunk, latent_chunk], dim=0)
                    else:
                        latent_in = latent_chunk

                    # scale model input if needed by scheduler (not here, because DPM‐solver++)
                    # latent_in = self.scheduler.scale_model_input(latent_in, t)

                    # Apply reference‐mask: 
                    #   latent_in * (1 - mask) + ref_lat * mask
                    latent_in = latent_in * (1.0 - ref_mask_chunks[chunk_idx]) + ref_latents[chunk_idx] * ref_mask_chunks[chunk_idx]

                    latent_in = latent_in.to(dtype=self.transformer.dtype)
                    latent_model_inputs.append(latent_in)

                # Build conditioning (text/audio) for guidance:
                if do_classifier_free_guidance:
                    conditioning_input = [torch.cat([c, uc], dim=0) for c, uc in zip(conditioning, uncond_conditioning)]
                    audio_input = torch.cat([audio_embeds, uncond_audio_embeds], dim=0)
                    text_input = torch.cat([text_embeds, uncond_text_embeds], dim=0)
                else:
                    conditioning_input = conditioning
                    audio_input = audio_embeds
                    text_input = text_embeds

                # Expand timestep for all batch‐elements in this chunk:
                timestep = t.expand(latent_model_inputs[0].shape[0])

                # Call transformer → predict noise
                noise_preds = self.transformer(
                    hidden_states=latent_model_inputs,
                    encoder_hidden_states=text_input,
                    audio_embeds=audio_input,
                    condition=conditioning_input,
                    sequence_infos=sequence_infos,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]  # [sum_over_chunks_of(batch_size)*?, C_z, F_lat, h_lat, w_lat]

                # “Step” through scheduler for each chunk:
                new_latents: List[torch.Tensor] = []
                new_old_preds: List[torch.Tensor] = []

                for c_idx, chunk_noise_pred in enumerate(noise_preds):
                    pred = chunk_noise_pred.float()

                    if do_classifier_free_guidance:
                        # noise_pred: [2*B, ...] → uncond vs cond
                        noise_pred, noise_pred_uncond = pred.chunk(2, dim=0)
                        pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)
                        ref_chunk = ref_latents[c_idx].chunk(2, dim=0)[0]
                    else:
                        ref_chunk = ref_latents[c_idx]

                    latent_prev = latents[c_idx]
                    old_pred = old_pred_original_samples[c_idx]

                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        # (in our training we always used DPM‐Solver++ so this branch is never reached)
                        raise NotImplementedError("Only CogVideoXDPMScheduler is supported in this inference script.")
                    else:
                        # For CogVideoXDPMScheduler.step:
                        #    Returns (next_latent, pred_original_sample)
                        next_latent, next_old_pred = self.scheduler.step(
                            pred,
                            old_pred,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latent_prev,
                            **extra_step_kwargs,
                            return_dict=False,
                        )

                    # Re‐inject reference‐pixels at each step:
                    next_latent = next_latent * (1.0 - ref_mask_chunks[c_idx]) + ref_chunk * ref_mask_chunks[c_idx]
                    next_latent = next_latent.to(audio_embeds.dtype)

                    new_latents.append(next_latent)
                    new_old_preds.append(next_old_pred)

                latents = new_latents
                old_pred_original_samples = new_old_preds

                progress_bar.update()

        # 8) Decode to actual frames if requested:
        if output_type != "latent":
            videos: List[torch.Tensor] = []
            for chunk_latent in latents:
                video = self.decode_latents(chunk_latent)  # [B, 3, F, H, W]
                # postprocess_video → moves frames from [-1..1] floats to PIL (uint8)
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                videos.append(video)

        else:
            videos = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos,)
        return videos  # in our simple script, just return List of [B,3,F,H,W] PIL lists

#
# ——— End of CAPVideoPipeline definition ———
#

#
# ——— Now: the “main” inference routine ———
#
def main():
    parser = argparse.ArgumentParser(description="CogVideoX Inference Script")
    parser.add_argument(
        "--model-config", type=str, required=True,
        help="Path to the same YAML used for training (height, width, num_reference, num_target, etc.)."
    )
    parser.add_argument(
        "--pretrained-model", type=str, required=True,
        help="Directory (or HF ID) of the original pretrained CogVideoX weights."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to your fine-tuned transformer checkpoint (e.g. checkpoint-500.pt)."
    )
    parser.add_argument(
        "--reference-frames", type=str, required=True,
        help="Directory containing one or more reference frames (e.g. PNG/JPG)."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Where to save the generated video (mp4)."
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="cuda or cpu"
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=50,
        help="How many diffusion steps to run (matching training's default)."
    )
    parser.add_argument(
        "--height", type=int, default=None,
        help="Output frame height. If not set, taken from model-config."
    )
    parser.add_argument(
        "--width", type=int, default=None,
        help="Output frame width. If not set, taken from model-config."
    )
    parser.add_argument(
        "--num-frames", type=int, default=None,
        help="Number of frames to generate. If not set, taken from model-config['num_target']."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    args = parser.parse_args()

    # 1) Load model_config YAML
    with open(args.model_config, "r") as f:
        model_cfg = yaml.safe_load(f)
    # If height/width/num_frames not overridden, grab from config:
    height = args.height or model_cfg["height"]
    width = args.width or model_cfg["width"]
    num_frames = args.num_frames or model_cfg["num_target"]

    # 2) Device & seed
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 3) Load pretrained VAE + scheduler
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model,
        subfolder="vae",
        torch_dtype=torch.float32
    )
    scheduler = CogVideoXDPMScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
    )

    # 4) Instantiate transformer with orig config, then load fine-tuned weights
    from cap_video.cap_transformer import CAPVideoXTransformer3DModel
    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
        cond_in_channels=1,
        sample_width=model_cfg["width"] // 8,
        sample_height=model_cfg["height"] // 8,
        max_text_seq_length=1,
        max_n_references=model_cfg["num_reference"],
        apply_attention_scaling=model_cfg["use_growth_scaling"],
        use_rotary_positional_embeddings=False,
    )
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    if "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        raw_state_dict = ckpt["model_state_dict"]
    else:
        # If the .pt is literally just a pure state_dict, do this:
        raw_state_dict = ckpt

    clean_state_dict = {}
    for key, val in raw_state_dict.items():
        new_key = key
        # e.g. if your keys start with "module.", remove it:
        if key.startswith("module."):
            new_key = key[len("module."):]
        # or if saved under "model.", do:
        # if key.startswith("model."):
        #     new_key = key[len("model."):]
        clean_state_dict[new_key] = val
    missing, unexpected = transformer.load_state_dict(clean_state_dict, strict=False)

    print("==> Missing keys (these will be randomly initialized because they weren't in the checkpoint):")
    for k in missing:
        print("   ", k)
    print("==> Unexpected keys (these were in the checkpoint but didn't match any parameter in your model):")
    for k in unexpected:
        print("   ", k)

    # 5) Move everything to device (float32)
    vae = vae.to(device).float()
    transformer = transformer.to(device).float()

    # 6) Wrap in CAPVideoPipeline
    pipe = CAPVideoPipeline(vae=vae, transformer=transformer, scheduler=scheduler)
    pipe = pipe.to(device)

    # 7) Read reference frames from disk (alphabetical order)
    ref_paths = sorted([
        os.path.join(args.reference_frames, fn)
        for fn in os.listdir(args.reference_frames)
        if fn.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    if len(ref_paths) == 0:
        raise ValueError("No .png/.jpg files found in --reference-frames")
    pil_frames: List[Image.Image] = [Image.open(p).convert("RGB") for p in ref_paths]

    # 8) Preprocess them → [1, T_ref, 3, H, W]
    ref_tensor = prepare_frames(pil_frames, video_size=(height, width), do_resize=True, do_crop=True)
    # Move to device & dtype float32:
    ref_tensor = ref_tensor.to(device=device, dtype=torch.float32)

    # 9) Encode through VAE (just like training’s encode_video)
    with torch.no_grad():
        # VAE expects [B, 3, F, H, W], so we must permute:
        #  ref_tensor: [1, T_ref, 3, H, W] → [1, 3, T_ref, H, W]
        ref_permuted = ref_tensor.permute(0, 2, 1, 3, 4)
        # Produce a Distribution object; take .latent_dist.sample()*scaling_factor
        latent_dist = vae.encode(ref_permuted).latent_dist
        ref_latents_full = latent_dist.sample() * vae.config.scaling_factor
        # shape: [1, C_z, F_lat, h_lat, w_lat]
    # We want to break ref_latents_full into “chunks” like training did.
    # In training, they assumed `num_reference=1` → so every “chunk” is one reference‐latent.
    # If you have multiple reference frames, you’d normally chunk by time. Here let's assume
    # we treat the entire latent volume as one single chunk. So:
    ref_latents_chunks = [ref_latents_full]  # List of length 1

    # 10) Build “ref_mask_chunks”
    # In training, they had “raw_ref_mask” = mask at frame‐level being 1 where reference pixels exist.
    # For inference, we often use an “all‐1” mask over the entire latent volume, so that reference latents
    # are injected fully at every step. This is typical for reference‐based generation.
    # So: create a tensor of ones of shape [B,1,F_lat,h_lat,w_lat].
    B, C_z, F_lat, h_lat, w_lat = ref_latents_full.shape
    ref_mask = torch.ones((B, 1, F_lat, h_lat, w_lat), dtype=torch.float32, device=device)
    # Permute to “cond_mask” shape [B, F_lat, 1, h_lat, w_lat]:
    cond_mask = ref_mask.permute(0, 2, 1, 3, 4).contiguous()

    ref_mask_chunks = [ref_mask]       # list of length 1
    cond_chunks = [cond_mask]          # list of length 1

    # 11) For classifier-free guidance—build “uncond” versions:
    # Here we’ll run diffusion with guidance_scale=1.0, so no guidance; but we still must pass
    # “uncond_latents” and “uncond_conditioning” as empty lists of exactly the same shape.
    uncond_latents_chunks = [torch.zeros_like(ref_latents_full)]
    uncond_cond_chunks = [torch.zeros_like(cond_mask)]

    # 12) Build sequence_infos: exactly the same as training at eval time.
    # They used sequence_infos = [(False, torch.arange(0, F_lat)), …] for each chunk.
    seq_idx = torch.arange(0, F_lat, device=device)
    sequence_infos = [(False, seq_idx)]

    # 13) Dummy audio/text embeddings (zeros) of the correct shape:
    # Training used: inner_dim = num_attention_heads * attention_head_dim
    inner_dim = transformer.config.num_attention_heads * transformer.config.attention_head_dim
    fake_text_embeds = torch.zeros((B, 1, inner_dim), dtype=torch.float32, device=device)
    # Wav2Vec audio feature dimension they used in training was 768:
    fake_audio_embeds = torch.zeros((B, F_lat, 768), dtype=torch.float32, device=device)

    # 14) Initialize latents for diffusion:
    # They used `prepare_latents` in training—but here we pass `latents=ref_latents_full.permute(0, 2, 1, 3, 4)`
    # Actually, training’s `noisy_latents = [clean.permute(0,2,1,3,4)]`. We want to start from noise all the way,
    # but preserve ref_latents_full for re‐injection. So we set initial_latents = [torch.randn_like(clean).permute(...)].
    # But CAPVideoPipeline already does “if sequence_infos[idx][0] is False, then it re‐samples noise internally.”
    # In our code, we want to explicitly pass `latents` = a single‐chunk list where chunk=ref_latents_full.permute(…).
    # In this implementation of __call__, `sequence_infos[0][0] = False`, so the pipeline will ignore that “ref_latent”
    # for initial noise generation, and replace with random noise internally. So:
    latents_for_pipeline = [ref_latents_full.permute(0, 2, 1, 3, 4).to(device)]

    uncond_latents_for_pipeline = [torch.zeros_like(ref_latents_full.permute(0,2,1,3,4))]

    # 15) Finally: call pipeline
    output_videos = pipe(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=args.num_inference_steps,
        latents=latents_for_pipeline,
        uncond_latents=uncond_latents_for_pipeline,
        conditioning=cond_chunks,
        uncond_conditioning=uncond_cond_chunks,
        ref_mask_chunks=ref_mask_chunks,
        audio_embeds=fake_audio_embeds,
        uncond_audio_embeds=torch.zeros_like(fake_audio_embeds),
        text_embeds=fake_text_embeds,
        uncond_text_embeds=torch.zeros_like(fake_text_embeds),
        sequence_infos=sequence_infos,
        guidance_scale=1.0,          # no classifier-free guidance
        use_dynamic_cfg=False,
        eta=0.0,
        generator=None,
        output_type="pil",
        return_dict=True,
    )
    # output_videos is a List of length = num_chunks (1), each an object [B, 3, F, H, W] in “pil” format.
    # So output_videos[0] is a list-of-lists of PIL Images: shape = [B][F], but B=1 here.
    pil_video: List[Image.Image] = output_videos[0][0]  # first (and only) batch element, list of F PIL frames

    # 16) Save as MP4 using torchvision or imageio
    # We'll use imageio-ffmpeg to write an mp4 at 16fps (you can change fps as desired).
    import imageio

    writer = imageio.get_writer(args.output, format="FFMPEG", mode="I", fps=16, codec="libx264", bitrate="16M")
    for frame in pil_video:  # each frame is a PIL.Image
        arr = np.array(frame)  # [H,W,3], uint8
        writer.append_data(arr)
    writer.close()
    print(f"Saved generated video to {args.output}")

if __name__ == "__main__":
    main()
