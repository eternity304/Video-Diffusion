import inspect
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from einops import rearrange, repeat
from diffusers.video_processor import VideoProcessor
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.models import AutoencoderKLCogVideoX
from diffusers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.cogvideo.pipeline_cogvideox import CogVideoXPipelineOutput, CogVideoXLoraLoaderMixin

from cap_video.cap_transformer import CAPVideoXTransformer3DModel


def resize_for_crop(image, crop_h, crop_w):
    img_h, img_w = image.shape[-2:]
    if img_h >= crop_h and img_w >= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    elif img_h <= crop_h and img_w <= crop_w:
        coef = max(crop_h / img_h, crop_w / img_w)
    else:
        coef = crop_h / img_h if crop_h > img_h else crop_w / img_w 
    out_h, out_w = int(img_h * coef), int(img_w * coef)
    resized_image = transforms.functional.resize(image, (out_h, out_w), antialias=True)
    return resized_image


def prepare_frames(input_images, video_size, do_resize=True, do_crop=True):
    input_images = np.stack([np.array(x) for x in input_images])
    images_tensor = torch.from_numpy(input_images).permute(0, 3, 1, 2) / 127.5 - 1
    if do_resize:
        images_tensor = [resize_for_crop(x, crop_h=video_size[0], crop_w=video_size[1]) for x in images_tensor]
    if do_crop:
        images_tensor = [transforms.functional.center_crop(x, video_size) for x in images_tensor]
    if isinstance(images_tensor, list):
        images_tensor = torch.stack(images_tensor)
    return images_tensor.unsqueeze(0) 


def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
    

class CAPVideoPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
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
        transformer: CAPVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()

        self.register_modules(
            vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            # num_frames // self.vae_scale_factor_temporal, or this
            (num_frames - 1) // self.vae_scale_factor_temporal + 1, # (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae.config.scaling_factor * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
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
            
    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt
        
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 256,
        width: int = 256,
        num_frames: int = 25,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 1,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        uncond_latents: Optional[torch.FloatTensor] = None,
        conditioning: Optional[List[torch.FloatTensor]] = None,
        uncond_conditioning: Optional[List[torch.FloatTensor]] = None,
        ref_mask_chunks: Optional[List[torch.FloatTensor]] = None,
        # ref_latent: torch.FloatTensor = None,
        sequence_infos: Optional[List[int]] = None,
        audio_embeds: Optional[torch.FloatTensor] = None,
        uncond_audio_embeds: Optional[torch.FloatTensor] = None,
        text_embeds: Optional[torch.FloatTensor] = None,
        uncond_text_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 1,
        growth_factor: float = None,
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        if num_frames > 49:
            raise ValueError(
                "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_size * self.vae_scale_factor_spatial
        num_videos_per_prompt = 1

        self._guidance_scale = guidance_scale
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = conditioning[-1].shape[0]

        device = self._execution_device

        if audio_embeds is not None:
            audio_embeds = audio_embeds.to(device)
        if text_embeds is not None:
            text_embeds = text_embeds.to(device)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        # assert not do_classifier_free_guidance  # not implemented yet

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)
        
        # 5. Prepare latents.
        latent_channels = 16 #self.transformer.config.in_channels

        initial_latents = []
        ref_latents = []

        for chunk_id in range(len(latents)):
            if sequence_infos[chunk_id][0]:
                noisy_latent = latents[chunk_id]
            else:
                noisy_latent = randn_tensor(
                    latents[chunk_id].shape, generator, device, conditioning[-1].dtype,
                )
            # noisy_latent = self.prepare_latents(
            #     batch_size * num_videos_per_prompt,
            #     num_channels_latents=latents[chunk_id].shape[2],
            #     num_frames=latents[chunk_id].shape[1],
            #     height=latents[chunk_id].shape[-2],
            #     width=latents[chunk_id].shape[-1],
            #     dtype=conditioning[-1].dtype,
            #     device=device,
            #     generator=generator,
            #     latents=latents[chunk_id] if sequence_infos[chunk_id][0] else None,
            # )
            initial_latents.append(noisy_latent)

            ref_latent = latents[chunk_id]
            if do_classifier_free_guidance:
                ref_latent = torch.cat([ref_latent, uncond_latents[chunk_id]], dim=0)
            ref_latents.append(ref_latent)

        latents = initial_latents
        
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 8. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 9. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_samples = [None] * len(latents)
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                
                latent_model_inputs = []
                for latent, ref_latent, ref_mask in zip(latents, ref_latents, ref_mask_chunks):
                    # assert not do_classifier_free_guidance
                    latent_model_input = torch.cat([latent] * 2, dim=0) if do_classifier_free_guidance else latent
                    # latent_model_input = self.scheduler.scale_model_input(latent_model_input, t) WE CANT SCALE REF

                    # apply ref mask!
                    latent_model_input = latent_model_input * (1. - ref_mask) + ref_latent * ref_mask

                    latent_model_input = latent_model_input.to(dtype=self.transformer.dtype)
                    latent_model_inputs.append(latent_model_input)

                conditioning_input = []
                if do_classifier_free_guidance:
                    for cond, uncond in zip(conditioning, uncond_conditioning):
                        conditioning_input.append(torch.cat([cond, uncond]))
                    audio_input = torch.cat([audio_embeds, uncond_audio_embeds])
                    text_input = torch.cat([text_embeds, uncond_text_embeds])
                else:
                    conditioning_input = conditioning
                    text_input = text_embeds
                    audio_input = audio_embeds
                
                timestep = t.expand(latent_model_inputs[0].shape[0])

                current_sampling_percent = i / len(timesteps)

                # prompt_embeds = torch.zeros(batch_size, 1, 4096, dtype=self.transformer.dtype, device=self.transformer.device)

                # predict noise model_output
                noise_preds = self.transformer(
                    hidden_states=latent_model_inputs,
                    encoder_hidden_states=text_input,
                    audio_embeds=audio_input,
                    condition=conditioning_input,
                    sequence_infos=sequence_infos,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    growth_factor=growth_factor,
                )[0]
                new_latents = []
                new_old_pred_original_samples = []
                for noise_pred, latent, old_pred_original_sample, ref_latent, ref_mask in zip(
                    noise_preds, latents, old_pred_original_samples, ref_latents, ref_mask_chunks
                ):
                    noise_pred = noise_pred.float()

                    # perform guidance
                    if use_dynamic_cfg:
                        assert False
                        self._guidance_scale = 1 + guidance_scale * (
                            (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                        )
                    if do_classifier_free_guidance:
                        noise_pred, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred - noise_pred_uncond)
                        ref_latent, _ = ref_latent.chunk(2)

                    # compute the previous noisy sample x_t -> x_t-1
                    if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                        assert False
                        latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    else:
                        latent, old_pred_original_sample = self.scheduler.step(
                            noise_pred,
                            old_pred_original_sample,
                            t,
                            timesteps[i - 1] if i > 0 else None,
                            latent,
                            **extra_step_kwargs,
                            return_dict=False,
                        )

                    latent = latent * (1. - ref_mask) + ref_latent * ref_mask
                    
                    latent = latent.to(audio_embeds.dtype)
                    new_latents.append(latent)
                    new_old_pred_original_samples.append(old_pred_original_sample)
                
                latents = new_latents
                old_pred_original_samples = new_old_pred_original_samples

                # call the callback, if provided
                if callback_on_step_end is not None:
                    assert False
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            videos = []
            for latent in latents:
                video = self.decode_latents(latent)
                video = self.video_processor.postprocess_video(video=video, output_type=output_type)
                videos.append(video)
        else:
            videos = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (videos,)

        return CogVideoXPipelineOutput(frames=videos)