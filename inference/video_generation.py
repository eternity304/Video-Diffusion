
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
import torchaudio

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger

from data.VideoDataset import VideoDataset 
from torch.utils.data import DataLoader, DistributedSampler

from model.scheduler import CogVideoXDPMScheduler
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from original.dit import CAPVideoXTransformer3DModel
from model.flameObj import *
from model.causalvideovae.model import *

from original.inference_pipeline import *
from data.VideoDataset import *
from wf_vae.model import *

def norm_c(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor of shape [T, V, 3] to [-1, 1] per channel.

    Args:
        x (torch.Tensor): Input tensor with shape [T, V, 3].

    Returns:
        torch.Tensor: Normalized tensor with same shape, values in [-1, 1].
    """
    if x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError("Input must have shape [T, V, 3]")

    # Compute per-channel min and max
    min_vals = x.amin(dim=(0, 1), keepdim=True)  # shape [1,1,3]
    max_vals = x.amax(dim=(0, 1), keepdim=True)  # shape [1,1,3]

    # Avoid division by zero
    denom = (max_vals - min_vals).clamp(min=1e-8)

    # Normalize [0,1], then scale [-1,1]
    x01 = (x - min_vals) / denom
    x_norm = x01 * 2 - 1
    return x_norm, min_vals, max_vals

def batch_delta_uv(head, path_list, min_frame=50, sample_frames=50, resolution=256, rotation=False, norm=False):
    out = []
    min_frame = 1e8

    for path in path_list:
        head.loadSequence(path)

        id = head.LSB(rotation=False, identity=True)
        seq = head.LSB(rotation=False)

        first = seq[0:1, ...] - id[0:1, ...]     # Get first difference from identity to sequence
        remainder = seq[1:, ...] - seq[:-1, ...]
        delta = torch.cat([first, remainder], dim=0).cumsum(dim=0)
        
        normed, _, _ = norm_c(delta)
        uvMesh = head.convertUV(customSeq=normed, rotation=False, norm=norm)

        frame, _, _ = normed.shape    
        if frame < min_frame: min_frame = frame                                       
        uv = head.get_uv_animation(uvMesh, resolution=resolution, sample_frames=sample_frames) 
        out.append(uv[..., :3].unsqueeze(0))

    out = [item[:, :min_frame, ...] for item in out]
    return torch.concat(out, dim=0)

def denorm_c(x_norm: torch.Tensor, min_vals: torch.Tensor, max_vals: torch.Tensor):
    """
    Invert the normalization: [-1,1] → original range.
    """
    denom = (max_vals - min_vals).clamp(min=1e-8)
    x01 = (x_norm + 1) / 2
    x = x01 * denom + min_vals
    return x

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Unconditioned Video Diffusion Inference")

    parser.add_argument("--dataset-path", type=str, required=True, help="Directory containing input reference videos.")
    parser.add_argument("--pretrained-model-name-or-path", type=str, required=True, help="Path or HF ID where transformer/vae/scheduler are stored.")
    parser.add_argument("--audio-model-path", type=str, required=True, default=None)
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
    parser.add_argument("--vae-ckpt", type=str, default=None)

    # If arg_list is None, argparse picks up sys.argv; 
    # otherwise it treats arg_list as the full argv list.
    return parser.parse_args(arg_list)

def encode_audio(
    audio_model, audio, sample_positions
):
    # print("AUDIO_DTYPE", audio.dtype)
    features = audio_model(audio).last_hidden_state.to(dtype=sample_positions.dtype)
    assert sample_positions.shape[0] == 1  # assert batch size of one
    first_index = (sample_positions[0][0] * features.shape[1]).long()  # WARNING: BATCH SIZE SQUEEZE
    last_index = (sample_positions[0][-1] * features.shape[1] + 1).long()  # WARNING: BATCH SIZE SQUEEZE

    # print(sample_positions.min(), sample_positions.max(), sample_positions.shape)
    # print("audio", first_index, last_index, features.shape)

    audio_embeds = features[:, first_index:last_index]

    return audio_embeds

def load_audio(
    audio_path: str,
    frame_ids: np.ndarray,
    n_timesteps: int,
    audio_window: int,
    audio_processor: Wav2Vec2Processor,
    min_n_frames: int = 0,
):
    # calculate frame per seconds with num frames and audio duration
    target_rate = 16000

    if audio_path is not None and os.path.exists(audio_path) and len(frame_ids) >= min_n_frames:
        input_audio, rate = torchaudio.load(str(audio_path))
        input_audio = input_audio[0]

        # duration = len(raw_audio) / rate

        if rate != target_rate:
            input_audio = torchaudio.functional.resample(input_audio, rate, target_rate)

        if n_timesteps <= max(frame_ids):
            # pad the audio wav
            n_samples_per_frame = input_audio.shape[0] / n_timesteps
            input_audio = F.pad(input_audio, (0, math.ceil(n_samples_per_frame * (max(frame_ids) - n_timesteps + 1))))
            n_timesteps = max(frame_ids) + 1

        audio_start = frame_ids[0]
        extended_audio_start = max(0, audio_start - audio_window)
        audio_end = frame_ids[-1] + 1
        extended_audio_end = min(n_timesteps, audio_end + audio_window)

        def to_sample(frame_id):
            return int(frame_id / n_timesteps * input_audio.shape[0])

        extended_audio = input_audio[to_sample(extended_audio_start):to_sample(extended_audio_end)]
        raw_audio = input_audio[to_sample(audio_start):to_sample(audio_end)].numpy()

        # this is where the extracted features are grid_sampled from
        sample_positions = (frame_ids - extended_audio_start) / (extended_audio_end - extended_audio_start)

        processed_audio = audio_processor(
            extended_audio,
            sampling_rate=target_rate,
        ).input_values[0]
        
        return True, raw_audio, processed_audio, sample_positions
    else:
        # no audio found, return zeros
        audio_features = np.zeros((target_rate)) # int(frame_ids.shape[0] / 25 * target_rate)))
        sample_positions = np.arange(len(frame_ids)) / len(frame_ids)

        return False, audio_features, audio_features, sample_positions
    
def batch_audio(paths, n_timesteps, audio_processor):
    out_audio, out_positions = [], []

    assert len(paths) == 1

    for path in paths:
        audio_path = os.path.join(os.path.dirname(path), "audio.m4a")
        _, _, audio, sample_positions = load_audio(audio_path, frame_ids=np.arange(0,n_timesteps), n_timesteps=n_timesteps, audio_window=25, audio_processor=audio_processor)
        audio, sample_positions = torch.from_numpy(audio).unsqueeze(0), torch.from_numpy(sample_positions).unsqueeze(0)
        out_audio.append(audio)
        out_positions.append(sample_positions)

    return torch.concat(out_audio, dim=0), torch.concat(out_positions, dim=0)

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
    with torch.no_grad(): latent_dist = vae.encode(video).latent_dist.sample() 
    return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format) * vae.config.scale[0]

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

    head = Flame(flamePath, device="cuda")

    # audio_dtype = torch.float32

    # audio_model = Wav2Vec2Model.from_pretrained(
    #     args.audio_model_path, 
    #     torch_dtype=audio_dtype, 
    #     # attn_implementation="flash_attention_2",
    # )
    # audio_model.freeze_feature_encoder()
    # audio_model.encoder.config.layerdrop = 0.
    # audio_processor = Wav2Vec2Processor.from_pretrained(args.audio_model_path)

    # audio_model.load_state_dict(ckpt["audio_state_dict"], strict=False)

    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        subfolder="transformer",
        torch_dtype=torch.float32,
        num_layers=32,
        cond_in_channels=16,  # only one channel (the ref_mask)
        sample_width=model_config["width"] // 8,
        sample_height=model_config["height"] // 8,
        sample_frames=args.sample_frames,
        max_text_seq_length=1,
        max_n_references=model_config["max_n_references"],
        apply_attention_scaling=model_config["use_growth_scaling"],
        use_rotary_positional_embeddings=False,
    )
    # vae = AutoencoderKLCogVideoX.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="vae"
    # )

    model_cls = ModelRegistry.get_model("WFVAE")
    vae = model_cls.from_pretrained("/scratch/ondemand28/harryscz/other/WF-VAE/weight")

    if args.vae_ckpt:
        vae_ckpt = torch.load(args.vae_ckpt, map_location="cpu")
        vae.load_state_dict(vae_ckpt["state_dict"],  strict=False)
        print("WF VAE checkpoint loaded!")

    # scheduler = CogVideoXDPMScheduler.from_pretrained(
    #     args.pretrained_model_name_or_path, subfolder="scheduler"
    # )
    scheduler = CogVideoXDPMScheduler(
        num_train_timesteps=1000,
        beta_start=8.5e-8,
        beta_end=0.0120,
        beta_schedule="scaled_linear",
        prediction_type="v_prediction",
        rescale_betas_zero_snr="True"
    )

    vae.to(weight_dtype)
    transformer.train().to(weight_dtype)

    ckpt_path = args.ckpt_path
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    transformer.load_state_dict(ckpt["state_dict"], strict=False)

    vae, transformer, scheduler, data_loader = accelerator.prepare(vae, transformer, scheduler, data_loader)
    
    pipe = CAPVideoPipeline(
        vae=vae,
        transformer=transformer,
        scheduler=scheduler
    )
    
    def decode_video(vae, latents):
        with torch.no_grad():
            latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
            frames = vae.decode(latents / vae.config.scale[0]).sample
            return frames

    for i in range(0, 1):

        batch = dataPath[i:i+1]
        print(i, batch)

        latent_chunks = []
        uncond_latent_chunks = []
        cond_chunks = []
        uncond_chunks = []
        ref_mask_chunks = []

        # uvs = head.batch_uv(batch, resolution=256, sample_frames=args.sample_frames, rotation=False).permute(0,1,4,2,3) # load UVs of shape B, F, C, H, W
        uvs = batch_delta_uv(
            head,
            batch
        ).permute(0,1,4,2,3)
        # audio_chunks, sample_pos_chunks = batch_audio(batch, n_timesteps=args.sample_frames, audio_processor=audio_processor)
        # audio_chunks, sample_pos_chunks = audio_chunks.to(device=audio_model.device, dtype=weight_dtype), sample_pos_chunks.to(device=audio_model.device, dtype=weight_dtype)
        
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
            (B, F_z, 3072), dtype=torch.float32, device=accelerator.device
        )
        # audio_encoding = encode_audio(audio_model=audio_model, audio=audio_chunks, sample_positions=sample_pos_chunks)

        uncond_audio_encoding = audio_encoding * 0.
        text_embeds = torch.zeros(1, 1, 1920, device=device, dtype=dtype)

        sequence_infos = []
        for chunk_id, latent in enumerate(latent_chunks):
            sequence_infos.append((False, torch.arange(0, latent.shape[1], device=accelerator.device)))

        orig = uvs[0].permute(0,2,3,1)
        torch.save(orig, "diffOut/gt.pt")
        with torch.no_grad():
            z = vae.encode(orig.permute(3,0,1,2).unsqueeze(0)) # input shape : C, T, H, W
            latents = z.latent_dist.sample()
            orig = vae.decode(latents).sample

        orig = orig[0].permute(1,2,3,0)

        out = pipe(
            height=256,
            width=256,
            num_frames=args.sample_frames,
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
            gt=latents.permute(0,2,1,3,4) * vae.config.scale[0]
        )
        
        min_x, max_x = torch.tensor([[[-0.0050, -0.0057, -0.0085]]], device=orig.device), torch.tensor([[[0.0085, 0.0066, 0.0024]]], device=orig.device)

        head.loadSequence(dataPath[0])
        id = head.LSB(rotation=False, identity=True)

        # recon = out[0][0][0].permute(0,2,3,1)
        recon = decode_video(vae, out[1][0])[0].permute(1,2,3,0)

        torch.save(orig, "diffOut/vae_gt.pt")
        torch.save(recon, "diffOut/overfit.pt")

        sampled_uv = head.sampleFromUV(recon, savePath="diffOut/cumsum_uv.mp4", resolution=512, fill=True)
        head.sampleTo3D(denorm_c(sampled_uv, min_x, max_x), savePath="diffOut/cumsum_3d.mp4", rotation=False, delta=id, norm=False, dist=0.6, resolution=512)

        sampled_gt = head.sampleFromUV(orig, savePath="diffOut/cumsum_uv_gt.mp4", resolution=512, fill=True)
        head.sampleTo3D(denorm_c(sampled_gt, min_x, max_x), savePath="diffOut/cumsum_3d_gt.mp4", rotation=False, delta=id, norm=False, dist=0.6, resolution=512)

        
        # sampledUV = head.sampleFromUV(orig, savePath=f"diffOut/_no_{i}_uv.mp4") # input has shape F, H, W, C
        # head.sampleTo3D(sampledUV, savePath=f"diffOut/_no_{i}_3d.mp4", resolution=512, dist=1.2, rotation=False)

        # sampledUV = head.sampleFromUV(recon, savePath=f"diffOut/_no_{i}_diff_uv.mp4") # input has shape F, H, W, C
        # head.sampleTo3D(sampledUV, savePath=f"diffOut/_no_{i}_diff_3d.mp4", resolution=512, dist=1.2, rotation=False)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
