
# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
import sys

import einops
import argparse
import logging
import math
import os
import shutil
import yaml
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
from tqdm.auto import tqdm
import numpy as np
from decord import VideoReader
from transformers import Wav2Vec2Model
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer
import cv2

import diffusers
from diffusers import AutoencoderKLCogVideoX, CogVideoXDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, export_to_video, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

from data.VideoDataset import VideoDataset, video_collate
from train.trainUtils import collate_fn
from model.cap_transformer import CAPVideoXTransformer3DModel

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for CogVideoX.")

    # Model information
    parser.add_argument(
        "--load_checkpoint_if_exists",
        type=int,
        default=False,
        help="Whether or not to load a checkpoint file if it exists.",
    )
    parser.add_argument(
        "--load_optimizer_state",
        type=int,
        default=False,
        help="Whether or not to load the optimizer state.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        required=True,
        help=("A folder containing the training data."),
    )
    parser.add_argument(
        "--stride_min",
        type=int,
        default=1,
        required=False,
        help=("Minimal stride between frames."),
    )
    parser.add_argument(
        "--stride_max",
        type=int,
        default=3,
        required=False,
        help=("Maximum stride between frames."),
    )
    parser.add_argument(
        "--downscale_coef",
        type=int,
        default=8,
        required=False,
        help=("Downscale coef as encoder decreases resolutio before apply transformer."),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    # CAP4D
    parser.add_argument(
        "--use_text",
        type=bool,
        default=0,
        help=(
            "Use text embeddings."
        ),
    )
    parser.add_argument(
        "--use_camera_conditioning",
        type=bool,
        default=0,
        help=(
            "Use seperate camera conditioning model."
        ),
    )
    parser.add_argument(
        "--use_camera_pos_enc",
        type=bool,
        default=0,
        help=(
            "Use positional encodings for camera conditioning."
        ),
    )
    parser.add_argument(
        "--sample_frames",
        type=int,
        default=50
    )
    # Validation
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help=(
            "Num steps for denoising on validation stage."
        ),
    )
    parser.add_argument(
        "--num_validation_videos",
        type=int,
        default=1,
        help="Number of videos that should be generated during validation per `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_videos`."
        ),
    )
    parser.add_argument(
        "--test_steps",
        type=int,
        default=200,
        help=(
            "Run test every X steps. Testing consists of running the test dataset"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3,
        help="The guidance scale to use while sampling validation videos.",
    )
    parser.add_argument(
        "--unconditional_probability",
        type=float,
        default=0.1,
        help="The dropout probability of all conditioning signals.",
    )
    parser.add_argument(
        "--use_dynamic_cfg",
        action="store_true",
        default=False,
        help="Whether or not to use the default cosine dynamic guidance schedule when sampling validation videos.",
    )

    # Training information
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="cogvideox-controlnet",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        required=True,
        help="Path to model config yaml file.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides `--num_train_epochs`.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--enable_slicing",
        action="store_true",
        default=False,
        help="Whether or not to use VAE slicing for saving memory.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether or not to use VAE tiling for saving memory.",
    )

    # Optimizer
    parser.add_argument(
        "--optimizer",
        type=lambda s: s.lower(),
        default="adam",
        choices=["adam", "adamw", "prodigy"],
        help=("The optimizer type to use."),
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.95, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="Coefficients for computing the Prodigy optimizer's stepsize using running averages. If set to None, uses the value of square root of beta2.",
    )
    parser.add_argument("--prodigy_decouple", action="store_true", help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--prodigy_use_bias_correction", action="store_true", help="Turn on Adam's bias correction.")
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        action="store_true",
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage.",
    )

    # Other information
    parser.add_argument("--tracker_name", type=str, default=None, help="Project tracker name")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help="Directory where logs are stored.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )

    return parser.parse_args()


def read_video(video_path, start_index=0, frames_count=49, stride=1):
    video_reader = VideoReader(video_path)
    end_index = min(start_index + frames_count * stride, len(video_reader)) - 1
    batch_index = np.linspace(start_index, end_index, frames_count, dtype=int)
    numpy_video = video_reader.get_batch(batch_index).asnumpy()
    return numpy_video


# def _get_t5_prompt_embeds(
#     tokenizer: T5Tokenizer,
#     text_encoder: T5EncoderModel,
#     prompt: Union[str, List[str]],
#     num_videos_per_prompt: int = 1,
#     max_sequence_length: int = 226,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
#     text_input_ids=None,
# ):
#     prompt = [prompt] if isinstance(prompt, str) else prompt
#     batch_size = len(prompt)

#     if tokenizer is not None:
#         text_inputs = tokenizer(
#             prompt,
#             padding="max_length",
#             max_length=max_sequence_length,
#             truncation=True,
#             add_special_tokens=True,
#             return_tensors="pt",
#         )
#         text_input_ids = text_inputs.input_ids
#     else:
#         if text_input_ids is None:
#             raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

#     prompt_embeds = text_encoder(text_input_ids.to(device))[0]
#     prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

#     # duplicate text embeddings for each generation per prompt, using mps friendly method
#     _, seq_len, _ = prompt_embeds.shape
#     prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
#     prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

#     return prompt_embeds


# def encode_prompt(
#     tokenizer: T5Tokenizer,
#     text_encoder: T5EncoderModel,
#     prompt: Union[str, List[str]],
#     num_videos_per_prompt: int = 1,
#     max_sequence_length: int = 226,
#     device: Optional[torch.device] = None,
#     dtype: Optional[torch.dtype] = None,
#     text_input_ids=None,
# ):
#     prompt = [prompt] if isinstance(prompt, str) else prompt
#     prompt_embeds = _get_t5_prompt_embeds(
#         tokenizer,
#         text_encoder,
#         prompt=prompt,
#         num_videos_per_prompt=num_videos_per_prompt,
#         max_sequence_length=max_sequence_length,
#         device=device,
#         dtype=dtype,
#         text_input_ids=text_input_ids,
#     )
#     return prompt_embeds


# def compute_prompt_embeddings(
#     tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
# ):
#     if requires_grad:
#         prompt_embeds = encode_prompt(
#             tokenizer,
#             text_encoder,
#             prompt,
#             num_videos_per_prompt=1,
#             max_sequence_length=max_sequence_length,
#             device=device,
#             dtype=dtype,
#         )
#     else:
#         with torch.no_grad():
#             prompt_embeds = encode_prompt(
#                 tokenizer,
#                 text_encoder,
#                 prompt,
#                 num_videos_per_prompt=1,
#                 max_sequence_length=max_sequence_length,
#                 device=device,
#                 dtype=dtype,
#             )
#     return prompt_embeds


# def encode_audio(
#     audio_model, audio, sample_positions, n_latents, temporal_compression_ratio,
# ):
#     # print("AUDIO_DTYPE", audio.dtype)
#     features = audio_model(audio).last_hidden_state.to(dtype=sample_positions.dtype)

#     assert sample_positions.shape[0] == 1  # assert batch size of one
#     first_index = (sample_positions[0][0] * features.shape[1]).long()  # WARNING: BATCH SIZE SQUEEZE
#     last_index = ((sample_positions[0][-1] + 1) * features.shape[1]).long()  # WARNING: BATCH SIZE SQUEEZE

#     # print(sample_positions.min(), sample_positions.max(), sample_positions.shape)
#     # print("audio", first_index, last_index, features.shape)

#     audio_embeds = features[:, first_index:last_index]

#     n_frames = (n_latents - 1) * temporal_compression_ratio + 1
#     audio_embeds = F.interpolate(audio_embeds.permute(0, 2, 1)[..., None], (n_frames, 1), mode="bilinear", align_corners=False)
#     audio_embeds = audio_embeds[..., 0].permute(0, 2, 1) # B T C
#     audio_embeds = F.pad(audio_embeds, (0, 0, temporal_compression_ratio-1, 0), mode="constant", value=0.)
#     audio_embeds = einops.rearrange(audio_embeds, 'b (t s) c -> b t (s c)', s=temporal_compression_ratio)

#     return audio_embeds


# def prepare_rotary_positional_embeddings(
#     height: int,
#     width: int,
#     num_frames: int,
#     vae_scale_factor_spatial: int = 8,
#     patch_size: int = 2,
#     attention_head_dim: int = 64,
#     device: Optional[torch.device] = None,
#     base_height: int = 480,
#     base_width: int = 720,
# ) -> Tuple[torch.Tensor, torch.Tensor]:
#     grid_height = height // (vae_scale_factor_spatial * patch_size)
#     grid_width = width // (vae_scale_factor_spatial * patch_size)
#     base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
#     base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

#     grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)
#     freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
#         embed_dim=attention_head_dim,
#         crops_coords=grid_crops_coords,
#         grid_size=(grid_height, grid_width),
#         temporal_size=num_frames,
#     )

#     freqs_cos = freqs_cos.to(device=device)
#     freqs_sin = freqs_sin.to(device=device)
#     return freqs_cos, freqs_sin


def get_optimizer(args, params_to_optimize, use_deepspeed: bool = False):
    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if args.optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not (args.optimizer.lower() not in ["adam", "adamw"]):
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if args.optimizer.lower() == "adamw":
        optimizer_class = bnb.optim.AdamW8bit if args.use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if args.use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.adam_weight_decay,
        )
    elif args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    return optimizer


##############################################################################################################################
##############################################################################################################################
#### MAIN
##############################################################################################################################
##############################################################################################################################


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)
    output_dir = Path(args.output_dir)

    with open(args.model_config_path) as f:
        model_config_yaml = yaml.safe_load(f)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # kwargs = DistributedDataParallelKwargs()

    logging.basicConfig(level=logging.INFO)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
        print("Seed:", args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Prepare models and scheduler
    if args.use_text:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
        )

        text_encoder = T5EncoderModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )

    # CogVideoX-2b weights are stored in float16
    # CogVideoX-5b and CogVideoX-5b-I2V weights are stored in bfloat16
    
    load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
    transformer = CAPVideoXTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        low_cpu_mem_usage=False,
        device_map=None,
        ignore_mismatched_sizes=True,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
        cond_in_channels=1,
        sample_width=model_config_yaml["width"] // 8,
        sample_height=model_config_yaml["height"] // 8,
        sample_frames=args.sample_frames,
        max_text_seq_length=1,
        max_n_references=model_config_yaml["max_n_references"],
        apply_attention_scaling=model_config_yaml["use_growth_scaling"],
    )

    # audio_dtype = torch.bfloat16

    # if model_config_yaml["use_audio"]:
    #     audio_model = Wav2Vec2Model.from_pretrained(
    #         "facebook/wav2vec2-base", 
    #         torch_dtype=audio_dtype, 
    #         attn_implementation="flash_attention_2",
    #     )
    #     audio_model.freeze_feature_encoder()
    #     audio_model.encoder.config.layerdrop = 0.

    #################################################################################################################
    #################################################################################################################
    ###### INITIALIZATION
    #################################################################################################################
    #################################################################################################################

    # Dataset and DataLoader
    # dataset_config = OmegaConf.load(args.dataset_config_path)
    train_dataset: VideoDataset = VideoDataset(
        "/scratch/ondemand28/harryscz/head_audio/data/data256/uv"
    )

    sampler = DistributedSampler(
        train_dataset, 
        num_replicas=accelerator.num_processes, 
        rank=accelerator.process_index, 
        shuffle=True
    )
    def worker_init_fn(worker_id):
        seed = torch.initial_seed() % (2**32)
        np.random.seed(seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler, 
        collate_fn=video_collate,
        num_workers=args.dataloader_num_workers,
        worker_init_fn=worker_init_fn,
    )

    # conditioning = CAPConditioning()

    # if model_config_yaml["use_camera_conditioning"]:
    #     cam_conditioning = CAPCameraConditioning(use_positional_encoding=model_config_yaml["use_camera_pos_enc"])
    #     cam_conditioning.requires_grad_(False)

    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    
    scheduler = CogVideoXDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.enable_slicing:
        vae.enable_slicing()
    if args.enable_tiling:
        vae.enable_tiling()

    # We only train the additional adapter controlnet layers
    # if args.use_text:
    #     text_encoder.requires_grad_(False)
    # if model_config_yaml["use_audio"]:
    #     audio_model.requires_grad_(True)
    transformer.requires_grad_(True)
    vae.requires_grad_(False)
    # conditioning.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # MOVE MODELS TO DEVICES

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    # if args.mixed_precision == "fp16":
        # only upcast trainable parameters into fp32
        # cast_training_params([controlnet], dtype=torch.float32)

    trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # if model_config_yaml["use_audio"]:
    #     audio_parameters = list(filter(lambda p: p.requires_grad, audio_model.parameters()))
    #     print("number of audio layers:", len(audio_parameters))
    #     trainable_parameters += audio_parameters

    # Optimization parameters
    trainable_parameters_with_lr = {"params": trainable_parameters, "lr": args.learning_rate}
    params_to_optimize = [trainable_parameters_with_lr]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    # CREATE OPTIMIZER

    optimizer = get_optimizer(args, params_to_optimize, use_deepspeed=use_deepspeed_optimizer)

    global_step = 0
    if args.load_checkpoint_if_exists:
        # check if checkpoints exist:
        recent_checkpoint = None
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            recent_checkpoint = checkpoints[0]
            print("Loading checkpoint from", recent_checkpoint)
            recent_checkpoint = torch.load(recent_checkpoint, map_location='cpu')

            transformer.load_state_dict(recent_checkpoint["state_dict"])

            # if model_config_yaml["use_audio"]:
            #     audio_model.load_state_dict(recent_checkpoint["audio_state_dict"])
        
    def encode_video(video):
        video = video.to(accelerator.device, dtype=vae.dtype)
        video = video.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]
        latent_dist = vae.encode(video).latent_dist.sample() * vae.config.scaling_factor
        return latent_dist.permute(0, 2, 1, 3, 4).to(memory_format=torch.contiguous_format)

    # if args.use_text:
    #     text_encoder.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    # conditioning.to(accelerator.device, dtype=torch.float32)
    # if model_config_yaml["use_audio"]:
    #     audio_model.to(accelerator.device, dtype=weight_dtype)
    # if model_config_yaml["use_camera_conditioning"]:
    #     cam_conditioning.to(accelerator.device, dtype=torch.float32)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # INITIATE INFERENCE PIPELINE

    # inference_pipeline = init_pipeline(
    #     args.pretrained_model_name_or_path,
    #     transformer=unwrap_model(transformer),
    #     # TODO: AUDIO
    #     vae=unwrap_model(vae),
    #     scheduler=scheduler,
    #     dtype=weight_dtype,
    # )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    if use_deepspeed_scheduler:
        # from accelerate.utils import DummyScheduler
        # assert False

        # lr_scheduler = DummyScheduler(
        #     name=args.lr_scheduler,
        #     optimizer=optimizer,
        #     total_num_steps=args.max_train_steps * accelerator.num_processes,
        #     num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        #     # last_epoch=global_step,
        # )
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
            # last_epoch=global_step,
        )
    else:
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps,
            num_training_steps=args.max_train_steps,
            num_cycles=args.lr_num_cycles,
            power=args.lr_power,
            # last_epoch=global_step,
        )

    # Prepare everything with our `accelerator`.
    # if model_config_yaml["use_audio"]:
    #     (transformer, audio_model), optimizer, lr_scheduler = accelerator.prepare(
    #         (transformer, audio_model), optimizer, lr_scheduler
    #     )
    #     # transformer, audio_model, optimizer, lr_scheduler = accelerator.prepare(
    #     #     transformer, audio_model, optimizer, lr_scheduler
    #     # )
    #     # print(audio_model.module.module.encoder.layers[0].feed_forward.intermediate_dense.weight[:10])
    # else:
    print(f"Rank {accelerator.process_index}: About to prepare model...")
    print(f"Rank {accelerator.process_index}: Model has {sum(p.numel() for p in transformer.parameters())} parameters")

    # Force synchronization
    if accelerator.num_processes > 1:
        torch.distributed.barrier()
        print(f"Rank {accelerator.process_index}: Passed barrier")

    transformer, optimizer, lr_scheduler = accelerator.prepare(  # REMOVED train_dataloader
        transformer, optimizer, lr_scheduler
    )

    if args.load_checkpoint_if_exists and recent_checkpoint is not None:
        if args.load_optimizer_state:
            global_step = recent_checkpoint["global_step"]
            print("Loading optimizer states and starting at global step", global_step)
            optimizer.load_state_dict(recent_checkpoint["optimizer"])
        del recent_checkpoint


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = args.tracker_name or "cogvideox"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_trainable_parameters = sum(param.numel() for model in params_to_optimize for param in model["params"])

    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {num_trainable_parameters}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)

    print("TODO: implement multi ref and ref in dataloader")
    torch.cuda.empty_cache()

    # For DeepSpeed training
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        # if model_config_yaml["use_audio"]:
        #     audio_model.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            # if model_config_yaml["use_audio"]:
            #     models_to_accumulate += [audio_model]

            # TODO: IMPLEMENT CFG
            is_uncond = torch.rand(1) < args.unconditional_probability

            with accelerator.accumulate(models_to_accumulate):
                latent_chunks = []
                for chunk_id in range(len(batch["video_chunks"])):
                    video = batch["video_chunks"].permute(0,2,1,3,4)
                    if video.shape[1] > args.sample_frames: video = video[:,:args.sample_frames,...]   
                    latent_chunk = encode_video(video).to(dtype=weight_dtype)
                    latent_chunks.append(latent_chunk)
                    # batch_cond_chunk = {key: batch["cond_chunks"][key][chunk_id] for key in batch["cond_chunks"]}
                    # for key in batch_cond_chunk:
                    #     batch_cond_chunk[key] = batch_cond_chunk[key].to(dtype=torch.float32, device=accelerator.device)
                    
                    # cond_chunk = conditioning(batch_cond_chunk, latent_chunk.shape[-2])
                    # ref_mask_chunks.append(cond_chunk["ref_mask"].permute(0, 1, 4, 2, 3))
                    # if model_config_yaml["disable_conditioning"]:
                    #     cond_chunk["condition"][..., :-2] = 0.
                    # if is_uncond:
                    #     cond_chunk["condition"] *= 0.
                    # cond_chunk = einops.rearrange(cond_chunk["condition"], 'b f h w c -> b f c h w').to(device=accelerator.device)
                    # cond_chunk = cond_chunk.to(dtype=weight_dtype)
                    # cond_chunks.append(cond_chunk)

                    # if model_config_yaml["use_camera_conditioning"]:
                    #     cam_cond_chunk = cam_conditioning(batch_cond_chunk, latent_chunk.shape[-2])
                    #     cam_cond_chunk = einops.rearrange(cam_cond_chunk["condition"], 'b f h w c -> b f c h w').to(device=accelerator.device)
                    #     cam_cond_chunk = cam_cond_chunk.to(dtype=weight_dtype)
                    #     cam_cond_chunks.append(cam_cond_chunk)

                    # print(chunk_id, latent_chunks[-1].shape, cond_chunks[-1].shape)
                
                # # encode prompts
                # if args.use_text:
                #     assert False
                #     prompt_embeds = compute_prompt_embeddings(
                #         tokenizer,
                #         text_encoder,
                #         prompts,
                #         model_config.max_text_seq_length,
                #         accelerator.device,
                #         weight_dtype,
                #         requires_grad=False,
                #     )

                # Sample noise that will be added to the latents
                batch_size, num_frames, num_channels, height, width = latent_chunks[0].shape
                # assert batch_size == 1  # only batch size 1 supported for now

                # if model_config_yaml["use_audio"]:
                #     audio_encoding = encode_audio(
                #         audio_model=audio_model,
                #         audio=batch["audio"].to(dtype=weight_dtype, device=accelerator.device),
                #         sample_positions=batch["sample_positions"].to(dtype=weight_dtype, device=accelerator.device),
                #     )
                #     if not batch["has_audio"][0]:  # if there is no audio, set audio encoding to zero
                #         audio_encoding = audio_encoding * 0. #  torch.zeros_like(audio_encoding)
                #         # print('no audio')
                #     if is_uncond:
                #         audio_encoding = audio_encoding * 0.
                # else:
                #     # TODO creat audio encoding
                audio_encoding = torch.zeros(
                    batch_size, batch["video_chunks"][-1].shape[1], 768, dtype=weight_dtype, device=accelerator.device
                )

                text_embeds = torch.zeros(batch_size, 1, 1920, device=accelerator.device, dtype=weight_dtype)

                if False:
                    weight = audio_model.module.encoder.layers[-1].feed_forward.intermediate_dense.weight.detach().cpu()
                    print(weight.max().item(), weight.min().item())

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (batch_size,), device=accelerator.device
                )
                timesteps = timesteps.long()
        
                # Prepare rotary embeds
                assert not model_config.use_rotary_positional_embeddings
                if False:
                    image_rotary_emb = (
                        prepare_rotary_positional_embeddings(
                            height=args.height,
                            width=args.width,
                            num_frames=num_frames,
                            vae_scale_factor_spatial=vae_scale_factor_spatial,
                            patch_size=model_config.patch_size,
                            attention_head_dim=model_config.attention_head_dim,
                            device=accelerator.device,
                        )
                        if model_config.use_rotary_positional_embeddings
                        else None
                    )
                else:
                    image_rotary_emb = None

                sequence_infos = []

                # Add noise to the model input according to the noise magnitude at each timestep
                # (this is the forward diffusion process)

                # TODO: ADD NOISE TO REF LATENTS!! ===========================================================================

                noisy_latents = []
                for chunk_id in range(len(latent_chunks)):
                    latent = latent_chunks[chunk_id]
                    # ref_latent = latent * 0. if is_uncond else latent
                    # ref_mask_chunk = ref_mask_chunks[chunk_id]
                    noise = torch.randn_like(latent)
                    noisy_latent = scheduler.add_noise(latent, noise, timesteps)
                    # noisy_latent = noisy_latent * (1. - ref_mask_chunk) + ref_latent * ref_mask_chunk
                    noisy_latents.append(noisy_latent.to(dtype=weight_dtype))
                    sequence_infos.append((False, torch.arange(0, latent.shape[1], device=accelerator.device)))
                B, F_z, C, H_z, W_z = latent_chunks[0].shape
                zero_cond = [torch.zeros((B, F_z, 1, H_z, W_z), dtype=weight_dtype, device=accelerator.device)] * len(latent)
                # Predict the noise residual
                model_outputs = transformer(
                    hidden_states=noisy_latents,
                    encoder_hidden_states=text_embeds,
                    audio_embeds=audio_encoding,
                    condition=zero_cond,
                    sequence_infos=sequence_infos,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                )[0]

                model_output = torch.cat(model_outputs, dim=1)
                model_input = torch.cat(latent_chunks, dim=1)
                noisy_input = torch.cat(noisy_latents, dim=1)

                # print("model_output", model_output.min(), model_output.max())
                model_pred = scheduler.get_velocity(model_output, noisy_input, timesteps)

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                target = model_input

                loss = (weights * (model_pred - target) ** 2)
                # loss = loss * non_ref_mask / non_ref_mask.mean()
                loss = torch.mean(loss.reshape(batch_size, -1), dim=1)
                loss = loss.mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                if accelerator.state.deepspeed_plugin is None:
                    optimizer.step()
                    optimizer.zero_grad()

                lr_scheduler.step(global_step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}.pt")
                        save_dict = {
                            'state_dict': unwrap_model(transformer).state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                        }
                        torch.save(save_dict, save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            # if accelerator.is_main_process:
            #     monitor_dataloaders = []
            #     if (step + 1) == args.gradient_accumulation_steps * 2:
            #         monitor_dataloaders.append(("val", val_dataloader, 1.))
            #         monitor_dataloaders.append(("test", test_dataloader, args.guidance_scale))
            #     if (step + 1) % args.validation_steps == 0:
            #         monitor_dataloaders.append(("val", val_dataloader, 1.))
            #     if (step + 1) % args.test_steps == 0:
            #         monitor_dataloaders.append(("test", test_dataloader, args.guidance_scale))

            #     for monitor_name, monitor_dataloader, cfg_scale in monitor_dataloaders:
            #         for test_id, batch in enumerate(monitor_dataloader):
            #             if test_id + 1 > args.num_validation_videos:
            #                 break

            #             latent_chunks = []
            #             uncond_latent_chunks = []
            #             cond_chunks = []
            #             uncond_chunks = []
            #             cond_vises = []
            #             ref_mask_chunks = []
            #             cam_cond_chunks = []
            #             for chunk_id in range(len(batch["video_chunks"])):
            #                 latent_chunk = encode_video(batch["video_chunks"][chunk_id]).to(dtype=weight_dtype)
            #                 latent_chunks.append(latent_chunk)
            #                 uncond_latent_chunks.append(latent_chunk * 0.)
            #                 batch_cond_chunk = {key: batch["cond_chunks"][key][chunk_id] for key in batch["cond_chunks"]}
            #                 for key in batch_cond_chunk:
            #                     batch_cond_chunk[key] = batch_cond_chunk[key].to(dtype=torch.float32, device=accelerator.device)
            #                 cond_chunk = conditioning(batch_cond_chunk, latent_chunk.shape[-2])
            #                 ref_mask_chunks.append(cond_chunk["ref_mask"].permute(0, 1, 4, 2, 3))
            #                 if model_config_yaml["disable_conditioning"]:
            #                     cond_chunk["condition"][..., :-2] = 0.
            #                 cond_vis = conditioning.get_vis(cond_chunk["condition"])
            #                 cond_chunk = einops.rearrange(cond_chunk["condition"], 'b f h w c -> b f c h w').to(device=accelerator.device)
            #                 cond_chunk = cond_chunk.to(dtype=weight_dtype)
            #                 cond_chunks.append(cond_chunk)
            #                 uncond_chunks.append(cond_chunk * 0.)

            #                 if model_config_yaml["use_camera_conditioning"]:
            #                     cam_cond_chunk = cam_conditioning(batch_cond_chunk, latent_chunk.shape[-2])
            #                     cam_cond_vis = cam_conditioning.get_vis(cam_cond_chunk["condition"])
            #                     for key in cam_cond_vis:
            #                         cond_vis[key] = cam_cond_vis[key]
            #                     cam_cond_chunk = einops.rearrange(cam_cond_chunk["condition"], 'b f h w c -> b f c h w').to(device=accelerator.device)
            #                     cam_cond_chunk = cam_cond_chunk.to(dtype=weight_dtype)
            #                     cam_cond_chunks.append(cam_cond_chunk)
                            
            #                 cond_vises.append(cond_vis)

            #             if model_config_yaml["use_audio"]:
            #                 audio_model.eval()
            #                 with torch.no_grad():
            #                     audio_encoding = encode_audio(
            #                         audio_model=audio_model,
            #                         audio=batch["audio"].to(dtype=audio_dtype, device=accelerator.device),
            #                         sample_positions=batch["sample_positions"].to(dtype=weight_dtype, device=accelerator.device),
            #                     )
            #                 audio_model.train()
            #                 if not batch["has_audio"][0]:  # if there is no audio, set audio encoding to zero
            #                     audio_encoding = audio_encoding * 0.
            #             else:
            #                 audio_encoding = torch.zeros(
            #                     batch_size, batch["video_chunks"].shape[1], 768, dtype=weight_dtype, device=accelerator.device
            #                 )
            #             uncond_audio_encoding = audio_encoding * 0.

            #             audio_encoding = audio_encoding.to(accelerator.device)
            #             text_embeds = torch.zeros(1, 1, 1920, device=accelerator.device, dtype=weight_dtype)

            #             sequence_infos = []
            #             for chunk_id, latent in enumerate(latent_chunks):
            #                 sequence_infos.append((batch["chunk_is_ref"][chunk_id], torch.arange(0, latent.shape[1], device=accelerator.device)))

            #             # run inference
            #             pred_videos = run_pipe(
            #                 inference_pipeline,
            #                 height=model_config_yaml["height"],
            #                 width=model_config_yaml["width"],
            #                 num_frames=batch["video_chunks"][-1].shape[1],
            #                 num_inference_steps=args.num_inference_steps,
            #                 conditioning=cond_chunks,
            #                 cam_conditioning=cam_cond_chunks,
            #                 uncond_conditioning=uncond_chunks,
            #                 latents=latent_chunks,
            #                 uncond_latents=uncond_latent_chunks,
            #                 ref_mask_chunks=ref_mask_chunks,
            #                 audio_embeds=audio_encoding,
            #                 uncond_audio_embeds=uncond_audio_encoding,
            #                 text_embeds=text_embeds,
            #                 uncond_text_embeds=text_embeds,
            #                 sequence_infos=sequence_infos,
            #                 use_dynamic_cfg=args.use_dynamic_cfg,
            #                 guidance_scale=cfg_scale,
            #                 seed=args.seed,
            #             )

            #             val_file = f"{monitor_name}_e{epoch}_gs{global_step}_{test_id}"

            #             # save results
            #             log_validation(
            #                 file_name=val_file,
            #                 pred_videos=pred_videos,
            #                 input_videos=batch["video_chunks"],
            #                 sequence_infos=sequence_infos,
            #                 cond_vises=cond_vises,
            #                 output_dir=output_dir,
            #                 audio=batch["raw_audio"],
            #             )
    
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    args = get_args()
    main(args)
