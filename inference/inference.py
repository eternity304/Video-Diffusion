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

sys.path.chdir("..")

def parse_args(arg_list=None):
    parser = argparse.ArgumentParser(description="Unconditioned Video Diffusion Inference")
    parser.add_argument("--dataset-path", type=str, required=True,help="Directory containing input reference videos.")
    parser.add_argument("--pretrained-model-name-or-path", type=str, required=True,help="Path or HF ID where transformer/vae/scheduler are stored.")
    parser.add_argument("--checkpoint-path", type=str, required=True,help="Path to fine‐tuned checkpoint containing transformer state_dict.")
    parser.add_argument("--output-dir", type=str, required=True,help="Where to write generated videos.")
    parser.add_argument( "--model-config", type=str, required=True,help="YAML file describing model params (height, width, num_reference, num_target, etc.)")
    parser.add_argument("--batch-size", type=int, default=1,help="Batch size per device (usually 1 for inference).")
    parser.add_argument("--num-inference-steps", type=int, default=50, help="Number of reverse diffusion steps to run.")
    parser.add_argument("--mixed-precision", type=str, default="bf16", help="Whether to run backbone in 'fp16', 'bf16', or 'fp32'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--shuffle", type=int, default=False, help="Whether to shuffle dataset. Usually False for inference.")

    return parser.parse_args(arg_list)