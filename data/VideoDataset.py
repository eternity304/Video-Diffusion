import os
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.io import read_video
from model.flameObj import *

class VideoPathDataset(Dataset):
    def __init__(
        self,
        source_dir
    ):
        super().__init__()
        self.source_dir = source_dir
        self.dataPath = [os.path.join(os.path.join(source_dir, data), "fit.npz") for data in os.listdir(source_dir)][:1]

    def __len__(self):
        return len(self.dataPath)

    def __getitem__(self, idx):
        return self.dataPath[idx]

def collate_fn(batch):
    return batch

class VideoDataset(Dataset):
    """
    Read each video in its entirety, resize, and return:
      - video_chunks:   [ tensor([1, 3, N, H, W]) ]         (N = #frames after downsampling)
      - cond_chunks:    {'ref_mask': [ tensor([1, N, H, W]) ]}
      - chunk_is_ref:   [ tensor([bool] * N) ]
      - raw_audio:      None
    """
    def __init__(
        self,
        source_dir: str,
        num_ref_frames: int = 50,
        resolution: tuple = (512, 512),
        frame_rate: int = None,
    ):
        super().__init__()
        self.source_dir = source_dir
        self.num_ref = num_ref_frames
        self.resolution = resolution
        self.frame_rate = frame_rate  # if None, keep *all* frames

        flamePath = "/scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz"
        self.dataPath = [os.path.join(os.path.join(source_dir, data), "fit.npz") for data in os.listdir(source_dir)]
        seqPath = "/scratch/ondemand28/harryscz/head/_-91nXXjrVo_00/fit.npz"

        self.head = Flame(flamePath, device="cuda")

    def __len__(self):
        return len(self.dataPath)

    def __getitem__(self, idx):
        path = self.dataPath[idx]

        self.head.loadSequence(path)                                       
        perFrameVerts = self.head.LSB()                                           
        uvMesh = self.head.convertUV()     
        uv = self.head.get_uv_animation(uvMesh, resolution=self.resolution)

        return {
            "video_chunks":    [uv],            # list with [1,3,N,H,W]
            # "cond_chunks":     {"ref_mask": [ref_mask]},
            # "chunk_is_ref":    [chunk_is_ref],     # list with length-N bool tensor
            # "raw_audio":       None,
        }
    
def video_collate(batch):
    # 1) find max frame-length N in this batch
    max_N = max(sample["video_chunks"][0].shape[0] for sample in batch)  # shape: [N, H, W, C]

    videos = []
    masks  = []

    for sample in batch:
        vid = sample["video_chunks"][0].squeeze(0)  # shape: [N_i, H, W, C]
        N_i = vid.shape[0]

        # pad along frame axis (dim=0)
        pad = torch.zeros(
            (max_N - N_i, *vid.shape[1:]),
            dtype=vid.dtype,
            device=vid.device
        )
        vid_padded = torch.cat([vid, pad], dim=0)  # shape: [max_N, H, W, C]
        videos.append(vid_padded)

        # mask: 1 for real frames, 0 for padded
        mask = torch.cat([
            torch.ones(N_i, device=vid.device),
            torch.zeros(max_N - N_i, device=vid.device)
        ])
        masks.append(mask)

    # stack batch dimension
    batched_videos = torch.stack(videos, dim=0)  # [B, max_N, H, W, C]
    batched_masks  = torch.stack(masks, dim=0)   # [B, max_N]

    return {
        "video_chunks": batched_videos,
        "mask": batched_masks,
    }