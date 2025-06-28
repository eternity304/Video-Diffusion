import os
import torch
import random
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.io import read_video

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
        videos_dir: str,
        num_ref_frames: int = 0,
        resize_hw: tuple = (256, 256),
        frame_rate: int = None,
    ):
        super().__init__()
        self.videos_dir = videos_dir
        self.num_ref = num_ref_frames
        self.resize_hw = resize_hw
        self.frame_rate = frame_rate  # if None, keep *all* frames

        exts = {".mp4", ".avi", ".mov", ".mkv"}
        self.video_files = [
            os.path.join(videos_dir, f)
            for f in sorted(os.listdir(videos_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not self.video_files:
            raise ValueError(f"No video files in {videos_dir!r}")

        # resize shortest side to resize_hw[0], then center‐crop to (H, W)
        self.resize_tf = Compose([
            Resize(resize_hw), 
            CenterCrop(resize_hw),
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        path = self.video_files[idx]

        # 1) load & optionally downsample
        video_frames, _, info = read_video(path, pts_unit="sec")
        if self.frame_rate is not None:
            orig_fps = float(info.get("video_fps", 30.0))
            skip = max(1, round(orig_fps / float(self.frame_rate)))
            video_frames = video_frames[::skip]

        # 2) normalize & resize each frame
        processed = []
        for frame in video_frames:               # [H, W, 3], uint8
            f = frame.permute(2,0,1).float() / 255.0  # [3, H, W], float
            f = self.resize_tf(f)                   # [3, H_out, W_out]
            processed.append(f)
        # stack → [N, 3, H, W] → permute → [3, N, H, W]
        video = torch.stack(processed, dim=0).permute(1,0,2,3)
        # add batch dim → [1, 3, N, H, W]
        video = video.unsqueeze(0)

        _, _, N, H, W = video.shape

        # 3) build ref_mask over full length N
        ref_mask = torch.zeros((1, N, H, W), dtype=torch.float32)
        if self.num_ref > 0:
            end = min(self.num_ref, N)
            ref_mask[0, :end] = 1.0

        # 4) chunk_is_ref boolean vector of length N
        chunk_is_ref = torch.arange(N) < self.num_ref

        return {
            "video_chunks":    [video],            # list with [1,3,N,H,W]
            "cond_chunks":     {"ref_mask": [ref_mask]},
            "chunk_is_ref":    [chunk_is_ref],     # list with length-N bool tensor
            "raw_audio":       None,
        }
    
def video_collate(batch):
    # 1) find max frame‐length N in this batch
    max_N = max(sample["video_chunks"][0].shape[2] for sample in batch)

    videos = []
    masks  = []

    for sample in batch:
        # unwrap and squeeze off the batch‐dim
        vid = sample["video_chunks"][0].squeeze(0)    # [3, N_i, H, W]
        N_i = vid.shape[1]

        # pad to max_N along the frame axis
        pad = torch.zeros(
            (vid.shape[0], max_N - N_i, *vid.shape[2:]),
            dtype=vid.dtype,
            device=vid.device
        )
        vid_padded = torch.cat([vid, pad], dim=1)     # [3, max_N, H, W]
        videos.append(vid_padded)

        # mask: 1 for real frames, 0 for padded
        mask = torch.cat([
            torch.ones(N_i,  device=vid.device),
            torch.zeros(max_N - N_i, device=vid.device)
        ])
        masks.append(mask)

    # stack into batch dims
    batched_videos = torch.stack(videos, dim=0)   # [B, 3, max_N, H, W]
    batched_masks  = torch.stack(masks,  dim=0)   # [B, max_N]
    if len(batched_videos.shape) == 4:
        batched_videos = batched_videos.unsqueeze(0)   # [B, 3, max_N, H, W]
        batched_masks  = batched_masks.unsqueeze(0)

    return {
        "video_chunks": batched_videos,
        "mask":  batched_masks,
        # you can also collate cond_chunks/chunk_is_ref here if needed...
    }