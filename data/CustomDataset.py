import os
import random
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_video

class VideoDataset(Dataset):
    """
    Given a folder of video files, this Dataset returns for each index:
      - video_chunks: [ [1, 3, F, H, W] ]       (a list of length 1)
      - cond_chunks: {'ref_mask': [ [1, F, H, W] ]}
      - chunk_is_ref: [ [True/False × F] ]
      - raw_audio: None   (we’re not using audio)
    
    Here, F = num_ref_frames + num_target_frames.
    The first num_ref_frames frames in each chunk are marked as reference.
    """
    def __init__(
        self,
        videos_dir: str,
        num_ref_frames: int,
        num_target_frames: int,
        resize_hw: tuple = (256, 256),
        frame_rate: int = 16,
    ):
        """
        Args:
          videos_dir: path to folder containing .mp4 (or .avi, etc.) files.
          num_ref_frames: how many frames in each chunk to treat as reference (keep unnoised).
          num_target_frames: how many frames in each chunk to actually train/denoise.
          resize_hw: resize each frame to this (H, W) before stacking into tensors.
          frame_rate: if you want to uniformly sample at this FPS; pass None to read all frames.
        """
        super().__init__()
        self.videos_dir = videos_dir
        self.num_ref = num_ref_frames
        self.num_target = num_target_frames
        self.total_frames = num_ref_frames + num_target_frames
        self.resize_hw = resize_hw
        self.frame_rate = frame_rate

        # List all video files in the directory
        exts = {".mp4", ".avi", ".mov", ".mkv"}
        self.video_files = [
            os.path.join(videos_dir, f)
            for f in sorted(os.listdir(videos_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if len(self.video_files) == 0:
            raise ValueError(f"No video files found in {videos_dir}.")

        # Precompute a torchvision resize transform:
        self.resize_transform = T.Compose([
            T.Resize(resize_hw),       # resize shortest side to resize_hw[0], maintaining aspect, then center‐crop
            T.CenterCrop(resize_hw),   # ensure final H×W
        ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        """
        Returns a dict with keys:
          - 'video_chunks':   [ tensor([1, 3, F, H, W]) ]
          - 'cond_chunks':    {'ref_mask': [ tensor([1, F, H, W]) ] }
          - 'chunk_is_ref':   [ tensor([True/False] * F) ]
          - 'raw_audio':      None
        """
        video_path = self.video_files[idx]

        # --------------------------------------------------------------------------------
        # 1. Read the video (all frames). This returns:
        #    video_frames: Tensor [N_frames_total, H_orig, W_orig, 3]
        #    audio:        Tensor [num_audio_samples, channels] (unused here, so we ignore it)
        #    info:         dict with metadata (fps, etc.)
        # --------------------------------------------------------------------------------
        if self.frame_rate is None:
            video_frames, _, info = read_video(video_path, pts_unit="sec")
        else:
            # read_video doesn't let you directly downsample FPS, so read all then pick frames:
            video_frames, _, info = read_video(video_path, pts_unit="sec")
            # info['video_fps'] is the original FPS; we take every round‐off of fps/self.frame_rate
            orig_fps = float(info.get("video_fps", 30.0))
            skip_rate = max(1, round(orig_fps / float(self.frame_rate)))
            video_frames = video_frames[::skip_rate]

        # video_frames: shape [N_frames_total, H_orig, W_orig, 3], dtype=torch.uint8 (0–255)
        N_total = video_frames.shape[0]
        F_req = self.total_frames

        if N_total < F_req:
            # If the video is shorter than required frames, pad by repeating last frame:
            pad_amt = F_req - N_total
            last_frame = video_frames[-1:].repeat(pad_amt, 1, 1, 1)  # [pad_amt, H, W, 3]
            video_frames = torch.cat([video_frames, last_frame], dim=0)
            N_total = video_frames.shape[0]

        # Randomly choose a starting index so that we have F_req consecutive frames
        start_idx = random.randint(0, N_total - F_req)
        clip = video_frames[start_idx : start_idx + F_req]  # [F_req, H_orig, W_orig, 3]

        # --------------------------------------------------------------------------------
        # 2. Resize & permute into shape [3, F_req, H, W]
        # --------------------------------------------------------------------------------
        # clip is uint8 in [0,255]. First convert to float in [0,1], then normalize (0–1).
        # torchvision transforms expect [C, H, W], so we do per‐frame transform in a loop:
        frames_list = []
        for f in range(F_req):
            frame = clip[f]                           # [H_orig, W_orig, 3], uint8
            frame = frame.permute(2, 0, 1).float() / 255.0   # [3, H_orig, W_orig], float
            frame = self.resize_transform(frame)            # [3, H, W], still float
            frames_list.append(frame)

        # Stack into shape [F_req, 3, H, W], then permute to [3, F_req, H, W]
        video_tensor = torch.stack(frames_list, dim=0).permute(1, 0, 2, 3)  # [3, F_req, H, W]

        # Add a dummy batch‐dim so it becomes [1, 3, F_req, H, W]:
        video_tensor = video_tensor.unsqueeze(0)

        # --------------------------------------------------------------------------------
        # 3. Build the “reference mask”: shape [1, F_req, H, W], float32 {0,1}.
        #    First num_ref frames are reference → mask=1; rest mask=0.
        # --------------------------------------------------------------------------------
        ref_mask = torch.zeros((F_req, *self.resize_hw, 3), dtype=torch.float32)
        ref_mask[: self.num_ref] = 1.0

        # Add batch‐dim → [1, F_req, H, W]
        ref_mask = ref_mask.unsqueeze(0)

        # --------------------------------------------------------------------------------
        # 4. Build chunk_is_ref: a length‐F_req boolean vector, True for reference frames
        # --------------------------------------------------------------------------------
        chunk_is_ref = torch.zeros(F_req, dtype=torch.bool)
        # chunk_is_ref[: self.num_ref] = True

        # --------------------------------------------------------------------------------
        # 5. Package into the format training loop expects:
        #    - video_chunks:   a list of length 1, containing [1,3,F_req,H,W]
        #    - cond_chunks:    {"ref_mask": [1, F_req, H, W]}
        #    - chunk_is_ref:   a list of length 1, containing a [F_req] boolean tensor
        #    - raw_audio:      None (we’re not using audio at all)
        # --------------------------------------------------------------------------------
        return {
            "video_chunks":       [video_tensor],          # list length=1
            "cond_chunks":        {"ref_mask": [ref_mask]}, # list length=1 inside dict
            "chunk_is_ref":       [chunk_is_ref],          # list length=1
            "raw_audio":          None,
            # If your training loop later tries to read batch["audio"] or batch["has_audio"],
            # you could put dummy entries like:
            # "audio": torch.zeros((1, some_length, some_dim)),
            # "has_audio": [False],
            # "sample_positions": torch.zeros((1, some_length)),
        }