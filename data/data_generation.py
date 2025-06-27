import os
import sys

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, repo_root)

from model.flameObj import *

import os
from tqdm import tqdm

device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

flamePath = flamePath = "/scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz"
sourcePath = "/scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit"
dataPath = [os.path.join(os.path.join(sourcePath, data), "fit.npz") for data in os.listdir(sourcePath)]
seqPath = "/scratch/ondemand28/harryscz/head/_-91nXXjrVo_00/fit.npz"

head = Flame(flamePath, device=device)

def remove_item_in_place(arr, item):
    while item in arr:
        arr.remove(item)

bad_path = [
    "/scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit/5EuPFh6M1b4_00/fit.npz",
]

for item in bad_path:
    remove_item_in_place(dataPath, item)

for bad in bad_path:
    if not not(bad_path[0] in dataPath): raise ValueError(bad)

for path in tqdm(dataPath):
    try:
        save_path = f"/scratch/ondemand28/harryscz/head_audio/data/data256/uv/{path.split('/')[-2]}.mp4"
        if os.path.isfile(save_path): continue
        head.loadSequence(path)
        head.LSB()
        uvMesh = head.convertUV()
        head.get_uv_animation(uvMesh, savePath=save_path)
    except Exception as e:
        print(f"[ERROR] Failed on path: {path}\nReason: {e}")
        break