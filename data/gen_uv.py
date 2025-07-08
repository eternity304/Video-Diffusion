from tqdm import tqdm
import os

from model.flameObj import *


def main():
    flamePath = flamePath = "/scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz"
    sourcePath = "/scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit"
    dataPath = [os.path.join(os.path.join(sourcePath, data), "fit.npz") for data in os.listdir(sourcePath)]
    seqPath = "/scratch/ondemand28/harryscz/head/_-91nXXjrVo_00/fit.npz"

    head = Flame(flamePath, device="cuda")

    for i in tqdm(range(len(dataPath))):
        head.loadSequence(dataPath[i])                                       
        perFrameVerts = head.LSB()                                           
        uvMesh = head.convertUV()     
        uv = head.get_uv_animation(uvMesh, resolution=512)
        print(i, uv.shape)                                       

if __name__ == "__main__":
    main()