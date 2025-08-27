import os
print(os.chdir(".."))

from model.flameObj import *

flamePath = "/scratch/ondemand28/harryscz/head_audio/head/code/flame/flame2023_no_jaw.npz"
sourcePath = "/scratch/ondemand28/harryscz/head_audio/head/data/vfhq-fit"
dataPath = [os.path.join(os.path.join(sourcePath, data), "fit.npz") for data in os.listdir(sourcePath)]

head = Flame(flamePath, device="cuda")
head.loadSequence(dataPath[0])
id_seq = head.LSB(rotation=False, identity=True)

head.loadSequence(dataPath[0])

gt = torch.load("/scratch/ondemand28/harryscz/diffusion/diffOut/gt.pt")
vae_gt = torch.load("/scratch/ondemand28/harryscz/diffusion/diffOut/vae_gt.pt")
recon = torch.load("/scratch/ondemand28/harryscz/diffusion/diffOut/overfit.pt")

gt
recon

f = min(gt.shape[0], recon.shape[0])

gt = head.sampleFromUV(gt)
gt = head.sampleTo3D(gt)
recon = head.sampleFromUV(recon)
recon = head.sampleTo3D(recon)

min_val = torch.tensor(0.0287, device="cuda:0")
max_val = torch.tensor(3.9643e-05, device="cuda:0")

diff = gt[:f, ...] - recon[:f, ...]

diff = (diff - min_val) / (max_val - min_val)

head.get_3d_animation(id_seq[:f], "test/testDiff3d.mp4", perVertsTexture=diff, dist=1.2, resolution=720)


diff = lambda x, i: (torch.min(x[..., i]), torch.max(x[..., i]), x.shape)

diff(gt, 0), diff(gt, 1), diff(gt, 2)

diff(vae_gt, 0), diff(vae_gt, 1), diff(vae_gt, 2)

diff(recon, 0), diff(recon, 1), diff(recon, 2)