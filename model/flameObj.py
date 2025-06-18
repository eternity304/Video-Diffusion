import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import imageio

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    OrthographicCameras,
    TexturesVertex,
    SoftSilhouetteShader,
    PerspectiveCameras,
    BlendParams,
    HardFlatShader
)

import numpy as np

def dot(
    x: torch.Tensor,
    y: torch.Tensor,
    dim=-1,
    keepdim=False,
) -> torch.Tensor:
    return torch.sum(x * y, dim=dim, keepdim=keepdim)


def safe_length(
    x: torch.Tensor,
    dim=-1,
    keepdim=False,
    eps=1e-20,
) -> torch.Tensor:
    # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN
    return torch.sqrt(torch.clamp(dot(x, x, dim=dim, keepdim=keepdim), min=eps))

def batch_rodrigues(
    rot_vecs: torch.Tensor, epsilon=1e-8  # (B, 3)
) -> torch.Tensor:  # (B, 3, 3)
    """Calculates the rotation matrices for a batch of rotation vectors.
    Reference:
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
    Parameters
    ----------
    rot_vecs: torch.tensor Bx3
        array of B axis-angle vectors
    Returns
    -------
    R: torch.tensor Bx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = safe_length(rot_vecs, keepdim=True, eps=epsilon)  # (B, 1)
    rot_dir = rot_vecs / angle  # (B, 3)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)  # each (B, 1)

    zeros = torch.zeros(
        (batch_size, 1), dtype=torch.float32, device=device
    )  # (B, 1)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    )  # (B, 9)
    K = K.view(batch_size, 3, 3)  # (B, 3, 3)

    ident: torch.Tensor = torch.eye(
        3, dtype=torch.float32, device=device
    ).unsqueeze(
        dim=0
    )  # (1, 3, 3)
    cos = torch.unsqueeze(torch.cos(angle), dim=1)  # (B, 1, 1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)  # (B, 1, 1)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_vertices(
    transform: torch.Tensor, vertices: torch.Tensor
) -> torch.Tensor:
    """
    Transforms a list of vertices by a transform.

    Parameters
    ----------
    transform: torch.Tensor [B, 4, 4]
    vertices: torch.Tensor [B, N, 3]

    Returns
    -------
    transformed_vertices: torch.Tensor [B, N, 3]
    """
    transformed_verts = transform[:, :3, :3] @ vertices.permute(0, 2, 1)
    transformed_verts = transformed_verts + transform[:, :3, [3]]
    return transformed_verts.permute(0, 2, 1)

OPENCV2PYTORCH3D = torch.eye(4)
OPENCV2PYTORCH3D[1, 1] = -1
OPENCV2PYTORCH3D[2, 2] = -1

class Flame():
    def __init__(
        self, 
        flamePath : str,
        device : str = "cuda",
        n_shape_params: int = 150,
        n_expr_params: int = 65,
        uvPath :str =  "/scratch/ondemand28/harryscz/head_audio/head/head_template_mesh.obj"
    ):
        '''
        flamePath : str - path to head template object
        device : str - specify device to run model on
        '''

        obj = np.load(flamePath)
        self.device = device
        
        # Identity Faces
        self.verts = torch.tensor(obj['v_template']).to(self.device) # (N, 3)
        self.faces = torch.tensor(obj['f']).to(self.device)            # (numFaces, 3)

        # PCs on shape, expression, and pose
        self.shapeEigen = torch.tensor(obj['shapedirs'][..., :300]).to(self.device) # (N, 3, 300)
        # self.exprEigen = torch.tensor(obj['shapedirs'][..., 300:]).to(self.device)  # (N, 3, 100)
        self.poseEigen = torch.tensor(obj['posedirs']).to(self.device)              # (N, 3, 36) 

        self.exprEigen = torch.tensor(obj["shapedirs"][..., 300:300+400]).to(self.device)

        # Parameters for LSB
        self.skinningWt = torch.tensor(obj['weights']).to(self.device)           # (5023, 5)
        self.jointRegressor = torch.tensor(obj['J_regressor']).to(self.device)   # (5, 5023)
        self.joint_parents = torch.tensor(obj['kintree_table']).to(self.device) # (2, 5)    

        # bounds for 3d positions
        self.Rmax = 0.26367783546447754
        self.Rmin = -0.29412949085235596
        self.Gmax = 0.21029800176620483
        self.Gmin = -0.2735879421234131
        self.Bmax = -0.5083720088005066
        self.Bmin = -1.6089853048324585

        # UV info
        verts, faces, aux = load_obj(uvPath)

        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)

        self.facesUV, self.vertsUV = faces.textures_idx.to(self.device), aux.verts_uvs.to(self.device)
        self.faces3d, self.verts3d = faces_idx, verts
        self.nFaces, _ = self.faces3d.shape

        # initialize seq
        self.seq = None


    def loadSequence(
        self,
        sequencePath : str
    ):
        seq = np.load(sequencePath)
        
        # Blend shape parameter (1, 150), (nFrames, 65)
        self.beta, self.psi = torch.tensor(seq['shape']).to(self.device), torch.tensor(seq['expr']).to(self.device)
        self.nFrames = self.psi.shape[0]

        # Head Positions
        self.extr = torch.tensor(seq['extr']).to(self.device)
        self.rot, self.tra = torch.tensor(seq['rot']).to(self.device), torch.tensor(seq['tra']).to(self.device)

        # Camera parameters
        self.focalLength = torch.ones(self.nFrames, 2).to(self.device)
        self.principalPoint = torch.ones(self.nFrames, 2).to(self.device)

        self.focalLength[:, 0] = torch.tensor(seq['fx']).to(self.device).squeeze() 
        self.focalLength[:, 1] = torch.tensor(seq['fy']).to(self.device).squeeze()

        self.principalPoint[:, 0] = torch.tensor(seq['cx']).to(self.device).squeeze() 
        self.principalPoint[:, 1] = torch.tensor(seq['cy']).to(self.device).squeeze()

        self.extrRot = self.extr[:, :3, :3]
        self.extrTra = self.extr[:, :3, 3]

    def blendShape(self):
        # Get Number of basis used
        nPC = self.beta.shape[1]

        # Retrieve corresponding basis
        U = einops.rearrange(self.shapeEigen, "N xyz V ->  (N xyz) V")[:, :nPC].unsqueeze(0).expand(self.nFrames, 3*5023, nPC)

        # Expand beta
        self.beta = self.beta.unsqueeze(0).expand(self.nFrames, 1, nPC) 

        return torch.einsum('b i v, b n v -> b n i ', self.beta, U).squeeze(-1)

    def blendExpr(self):
        # Get number of basis
        nPC = self.psi.shape[1]

        assert self.psi.shape[1] == 65

        return einops.einsum(
            self.psi, 
            # self.shape_expr_eigenvecs[..., self.n_shape_params:self.n_shape_params+self.n_expr_params],
            self.exprEigen[:, :, :nPC],
            "b betas, V xyz betas -> b V xyz"
        )

        # Get matrix
        # U = einops.rearrange(self.shapeEigen, "N xyz V -> (N xyz) V")[:, :nPC].unsqueeze(0).expand(self.nFrames, 3*5023, nPC)

        # print(U.shape, self.exprEigen)

        # return torch.einsum('b n v, b v -> b n', U, self.psi)

    def identityRotation(
        self,
        perFrameVerts
    ):
        # initialize identity rotation for 5 joints across all frames
        rotation = torch.eye(3, device=self.device).unsqueeze(0).expand(5, 3, 3).unsqueeze(0).expand(self.nFrames, 5, 3, 3)

        _U = einops.rearrange(self.poseEigen, "N xyz v -> v (N xyz)")
        U = einops.rearrange(_U, "(J i j) (N xyz) -> J i j N xyz", i=3, j=3, xyz=3)

        _perFrameVerts = einops.rearrange(perFrameVerts, "b (N xyz) -> b N xyz", N = 5023)

        _jointRegressor = self.jointRegressor.unsqueeze(0).expand(self.nFrames, 5, 5023)
        joints = torch.einsum('b N c, b k N -> b k c', _perFrameVerts, _jointRegressor)

        rotatedJoints = (rotation @ joints.unsqueeze(-1)).squeeze(-1)

        # Modify the rotation to homogenous transformation
        transform = F.pad(rotation, [0,1,0,1,0,0,0,0]) # (B, 3N, 4, 4)
        transform[..., -1, -1] = 1
        transform[..., :3, -1] = joints - rotatedJoints # add translation
        weightedTransforms = torch.einsum('b K i j, N K -> b N i j', transform, self.skinningWt)
        vertsHomo = F.pad(_perFrameVerts, [0, 1], value = 1)
        vertsRotated = torch.einsum('b N i j, b N j -> b N i', weightedTransforms, vertsHomo)[..., :3] # (B, 3N, 3)
        return vertsRotated

    def LSB(self, rotation=True):
        '''
        This implementation use identity rotation
        '''

        # shapeOffset = self.blendShape()
        exprOffset = self.blendExpr()
        # perFrameVerts = self.verts.unsqueeze(0).expand(self.nFrames, 5023, 3).view(self.nFrames, -1) + shapeOffset + exprOffset
        perFrameVerts = self.verts.unsqueeze(0).expand(self.nFrames, 5023, 3).view(self.nFrames, -1) + exprOffset.view(self.nFrames, -1)
        rotatedPerFrameVerts = self.identityRotation(perFrameVerts)

        transform = torch.eye(4, device=self.device)[None]
        transform = transform.repeat(self.rot.shape[0], 1, 1)

        if rotation:
            transform[:, :3, :3] = batch_rodrigues(self.rot)
            transform[:, :3, 3] = self.tra

        perFrameVerts = transform_vertices(transform, rotatedPerFrameVerts)

        self.seq = perFrameVerts
        
 
        transform = self.extr @ OPENCV2PYTORCH3D[None].to(self.extr.device)
        verts = transform_vertices(transform, perFrameVerts)

        return perFrameVerts

    def renderAnimation(
        self,
        savePath,
        vertsTexture = False,
        customVerts=None,
        resolution=256,
        dist=0.2
    ):
        
        if customVerts is not None:
            perFrameVerts = customVerts
        elif self.seq is not None:
            perFrameVerts = self.seq
        else:
            perFrameVerts = self.LSB()

        self.nFrames, _, _ = perFrameVerts.shape

        perFrameFaces = self.faces.unsqueeze(0).expand(self.nFrames, self.faces.shape[0], 3)
        if not vertsTexture:
            perVertsTexture = torch.ones((self.nFrames, 5023, 3), dtype=torch.float32, device=self.device)
        else: 
            perVertsTexture = torch.ones((self.nFrames, 5023, 3), dtype=torch.float32, device=self.device)

        headMesh = Meshes(verts=perFrameVerts, faces=perFrameFaces)
        headMesh.textures = TexturesVertex(verts_features=perVertsTexture)

        R, T = look_at_view_transform(dist=dist, elev=0, azim=0)
        cameras = PerspectiveCameras(
            device=self.device, 
            focal_length=self.focalLength[0, :].unsqueeze(0).expand(self.nFrames, 2),
            principal_point=self.principalPoint[0, :].unsqueeze(0).expand(self.nFrames, 2),
            R=R.expand(self.nFrames, 3, 3), T=T.expand(self.nFrames, 3),
            in_ndc=False, image_size=((1096, 997),))

        raster_settings = RasterizationSettings(
            image_size=resolution,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        lights = PointLights(device=self.device, location=[[10.0, 10.0, 10.0]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=lights)
        )

        images = renderer(headMesh)
        frames = images.cpu().numpy()  # shape: (B, H, W, 3)

        # Convert from [0, 1] floats to [0, 255] uint8 if needed:
        frames = (frames * 255).astype(np.uint8)

        # Save as a GIF with 30 frames per second:
        imageio.mimsave(savePath, frames, fps=30, quality=10) 

    def convertUV(self, uvPath="/scratch/ondemand28/harryscz/head_audio/head/head_template_mesh.obj", customSeq=None):
        # Load UV faces information
        verts, faces, aux = load_obj(uvPath)

        faces_idx = faces.verts_idx.to(self.device)
        verts = verts.to(self.device)

        self.facesUV, self.vertsUV = faces.textures_idx.to(self.device), aux.verts_uvs.to(self.device)
        self.faces3d, self.verts3d = faces_idx, verts
        self.nFaces, _ = self.faces3d.shape
        
        #  Load and scale sequence
        seq = None
        if customSeq is not None: seq = customSeq
        elif self.seq is None: seq = self.LSB()
        else: seq = self.seq
        # seq = self.scaleIsotropic(seq)
        self.nFrames, _, _ = seq.shape
        gMin = torch.tensor([-0.1776, -0.2074, -1.0983], dtype=torch.float32).to(self.device)
        gMax = torch.tensor([ 0.1772,  0.1642, -0.6894], dtype=torch.float32).to(self.device)

        _seq = (seq - gMin) / (gMax - gMin)

        r, g, b = _seq[:,:,0], _seq[:,:,1], _seq[:,:,2]
        colours = torch.stack([r,g,b], dim=2)

        nColours = colours[:, self.faces3d.view(-1), :].view(self.nFrames, self.nFaces, 3, 3)
        nUVFeatures = torch.zeros(self.nFrames, self.vertsUV.shape[0], 3).to(self.device)
        nUVFeatures[:, self.facesUV.view(-1), ...] = nColours.view(self.nFrames, self.nFaces * 3, 3)

        vertsUV3d = torch.stack([self.vertsUV[:, 0], self.vertsUV[:, 1], torch.ones(self.vertsUV.shape[0]).to(self.device)], dim=1) * 2.025
        self.vertsUV3d = vertsUV3d.unsqueeze(0).expand(self.nFrames, -1, -1)
        self.facesUV3d = self.facesUV.unsqueeze(0).expand(self.nFrames, -1, -1)
        uvTexture = TexturesVertex(verts_features=nUVFeatures)

        return Meshes(verts=self.vertsUV3d, faces=self.facesUV3d, textures=uvTexture)

    def fillBlackPixels(self, images, max_iter=50):
        """
        images: a tensor of shape (B, C, H, W) where black pixels are assumed to be [0,0,0].
        max_iter: maximum number of iterations to avoid infinite loops.
        
        Returns a tensor with black pixels filled using nearest neighbor propagation.
        """
        # Create a binary mask where non-black pixels are 1
        non_black_mask = (images != 0).any(dim=1, keepdim=True).float()
        # Copy the original image to start filling
        result = images.clone()
        
        iter_count = 0
        # Loop until there are no black pixels or we hit the iteration limit
        while (non_black_mask == 0).sum() > 0 and iter_count < max_iter:
            # Use max pooling to dilate the image. It propagates non-zero (non-black) neighbors.
            dilated = F.max_pool2d(result, kernel_size=3, stride=1, padding=1)
            
            # Create a mask for black pixels that now have a non-black neighbor from dilation
            fill_mask = (non_black_mask == 0) & ((dilated != 0).any(dim=1, keepdim=True))
            
            # Only update the black pixels with the corresponding dilated values
            # Use expand_as(result) to align the mask with the image channels.
            result = torch.where(fill_mask.expand_as(result), dilated, result)
            
            # Update the mask based on the new values in result
            non_black_mask = (result != 0).any(dim=1, keepdim=True).float()
            iter_count += 1
            
        return result

    def renderUV(self, uvMesh, savePath=None, resolution=256, fill=True):
        uvBlendParams = BlendParams(background_color=(0.0, 0.0, 0.0))

        # UV renderer
        uvCameras = OrthographicCameras(
            device=self.device,
            R=torch.eye(3).unsqueeze(0),               # No rotation
            T=torch.tensor([[-1.009,-1.02,5]], device=self.device)   # Camera at (0,0,5)
        )

        uvLights = DirectionalLights(
            device=self.device,
            direction=torch.tensor([[0.2, 0, 1]], dtype=torch.float32, device=self.device),
            ambient_color=((0.5, 0.5, 0.5),),
            specular_color=((1.0, 1.0, 1.0),)
        )

        uvRasterSettings = RasterizationSettings(
            image_size=256,
            blur_radius=0.0,
            faces_per_pixel=1,
        )
        # Create a MeshRenderer using a MeshRasterizer and a Phong shader.
        uvRenderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=uvCameras,
                raster_settings=uvRasterSettings
            ),
            shader=HardFlatShader(
                device=self.device,
                cameras=uvCameras,
                blend_params=uvBlendParams
            )
        )
        
        perFrameUV = uvRenderer(uvMesh)
        
        if fill:
            smoothedPerFrameUV = self.fillBlackPixels(perFrameUV.permute(0,3,1,2)).permute(0,2,3,1)
        else:
            smoothedPerFrameUV = perFrameUV

        if savePath is not None:
            frames_rgb = (smoothedPerFrameUV[..., :3] * 255).byte().cpu().numpy()  
            imageio.mimsave(savePath, frames_rgb, fps=30, quality=10) 

        return smoothedPerFrameUV
    
    def sampleFromUV(self, perFrameUV, savePath=None, resolution=256):
        self.nFrames, W, H, C = perFrameUV.shape

        vertsUV3d = torch.stack([self.vertsUV[:, 0], self.vertsUV[:, 1], torch.ones(self.vertsUV.shape[0]).to(self.device)], dim=1) * 2.025
        self.vertsUV3d = vertsUV3d.unsqueeze(0).expand(self.nFrames, -1, -1) 

        self.facesUV3d = self.facesUV.unsqueeze(0).expand(self.nFrames, -1, -1)

        epsilon = 1e-6
        grid = self.vertsUV.clone()
        grid = grid.clamp(epsilon, 1.0 - epsilon)

        # x = 2*u - 1  maps [0,1]->[-1,+1]
        grid[..., 0] = 1.0 - 2.0 * grid[..., 0] + 0.0005
        # y = 1 - 2*v  flips the v axis, if needed
        grid[..., 1] = 1.0 - 2.0 * grid[..., 1] - 0.025

        # grid[...,0] =  grid[..., 0]*2 - 1     # not 1 - 2*u
        # grid[...,1] =  1 - grid[..., 1]*2     # if you need a v-flip


        grid = grid.unsqueeze(0).expand(self.nFrames, -1, -1)
        grid = grid.unsqueeze(2)

        perFrameUV = perFrameUV[..., :3].permute(0, 3, 1, 2)

        sampledUV = F.grid_sample(
            perFrameUV,
            grid,
            mode='nearest',
            align_corners=True,
            padding_mode='border'   
        )
  
        sampledUV = sampledUV.squeeze(-1).permute(0, 2, 1)[...,  :3] 
        
        sampledUVMesh = Meshes(verts=self.vertsUV3d, faces=self.facesUV3d, textures=TexturesVertex(verts_features=sampledUV))

        if savePath is not None: self.renderUV(sampledUVMesh, savePath=savePath, fill=False, resolution=resolution)
        
        return sampledUV

    def sampleTo3D(self, sampledUV, savePath=None, dist=0.2):
        self.nFrames, nVerts, nChannel = sampledUV.shape 

        _sampledUV = sampledUV.clone()

        # index all faces in UV space from sampled UV image in 1 array  -->  (num frames, num faces, 3 vertices, position of each vertices) i.e. the faces  with verts position in 3d space instead of index
        faces3dFromUV = _sampledUV[:, self.facesUV.view(-1),:].reshape(self.nFrames,  -1, 3)

        # Get sampled sequence from sampled features 
        sampledSeq = torch.zeros(self.nFrames, 5023, 3).to(self.device)
        sampledSeq[:, self.faces3d.view(-1),:] = faces3dFromUV[:, :, :]

        gMin = torch.tensor([-0.1776, -0.2074, -1.0983], dtype=torch.float32).to(self.device)
        gMax = torch.tensor([ 0.1772,  0.1642, -0.6894], dtype=torch.float32).to(self.device)

        sampledSeq = sampledSeq * (gMax - gMin) + gMin 
                
        if savePath is not None: self.renderAnimation(savePath, customVerts=sampledSeq * 2, dist=dist)

        return sampledSeq


