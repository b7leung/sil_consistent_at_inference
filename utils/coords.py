import copy
import math

import numpy as np
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform

from utils import general_utils

# from https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/utils/coords.py


def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
    origin.
    Inputs:
    - azimuth: Rotation about the z-axis, in degrees
    - elevation: Rotation above the xy-plane, in degrees
    - distance: Distance from the origin
    Returns:
    - FloatTensor of shape (4, 4)
    """
    azimuth, elevation, distance = (float(azimuth), float(elevation), float(distance))
    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor([[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]])
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -R_obj2cam.mm(cam_location)
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # For some reason I cannot fathom, when Blender loads a .obj file it rotates
    # the model 90 degrees about the x axis. To compensate for this quirk we roll
    # that rotation into the extrinsic matrix here
    #rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    #RT = RT.mm(rot.to(RT))

    return RT


def rotate_verts(RT, verts):
    """
    Inputs:
    - RT: (N, 4, 4) array of extrinsic matrices
    - verts: (N, V, 3) array of vertex positions
    """
    singleton = False
    if RT.dim() == 2:
        assert verts.dim() == 2
        RT, verts = RT[None], verts[None]
        singleton = True

    if isinstance(verts, list):
        verts_rot = []
        for i, v in enumerate(verts):
            verts_rot.append(rotate_verts(RT[i], v))
        return verts_rot

    R = RT[:, :3, :3]
    verts_rot = verts.bmm(R.transpose(1, 2))
    if singleton:
        verts_rot = verts_rot[0]
    return verts_rot



def project_verts(verts, P, eps=1e-1):
    """
    Project verticies using a 4x4 transformation matrix
    Inputs:
    - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions.
    - P: FloatTensor of shape (N, 4, 4) giving projection matrices
    Outputs:
    - verts_out: FloatTensor of shape (N, V, 3) giving vertex positions (x, y, z)
        where verts_out[i] is the result of transforming verts[i] by P[i].
    """
    # Handle unbatched inputs
    singleton = False
    if verts.dim() == 2:
        assert P.dim() == 2
        singleton = True
        verts, P = verts[None], P[None]

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device

    # Add an extra row of ones to the world-space coordinates of verts before
    # multiplying by the projection matrix. We could avoid this allocation by
    # instead multiplying by a 4x3 submatrix of the projectio matrix, then
    # adding the remaining 4x1 vector. Not sure whether there will be much
    # performance difference between the two.
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)
    verts_hom = torch.cat([verts, ones], dim=2)
    verts_cam_hom = torch.bmm(verts_hom, P.transpose(1, 2))

    # Avoid division by zero by clamping the absolute value
    w = verts_cam_hom[:, :, 3:]
    w_sign = w.sign()
    w_sign[w == 0] = 1
    w = w_sign * w.abs().clamp(min=eps)

    verts_proj = verts_cam_hom[:, :, :3] / w

    if singleton:
        return verts_proj[0]
    return verts_proj


def reflect_batch(batch, sym_plane, device):
    N = np.array([sym_plane])
    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float).to(device)
    for i in range(batch.shape[0]):
        batch[i] = batch[i] @ reflect_matrix
    return batch


# aligning and normalizing a batch of vertex positions so (-1,-1) is the top left, (1,1) is the bottom right relative to the feature m
# Inputs: 
# - verts: FloatTensor of shape (N, V, 3) giving a batch of vertex positions
# - poses:  a b x 3 tensor specifying distance, elevation, azimuth (in that order)
# Outputs:
# - aligned_verts: Tensor of shape (N,V,3) giving a batch of aligned vertex positions
def align_and_normalize_verts_1(verts, poses, device):

    distances = poses[:,0]
    elevations = poses[:,1]
    azimuths = poses[:,2]

    # creating batch of 4x4 extrinsic matrices from rotation matrices and translation vectors
    temp = torch.tensor([0,0,0,1])
    temp = temp.repeat(poses.shape[0],1).unsqueeze(1)
    R, T = look_at_view_transform(distances, elevations, azimuths)
    T = T.unsqueeze(2)
    P = torch.cat([R,T], 2)
    P = torch.cat([P,temp], 1)

    # changing vertices from world coordinates to camera coordinates
    P_inv = torch.inverse(P).to(device)
    aligned_verts = rotate_verts(P_inv, verts)

    # TODO: for some reason, x2 seems to be a good way to normalize verts to roughly be in (-1,-1), (1,1). not sure if this is always the case
    aligned_verts = aligned_verts * 2

    # TODO: not sure if this should always be applied
    aligned_verts = reflect_batch(aligned_verts, [1,0,0], device)
    aligned_verts = reflect_batch(aligned_verts, [0,1,0], device)

    return aligned_verts


def rotate_verts_2(RT, verts):
    """
    Inputs:
    - RT: (N, 4, 4) array of extrinsic matrices
    - verts: (N, V, 3) array of vertex positions
    """
    singleton = False
    if RT.dim() == 2:
        assert verts.dim() == 2
        RT, verts = RT[None], verts[None]
        singleton = True

    if isinstance(verts, list):
        verts_rot = []
        for i, v in enumerate(verts):
            verts_rot.append(rotate_verts(RT[i], v))
        return verts_rot

    #R = RT[:, :3, :3]
    #verts_rot = verts.bmm(R.transpose(1, 2))

    N, V = verts.shape[0], verts.shape[1]
    dtype, device = verts.dtype, verts.device
    ones = torch.ones(N, V, 1, dtype=dtype, device=device)

    verts_hom = torch.cat([verts, ones], dim=2)
    verts_rot = torch.bmm(verts_hom, RT)
    #verts_rot = torch.bmm(verts_hom, RT.transpose(1, 2))
    verts_rot = verts_rot[:,:,:3]

    if singleton:
        verts_rot = verts_rot[0]
    return verts_rot


def compute_extrinsic_matrix(azimuth, elevation, distance):
    """
    Compute 4x4 extrinsic matrix that converts from homogenous world coordinates
    to homogenous camera coordinates. We assume that the camera is looking at the
    origin.
    Inputs:
    - azimuth: Rotation about the z-axis, in degrees
    - elevation: Rotation above the xy-plane, in degrees
    - distance: Distance from the origin
    Returns:
    - FloatTensor of shape (4, 4)
    """
    azimuth, elevation, distance = (float(azimuth), float(elevation), float(distance))
    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor([[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]])
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -R_obj2cam.mm(cam_location)
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # For some reason I cannot fathom, when Blender loads a .obj file it rotates
    # the model 90 degrees about the x axis. To compensate for this quirk we roll
    # that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RT = RT.mm(rot.to(RT))

    return RT

def align_and_normalize_verts_2(verts, poses, device):

    distances = poses[:,0] *1
    elevations = poses[:,1]
    azimuths = poses[:,2]

    # creating batch of 4x4 extrinsic matrices from rotation matrices and translation vectors
    temp = torch.tensor([0,0,0,1])
    temp = temp.repeat(poses.shape[0],1).unsqueeze(1)
    R, T = look_at_view_transform(distances, elevations, azimuths)
    print(T)
    T = T.unsqueeze(2)
    P = torch.cat([R,T], 2)
    P = torch.cat([P,temp], 1).double()
    P_inv = torch.inverse(P).to(device)

    #P = compute_extrinsic_matrix(azimuths[0], elevations[0],distances[0]).unsqueeze(0)
    #P_inv = P.to(device)


    # changing vertices from world coordinates to camera coordinates
    aligned_verts = rotate_verts_2(P_inv, verts)
    #aligned_verts = project_verts(verts, P_inv)

    # TODO: for some reason, x2 seems to be a good way to normalize verts to roughly be in (-1,-1), (1,1). not sure if this is always the case
    aligned_verts = aligned_verts*2

    # TODO: not sure if this should always be applied
    aligned_verts = reflect_batch(aligned_verts, [1,0,0], device)
    aligned_verts = reflect_batch(aligned_verts, [0,1,0], device)

    return aligned_verts







# aligning and normalizing a batch of meshes so vertex positions are such that (-1,-1) is the top left, (1,1) is the bottom right relative to the feature map
# Inputs: 
# - meshes: a batch of b meshes
# - poses:  a b x 3 tensor specifying distance, elevation, azimuth (in that order)
# - TODO: not sure about image_size param
# Outputs:
# - aligned_verts: Tensor of shape (N,V,3) giving a batch of aligned vertex positions
def get_aligned_verts(meshes, poses, device, img_size=224):

    distances = poses[:,0]
    elevations = poses[:,1]
    azimuths = poses[:,2]

    batch_size = poses.shape[0]
    aligned_verts = []

    # TODO: with torch.no_grad ?
    for i in range(batch_size):
        R, T = look_at_view_transform(distances[i], elevations[i], azimuths[i])    
        renderer = general_utils.render_mesh(None, R, T, device, img_size=img_size, return_renderer_only=True)
        mesh_on_screen = renderer.rasterizer.transform(meshes[i])
        pc = mesh_on_screen.verts_padded()[0]
        # TODO: this transformation necessary for shapenet. Not sure about other datasets
        pc = pc * torch.tensor([-1,-1,0], device=device)
        aligned_verts.append(pc.unsqueeze(0))
    aligned_verts = torch.cat(aligned_verts, axis=0)

    return aligned_verts


