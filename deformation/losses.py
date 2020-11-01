import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from torch.nn import functional as F
import pytorch3d
from pytorch3d.renderer import look_at_view_transform, TexturesVertex, Textures, PointLights
from torchvision import transforms
import pickle

from utils import general_utils

# given a mesh and a plane of symmetry (represented by its unit normal)
# computes the symmetry loss of the mesh, defined as the average
# distance between points on one side and its nearest neighbor point on the other side
# args:
#   Sym_plane: list of 3 numbers
# https://math.stackexchange.com/questions/693414/reflection-across-the-plane
def vertex_symmetry_loss(mesh, sym_plane, device, asym_conf_scores=False):
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")


    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float).to(device)

    mesh_verts = mesh.verts_packed()
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(mesh_verts.detach().cpu())

    sym_points = mesh_verts @ reflect_matrix
    distances, indices = nbrs.kneighbors(sym_points.detach().cpu())
    #avg_sym_loss = F.mse_loss(sym_points, torch.squeeze(mesh_verts[indices],1), reduction='sum') # not used
    #avg_sym_loss = F.l1_loss(sym_points, torch.squeeze(mesh_verts[indices],1), reduction='sum')
    nn_dists = torch.unsqueeze(torch.sum(F.l1_loss(sym_points, torch.squeeze(mesh_verts[indices],1), reduction='none'), 1),1)

    sym_bias = 0.005 #0.01
    if asym_conf_scores is not None:
        #avg_sym_loss = torch.mean(0.5*(torch.log(asym_conf_scores**2)) + torch.div(1,2*(asym_conf_scores**2))*nn_dists)
        avg_sym_loss = torch.mean(-torch.log(asym_conf_scores)*sym_bias + (asym_conf_scores*nn_dists))
    else:
        avg_sym_loss = torch.mean(nn_dists)
    return avg_sym_loss


# naive batched version of vertex symmetry loss
# assumes all meshes use the same sym_plane
def vertex_symmetry_loss_batched(meshes, sym_plane, device, asym_conf_scores=None):
    total_vtx_sym_loss = 0
    for mesh in meshes:
        curr_sym_loss = vertex_symmetry_loss(mesh, sym_plane, device, asym_conf_scores)
        total_vtx_sym_loss += curr_sym_loss
    avg_vtx_sym_loss = total_vtx_sym_loss / len(meshes)
    return avg_vtx_sym_loss


# image based symmetry loss
# renders mesh at offsets about the plane of symmetry and computes a MSE loss in pixel space
# silhouette should normally be true; only false for debug purposes
def image_symmetry_loss(mesh, sym_plane, num_azim, device, asym_conf_scores=None, render_silhouettes=True, dist=1.9):
    N = np.array([sym_plane])
    if np.linalg.norm(N) != 1:
        raise ValueError("sym_plane needs to be a unit normal")

    # camera positions for one half of the sphere. Offset allows for better angle even when num_azim = 1
    num_views_on_half = num_azim * 2
    offset = 15
    #azims = torch.linspace(0,90,num_azim+2)[1:-1].repeat(2)
    azims = torch.linspace(0+offset,90-offset,num_azim).repeat(2)
    elevs = torch.Tensor([-45 for i in range(num_azim)] + [45 for i in range(num_azim)] )
    dists = torch.ones(num_views_on_half) * dist
    R_half_1, T_half_1 = look_at_view_transform(dists, elevs, azims)
    R = [R_half_1]

    # compute the other half of camera positions according to plane of symmetry
    reflect_matrix = torch.tensor(np.eye(3) - 2*N.T@N, dtype=torch.float)
    for i in range(num_views_on_half):
        camera_position = pytorch3d.renderer.cameras.camera_position_from_spherical_angles(dists[i], elevs[i], azims[i])
        R_sym = pytorch3d.renderer.cameras.look_at_rotation(camera_position@reflect_matrix)
        R.append(R_sym)
    R = torch.cat(R)
    T = torch.cat([T_half_1, T_half_1])

    # rendering
    meshes = mesh.extend(num_views_on_half*2)
    if render_silhouettes:
        renders = general_utils.render_mesh(meshes, R, T, device, img_size=224, silhouette=True)[...,3]
    else:
        renders = general_utils.render_mesh(meshes, R, T, device, img_size=224, silhouette=False)

    # rendering sym conf. map
    if asym_conf_scores is not None:
        no_lights = PointLights(device=device, ambient_color=((0.5,0.5,0.5),), diffuse_color=((0.0,0.0,0.0),), specular_color=((0.0,0.0,0.0),))
        meshes_asym_conf_scores = asym_conf_scores.unsqueeze(0).repeat(num_views_on_half*2,1,3)
        meshes.textures = TexturesVertex(verts_features=meshes_asym_conf_scores)
        sym_conf_maps = general_utils.render_mesh(meshes[num_views_on_half:], R[num_views_on_half:], T[num_views_on_half:], device, img_size=224, silhouette=False, custom_lights=no_lights)

    # a sym_triple is [R1, R1_flipped, R2, vertex texture map using asym conf at R2]
    sym_triples = []
    for i in range(num_views_on_half):
        if asym_conf_scores is None:
            sym_triples.append([renders[i], torch.flip(renders[i], [1]), renders[i+num_views_on_half]])
        else:
            # need to turn white background black
            #sym_conf_map = sym_conf_maps[i][...,0] - (sym_conf_maps[i][...,3]==0).type(torch.float32)
            sym_conf_map = sym_conf_maps[i][...,0]
            sym_triples.append([renders[i], torch.flip(renders[i], [1]), renders[i+num_views_on_half], sym_conf_map])

    # calculating loss. 
    sym_loss = 0
    sym_bias = 1
    for sym_triple in sym_triples:
        if asym_conf_scores is None:
            sym_loss += F.mse_loss(sym_triple[1], sym_triple[2])
        else:
            #sym_loss += F.mse_loss(sym_triple[1], sym_triple[2])
            #print(sym_triple[3])
            #print(torch.mean((-torch.log(sym_triple[3])*sym_bias) + (F.mse_loss(sym_triple[1], sym_triple[2], reduction="none") * sym_triple[3])))

            sym_loss += torch.mean((-torch.log(sym_triple[3])*sym_bias) + (F.mse_loss(sym_triple[1], sym_triple[2], reduction="none") * sym_triple[3]))
    sym_loss = sym_loss / num_views_on_half

    return sym_loss, sym_triples
    

# naive batched version of image symmetry loss; averages image symmetry loss for each mesh in batch
# assumes all meshes use the same sym_plane
def image_symmetry_loss_batched(meshes, sym_plane, num_azim, device, asym_conf_scores=None, render_silhouettes=True):
    total_img_sym_loss = 0
    for mesh in meshes:
        curr_sym_loss, _ = image_symmetry_loss(mesh, sym_plane, num_azim, device, asym_conf_scores, render_silhouettes)
        total_img_sym_loss += curr_sym_loss
    avg_img_sym_loss = total_img_sym_loss / len(meshes)
    return avg_img_sym_loss

