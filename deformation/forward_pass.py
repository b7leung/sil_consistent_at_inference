
import torch
from torch.nn import functional as F
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency

from utils import utils
import deformation.losses as def_losses


def compute_points_sem_dis_logits(meshes_batch, semantic_discriminator_net, device, cfg):
    logits = semantic_discriminator_net(meshes_batch.verts_padded())

    return logits, None


# computes semantic discriminator loss on a batch of meshes. Outputs [b,1] tensor, where b is batch size
def compute_render_sem_dis_logits(meshes_batch, semantic_discriminator_net, device, cfg):

    num_render = cfg["semantic_dis_training"]["semantic_dis_num_render"]
    random = cfg["semantic_dis_training"]["randomize_dis_inputs"]
    sil_dis_input = cfg["semantic_dis_training"]["sil_dis_input"]
    input_img_size = cfg["semantic_dis_training"]["dis_input_size"]
    num_meshes = len(meshes_batch)

    if random:
        d = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([360.0]))
        azims = d.sample((num_render*num_meshes,)).squeeze(-1)
    else:
        # 0.,  45.,  90., 135., 180., 225., 270., 315. 
        azims = torch.linspace(0, 360, num_render+1)[:-1].repeat(num_meshes)
    elevs = torch.Tensor([25 for i in range(num_meshes * num_render)])
    dists = torch.ones(num_meshes * num_render) * 1.7
    R, T = look_at_view_transform(dists, elevs, azims)

    extended_meshes = meshes_batch.extend(num_render)
    renders = utils.render_mesh(extended_meshes, R, T, device, img_size=input_img_size, silhouette=sil_dis_input)

    if sil_dis_input:
        # converting from [num_render, 224, 224, 4] silhouette render (only channel 4 has info) 
        # to [num_render, 224, 224, 3] rgb image (black/white)
        renders_rgb = torch.unsqueeze(renders[...,3], 3).repeat(1,1,1,3)
    else:
        renders_rgb = renders[...,:3]
    
    logits = semantic_discriminator_net(renders_rgb.permute(0,3,1,2).contiguous())

    return logits, renders_rgb


def compute_sem_dis_logits(meshes_batch, semantic_discriminator_net, device, cfg):
    if cfg['semantic_dis_training']['dis_type'] == "renders":
        return compute_render_sem_dis_logits(meshes_batch, semantic_discriminator_net, device, cfg)
    elif cfg['semantic_dis_training']['dis_type'] == "points":
        return compute_points_sem_dis_logits(meshes_batch, semantic_discriminator_net, device, cfg)
    else:
        raise ValueError("dis_type must be renders or pointnet")


# given an input batch (on the cpu) containing meshes, masks, and poses:
# computes a forward pass through a given deformation network and semantic discriminator network
# returns the deformed mesh and a (optionally) dict of (unweighed, raw) computed losses
# Note that mask_batch is not necessary (can be None) when compute_losses=False, since it is only used to compute silhouette loss
def batched_forward_pass(cfg, device, deform_net, semantic_dis_net, input_batch, compute_losses=True):
    # TODO: fix this
    real_labels_dist_gen = torch.distributions.Uniform(torch.tensor([1.0]), torch.tensor([1.0]))

    # deforming mesh
    # TODO: clean up double .to()
    deformation_output = deform_net(input_batch)
    mesh_batch = input_batch["mesh"].to(device)
    if cfg["model"]["output_delta_V"]:
        deformation_output = deformation_output.reshape((-1,3))
        deformed_meshes = mesh_batch.offset_verts(deformation_output)
    else:
        batch_size = deformation_output.shape[0]
        deformation_output = deformation_output.reshape((batch_size,-1,3))
        deformed_meshes = mesh_batch.update_padded(deformation_output)

    # computing network's losses
    loss_dict = {}
    if compute_losses:

        if cfg["training"]["sil_lam"] > 0:
            pose_batch = input_batch["pose"].to(device)
            pred_dist = pose_batch[:,0]
            pred_elev = pose_batch[:,1]
            pred_azim = pose_batch[:,2]
            R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim) 
            deformed_renders = utils.render_mesh(deformed_meshes, R, T, device, img_size=224, silhouette=True)
            deformed_silhouettes = deformed_renders[:, :, :, 3]
            mask_batch = input_batch["mask"].to(device)
            loss_dict["sil_loss"] = F.binary_cross_entropy(deformed_silhouettes, mask_batch)
        else:
            loss_dict["sil_loss"] = torch.tensor(0).to(device)

        if cfg["training"]["l2_lam"] > 0:
            num_vertices = deformed_meshes.verts_packed().shape[0]
            zero_deformation_tensor = torch.zeros((num_vertices, 3)).to(device)
            # TODO: fix this, or remove del v + v option entirely
            if cfg["model"]["output_delta_V"]:
                loss_dict["l2_loss"] = F.mse_loss(deformation_output, zero_deformation_tensor)
            else:
                loss_dict["l2_loss"] = torch.tensor(0).to(device)
        else:
            loss_dict["l2_loss"] = torch.tensor(0).to(device)

        if cfg["training"]["lap_smoothness_lam"] > 0:
            loss_dict["lap_smoothness_loss"] = mesh_laplacian_smoothing(deformed_meshes) # TODO: experiment with different variants (see pytorch3d docs)
        else:
            loss_dict["lap_smoothness_loss"] = torch.tensor(0).to(device)

        if cfg["training"]["normal_consistency_lam"] > 0:
            loss_dict["normal_consistency_loss"] = mesh_normal_consistency(deformed_meshes)
        else:
            loss_dict["normal_consistency_loss"] = torch.tensor(0).to(device)

        if cfg["training"]["semantic_dis_lam"] > 0:
            sem_dis_logits, _ = compute_sem_dis_logits(deformed_meshes, semantic_dis_net, device, cfg)
            real_labels = real_labels_dist_gen.sample((sem_dis_logits.shape[0],1)).squeeze(2).to(device)
            loss_dict["semantic_dis_loss"] = F.binary_cross_entropy_with_logits(sem_dis_logits, real_labels)
        else:
            loss_dict["semantic_dis_loss"] = torch.tensor(0).to(device)

        sym_plane_normal = [0,0,1] # TODO: make this generalizable to other classes
        if cfg["training"]["img_sym_lam"] > 0:
            loss_dict["img_sym_loss"] = def_losses.image_symmetry_loss_batched(deformed_meshes, sym_plane_normal, cfg["training"]["img_sym_num_azim"], device)
        else:
            loss_dict["img_sym_loss"] = torch.tensor(0).to(device)

        if cfg["training"]["vertex_sym_lam"] > 0:
            loss_dict["vertex_sym_loss"] = def_losses.vertex_symmetry_loss_batched(deformed_meshes, sym_plane_normal, device)
        else:
            loss_dict["vertex_sym_loss"] = torch.tensor(0).to(device)

    return loss_dict, deformed_meshes

