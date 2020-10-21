import argparse
import os
import glob
import pprint
import pickle
import time

import torch
from torch.nn import functional as F
import torch.optim as optim
import pytorch3d.structures
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader
)
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.io import save_obj
from tqdm.autonotebook import tqdm
import pandas as pd

from utils import utils, network_utils
from deformation.deformation_net import DeformationNetwork
from deformation.deformation_net_extended import DeformationNetworkExtended
import deformation.losses as def_losses
from deformation.semantic_discriminator_loss import SemanticDiscriminatorLoss, compute_sem_dis_loss, compute_sem_dis_logits
from adversarial.datasets import GenerationDataset, ShapenetRendersDataset
from deformation.semantic_discriminator_net import SemanticDiscriminatorNetwork
from adversarial.datasets import gen_data_collate


class AdversarialDiscriminatorEval():

    def __init__(self, saved_model_path, deform_net_weights_path, semantic_dis_net_weights_path, output_dir_name,
                 num_mesh_eval, device):
        cfg_path = os.path.join(saved_model_path, glob.glob(os.path.join(saved_model_path, "*.yaml"))[0].split('/')[-1])
        self.cfg = utils.load_config(cfg_path, "configs/default.yaml")

        self.device = device
        self.saved_model_path = saved_model_path 
        self.eval_output_path = os.path.join(saved_model_path, output_dir_name)
        if not os.path.exists(self.eval_output_path):
            os.makedirs(self.eval_output_path)
        self.deform_net_weights_path = deform_net_weights_path
        self.semantic_dis_net_weights_path = semantic_dis_net_weights_path
        self.mesh_num_vertices = self.cfg["semantic_dis_training"]["mesh_num_verts"]
        self.tqdm_out = utils.TqdmPrintEvery()

        self.semantic_dis_loss_num_render = self.cfg["training"]["semantic_dis_num_render"]
        self.label_noise = 0
        self.real_labels_dist = torch.distributions.Uniform(torch.tensor([1.0-self.label_noise]), torch.tensor([1.0]))
        self.fake_labels_dist = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([0.0+self.label_noise]))

        self.num_mesh_eval = num_mesh_eval


    def eval(self):

        # setting up dataloaders
        generation_dataset = GenerationDataset(self.cfg, self.device)
        generation_loader = torch.utils.data.DataLoader(generation_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=gen_data_collate)
        if self.num_mesh_eval == -1: self.num_mesh_eval = len(generation_loader) + 1

        # setting up network
        deform_net = DeformationNetworkExtended(self.cfg, self.mesh_num_vertices, self.device)
        deform_net.load_state_dict(torch.load(self.deform_net_weights_path))
        deform_net.to(self.device)
        
        semantic_dis_net = SemanticDiscriminatorNetwork(self.cfg)
        semantic_dis_net.load_state_dict(torch.load(self.semantic_dis_net_weights_path))
        semantic_dis_net.to(self.device)

        # eval loop
        loss_info = {}
        #with tqdm(total=self.num_mesh_eval, file=self.tqdm_out, desc="Evaluating Meshes") as pbar:
        with tqdm(total=self.num_mesh_eval, desc="Evaluating Meshes") as pbar:
            for batch_idx, gen_batch in enumerate(generation_loader):
                if batch_idx >= self.num_mesh_eval: break

                deform_net.eval()
                semantic_dis_net.eval()

                curr_instance_name = gen_batch["instance_name"][0]
                gen_batch_meshes = gen_batch["mesh"].to(self.device)
                gen_batch_vertices = gen_batch["mesh_verts"].to(self.device)
                gen_batch_images = gen_batch["image"].to(self.device)
                gen_batch_poses = gen_batch["pose"].to(self.device)
                gen_batch_masks = gen_batch["mask"].to(self.device)

                loss_dict, deformed_mesh = self.refine_mesh_batched(deform_net, semantic_dis_net, gen_batch_meshes, gen_batch_vertices, 
                                                            gen_batch_images, gen_batch_poses, gen_batch_masks, compute_losses=True)

                save_obj(os.path.join(self.eval_output_path, "{}.obj".format(curr_instance_name)), deformed_mesh.verts_packed(), deformed_mesh.faces_packed())

                total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])
                curr_train_info = {"total_loss": total_loss.item()}
                curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
                loss_info[curr_instance_name] = curr_train_info
                pbar.update(1)

        pickle.dump(loss_info, open(os.path.join(self.eval_output_path, "eval_loss_info.p"),"wb"))


    # given a batch of meshes, masks, and poses computes a forward pass through a given deformation network and semantic discriminator network
    # returns the deformed mesh and a (optionally) dict of (unweighed, raw) computed losses
    # Note that semantic_dis_net and mask_batch is not necessary (can be None) when compute_losses=False, since it is only used to compute losses
    def refine_mesh_batched(self, deform_net, semantic_dis_net, mesh_batch, mesh_verts_batch, img_batch, pose_batch, mask_batch, compute_losses=True):

        # computing mesh deformation
        delta_v = deform_net(pose_batch, img_batch, mesh_verts_batch)
        batch_size = delta_v.shape[0]
        delta_v = delta_v.reshape((-1,3))
        #delta_v = delta_v.reshape((batch_size,-1,3))

        deformed_meshes = mesh_batch.offset_verts(delta_v)
        loss_dict = {}

        if compute_losses:
            pred_dist = pose_batch[:,0]
            pred_elev = pose_batch[:,1]
            pred_azim = pose_batch[:,2]
            R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim) 

            deformed_renders = utils.render_mesh(deformed_meshes, R, T, self.device, img_size=224, silhouette=True)
            deformed_silhouettes = deformed_renders[:, :, :, 3]
            # TODO: check range of deformed silhouettes
            loss_dict["sil_loss"] = F.binary_cross_entropy(deformed_silhouettes, mask_batch)

            num_vertices = deformed_meshes.verts_packed().shape[0]
            zero_deformation_tensor = torch.zeros((num_vertices, 3)).to(self.device)
            loss_dict["l2_loss"] = F.mse_loss(delta_v, zero_deformation_tensor)

            loss_dict["lap_smoothness_loss"] = mesh_laplacian_smoothing(deformed_meshes) # TODO: experiment with different variants (see pytorch3d docs)
            loss_dict["normal_consistency_loss"] = mesh_normal_consistency(deformed_meshes)

            sem_dis_logits, _ = compute_sem_dis_logits(deformed_meshes, semantic_dis_net, self.device, self.cfg)
            real_labels = self.real_labels_dist.sample((sem_dis_logits.shape[0],1)).squeeze(2).to(self.device)
            
            loss_dict["semantic_dis_loss"] = F.binary_cross_entropy_with_logits(sem_dis_logits, real_labels)

            sym_plane_normal = [0,0,1] # TODO: make this generalizable to other classes
            loss_dict["img_sym_loss"] = def_losses.image_symmetry_loss_batched(deformed_meshes, sym_plane_normal, self.cfg["training"]["img_sym_num_azim"], self.device)

            loss_dict["vertex_sym_loss"] = def_losses.vertex_symmetry_loss_batched(deformed_meshes, sym_plane_normal, self.device)

        return loss_dict, deformed_meshes
            

# returns the latest deform_net and semantic_dis_net weights
def get_latest_weights_paths(saved_model_dir):
    if os.path.exists(os.path.join(saved_model_dir, "deform_net_weights.pt")):
        return os.path.join(saved_model_dir, "deform_net_weights.pt"), os.path.join(saved_model_dir, "semantic_dis_net_weights.pt")
    
    weights_path = glob.glob(os.path.join(saved_model_dir, "deform_net_weights_*"))
    sorted(weights_path)
    return weights_path[0], weights_path[0].replace("deform_net_weights", "semantic_dis_net_weights")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarially train a SemanticDiscriminatorNetwork.')
    parser.add_argument('saved_model_path', type=str, help='Path to saved model folder.')
    #parser.add_argument('--saved_weights_path', type=str, default="", help = "path to .p weights to use for deformation network")
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--num_mesh', type=int, default=10, help='Number of meshes to evaluate.')
    parser.add_argument('--output_name', type=str, default="eval", help='Name to use for output evaluation folder')
    args = parser.parse_args()

    deform_net_weights_path, semantic_dis_net_weights_path = get_latest_weights_paths(args.saved_model_path)
    print("Using weights at:\n{}\n{}\n".format(deform_net_weights_path, semantic_dis_net_weights_path))
    device = torch.device("cuda:"+str(args.gpu))
    adv_eval = AdversarialDiscriminatorEval(args.saved_model_path, deform_net_weights_path, semantic_dis_net_weights_path, args.output_name, args.num_mesh, device)
    adv_eval.eval()


