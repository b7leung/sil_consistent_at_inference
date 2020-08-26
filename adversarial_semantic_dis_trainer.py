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
from tqdm.autonotebook import tqdm
import pandas as pd

from utils import utils, network_utils
from deformation.deformation_net import DeformationNetwork
import deformation.losses as def_losses
from deformation.semantic_discriminator_loss import SemanticDiscriminatorLoss, compute_sem_dis_loss
from adversarial.datasets import GenerationDataset, ShapenetRendersDataset


class AdversarialDiscriminatorTrainer():

    def __init__(self, cfg_path, gpu_num, exp_name):
        self.cfg = utils.load_config(cfg_path, "configs/default.yaml")
        self.device = torch.device("cuda:"+str(gpu_num))

        self.batch_size = self.cfg["semantic_dis_training"]["batch_size"]
        self.total_training_iters = 2
        self.num_batches_dis_train = 5
        self.num_batches_gen_train = 5
        self.mesh_num_vertices = 1498
        self.label_noise = 0
        self.semantic_dis_loss_num_render = 8

        self.training_output_dir = os.path.join(cfg['semantic_dis_training']['output_dir'], "{}_{}".format(time.strftime("%Y_%m_%d--%H_%M_%S"), exp_name))
        if not os.path.exists(self.training_output_dir):
            os.makedirs(self.training_output_dir)
        self.tqdm_out = utils.TqdmPrintEvery()


    def train(self):

        # setting up dataloaders
        # https://stackoverflow.com/questions/51444059/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch
        generation_dataset = GenerationDataset(cfg, self.device)
        generation_loader = torch.utils.data.DataLoader(generation_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        shapenet_renders_dataset = ShapenetRendersDataset(cfg)
        shapenet_renders_loader = torch.utils.data.DataLoader(shapenet_renders_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)

        # setting up networks and optimizers
        deform_net = DeformationNetwork(self.cfg, self.mesh_num_vertices, self.device)
        deform_net.to(self.device)
        deform_optimizer = optim.Adam(deform_net.parameters(), lr=self.cfg["training"]["learning_rate"])

        semantic_dis_net = SemanticDiscriminatorNetwork(cfg)
        semantic_dis_net.to(self.device)
        dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=0.00001, weight_decay=1e-2)

        # for adding noise to training labels
        # real images have label 1, fake images has label 0 
        real_labels_dist = torch.distributions.Uniform(torch.tensor([1.0-self.label_noise]), torch.tensor([1.0]))
        fake_labels_dist = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([0.0+self.label_noise]))

        # training generative deformation network and discriminator in an alternating, GAN style
        for iter_i in tqdm(range(self.total_training_iters), file=self.tqdm_out):

            # training discriminator; generator weights are frozen
            # =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
            for param in semantic_dis_net.parameters(): param.requires_grad = True
            for param in deform_net.parameters(): param.requires_grad = False

            generation_iter = iter(generation_loader)
            shapenet_renders_iter = iter(shapenet_renders_loader)
            for batch_idx in tqdm(range(self.num_batches_dis_train), file=self.tqdm_out):

                semantic_dis_net.train()
                deform_net.eval() # not sure if supposed to set this
                dis_optimizer.zero_grad()

                real_render_batch = next(shapenet_renders_iter).to(self.device)
                pred_logits_real = semantic_dis_net(real_render_batch)

                gen_batch = next(generation_iter)
                gen_batch_vertices = gen_batch["mesh_verts"].to(self.device)
                gen_batch_images = gen_batch["image"].to(self.device)
                gen_batch_poses = gen_batch["pose"].to(self.device)
                deformed_meshes  = self.refine_mesh_batched(deform_net, semantic_dis_net, gen_batch_vertices, 
                                                            gen_batch_images, gen_batch_poses, compute_losses=False)
                # TODO: fix this to turn into logits, not sigmoid
                pred_logits_fake = compute_sem_dis_loss(deformed_meshes, self.semantic_dis_loss_num_render, semantic_dis_net, self.device)

                batch_size = real_render_batch.shape[0]
                real_labels = real_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)
                fake_labels = fake_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)

                dis_loss = F.binary_cross_entropy_with_logits(pred_logits_real, real_labels) + \
                    F.binary_cross_entropy_with_logits(pred_logits_fake, fake_labels)
                
                dis_loss.backward()
                dis_optimizer.step()
            
            continue
            # training generator; discriminator weights are frozen
            # =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/
            for param in semantic_dis_net.parameters(): param.requires_grad = False
            for param in deform_net.parameters(): param.requires_grad = True
            for gen_batch in tqdm(generation_loader[:self.num_batches_gen_train], file=self.tqdm_out):
                deform_net.train()
                semantic_dis_net.eval()
                deform_optimizer.zero_grad()
                deform_loss_dict, _ = self.refine_mesh_batched(deform_net, semantic_dis_net, gen_batch)
                # TODO: make sure loss is correct (follows minimax loss)
                total_loss = sum([deform_loss_dict[loss_name] * cfg['training'][loss_name.replace("loss", "lam")] for loss_name in deform_loss_dict])
                total_loss.backward()
                deform_optimizer.step()


    # given a batch of meshes, masks, and poses computes a forward pass through a given deformation network and semantic discriminator network
    # returns the deformed mesh and a (optionally) dict of (unweighed, raw) computed losses
    # TODO: fix mesh (currently, needs to already be in device)
    def refine_mesh_batched(self, deform_net, semantic_dis_net, mesh_verts_batch, img_batch, pose_batch, compute_losses=True):

        # computing mesh deformation
        delta_v = deform_net(pose_batch, img_batch, mesh_verts_batch)
        delta_v = delta_v.reshape((-1,3))
        deformed_mesh = mesh.offset_verts(delta_v)

        if not compute_losses:
            return deformed_mesh

        else:
            # prep inputs used to compute losses
            pred_dist = pose_batch[:,0]
            pred_elev = pose_batch[:,1]
            pred_azim = pose_batch[:,2]
            R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim) 
            mask = rgba_image[:,:,3] > 0
            mask_gt = torch.tensor(mask, dtype=torch.float).to(self.device)
            num_vertices = mesh.verts_packed().shape[0]
            zero_deformation_tensor = torch.zeros((num_vertices, 3)).to(self.device)
            sym_plane_normal = [0,0,1] # TODO: make this generalizable to other classes

            loss_dict = {}
            # computing losses
            rendered_deformed_mesh = utils.render_mesh(deformed_mesh, R, T, self.device, img_size=224, silhouette=True)
            loss_dict["sil_loss"] = F.binary_cross_entropy(rendered_deformed_mesh[0, :,:, 3], mask_gt)
            loss_dict["l2_loss"] = F.mse_loss(delta_v, zero_deformation_tensor)
            loss_dict["lap_smoothness_loss"] = mesh_laplacian_smoothing(deformed_mesh)
            loss_dict["normal_consistency_loss"] = mesh_normal_consistency(deformed_mesh)

            # TODO: remove weights?
            if self.img_sym_lam > 0:
                loss_dict["img_sym_loss"], _ = def_losses.image_symmetry_loss(deformed_mesh, sym_plane_normal, self.cfg["training"]["img_sym_num_azim"], self.device)
            else:
                loss_dict["img_sym_loss"] = torch.tensor(0).to(self.device)
            if self.vertex_sym_lam > 0:
                loss_dict["vertex_sym_loss"] = def_losses.vertex_symmetry_loss_fast(deformed_mesh, sym_plane_normal, self.device)
            else:
                loss_dict["vertex_sym_loss"] = torch.tensor(0).to(self.device)
            if self.semantic_dis_lam > 0:
                loss_dict["semantic_dis_loss"], _ = compute_sem_dis_loss(deformed_mesh, self.semantic_dis_loss_num_render, semantic_dis_net, self.device)
            else:
                loss_dict["semantic_dis_loss"] = torch.tensor(0).to(self.device)

            return loss_dict, deformed_mesh
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarially train a SemanticDiscriminatorNetwork.')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--exp_name', type=str, default="adv_semantic_discrim", help='name of experiment')
    parser.add_argument('--light', action='store_true', help='run a lighter version of training w/ smaller batch size and num_workers')
    parser.add_argument('--label_noise', type=float, default=0, help='amount of label noise to use during training')
    args = parser.parse_args()

    trainer = AdversarialDiscriminatorTrainer(args.cfg_path, args.gpu, args.exp_name)
    training_df = trainer.train()


