import argparse
import os
import glob
import pprint
import pickle
import time
import shutil

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
from deformation.deformation_net_extended import DeformationNetworkExtended
import deformation.losses as def_losses
from deformation.semantic_discriminator_loss import SemanticDiscriminatorLoss, compute_sem_dis_loss, compute_sem_dis_logits
from adversarial.datasets import GenerationDataset, ShapenetRendersDataset
from deformation.semantic_discriminator_net import SemanticDiscriminatorNetwork
from adversarial.datasets import gen_data_collate


class AdversarialDiscriminatorTrainer():

    def __init__(self, cfg_path, gpu_num, exp_name, light):
        self.cfg = utils.load_config(cfg_path, "configs/default.yaml")
        self.device = torch.device("cuda:"+str(gpu_num))

        self.mesh_num_vertices = self.cfg["semantic_dis_training"]["mesh_num_verts"]
        self.batch_size = self.cfg["semantic_dis_training"]["batch_size"]
        self.semantic_dis_loss_num_render = self.cfg["training"]["semantic_dis_num_render"]

        self.total_training_iters = self.cfg["semantic_dis_training"]["adv_iterations"]
        self.num_batches_dis_train = self.cfg["semantic_dis_training"]["num_batches_dis_train"]
        self.num_batches_gen_train = self.cfg["semantic_dis_training"]["num_batches_gen_train"]
        self.save_model_every = self.cfg["semantic_dis_training"]["save_model_every"]

        self.training_output_dir = os.path.join(self.cfg['semantic_dis_training']['output_dir'], "{}_{}".format(time.strftime("%Y_%m_%d--%H_%M_%S"), exp_name))
        if not os.path.exists(self.training_output_dir):
            os.makedirs(self.training_output_dir)
        shutil.copyfile(cfg_path, os.path.join(self.training_output_dir, cfg_path.split("/")[-1]))

        # for adding noise to training labels. Real images have label 1, fake images has label 0 
        self.label_noise = self.cfg["semantic_dis_training"]["label_noise"]
        self.real_labels_dist = torch.distributions.Uniform(torch.tensor([1.0-self.label_noise]), torch.tensor([1.0]))
        self.fake_labels_dist = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([0.0+self.label_noise]))

        if light:
            self.num_workers = 4
        else:
            self.num_workers = 8
        self.tqdm_out = utils.TqdmPrintEvery()


    def train(self):

        # setting up dataloaders
        generation_dataset = GenerationDataset(self.cfg, self.device)
        #if self.num_batches_gen_train == -1:
        #    self.num_batches_gen_train = len(generation_loader) + 1
        generation_loader = torch.utils.data.DataLoader(generation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=gen_data_collate)
        
        # shapenet batch size is mesh batch size x number of renders for the semantic loss
        shapenet_renders_dataset = ShapenetRendersDataset(self.cfg)
        shapenet_renders_loader = torch.utils.data.DataLoader(shapenet_renders_dataset, batch_size=self.batch_size * self.semantic_dis_loss_num_render, num_workers=self.num_workers, shuffle=True)

        # setting up networks and optimizers
        # TODO: add option to continue training from previous weights
        deform_net = DeformationNetworkExtended(self.cfg, self.mesh_num_vertices, self.device)
        deform_net.to(self.device)
        deform_optimizer = optim.Adam(deform_net.parameters(), lr=self.cfg["training"]["learning_rate"])
        #deform_scheduler = optim.lr_scheduler.MultiStepLR(deform_optimizer, milestones=[400, 800], gamma=0.1) #TODO: experiment with plateau scheduler

        semantic_dis_net = SemanticDiscriminatorNetwork(self.cfg)
        semantic_dis_net.to(self.device)
        dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=0.00001, weight_decay=1e-2)

        # training generative deformation network and discriminator in an alternating, GAN style
        training_df = {"deform_net_gen": pd.DataFrame(), "semantic_dis": pd.DataFrame()}
        for iter_idx in tqdm(range(self.total_training_iters), file=self.tqdm_out, desc="Alternating MiniMax Iterations"):

            # training discriminator; generator weights are frozen
            ############################################################################################################################
            if self.num_batches_dis_train > 0:
                for param in semantic_dis_net.parameters(): param.requires_grad = True
                for param in deform_net.parameters(): param.requires_grad = False
                with tqdm(total=self.num_batches_dis_train, file=self.tqdm_out, desc="Iteration {} Dis Batches".format(iter_idx)) as pbar:
                    for batch_idx, (gen_batch, real_render_batch) in enumerate(zip(generation_loader, shapenet_renders_loader)):
                        if batch_idx >= self.num_batches_dis_train: break

                        semantic_dis_net.train()
                        deform_net.eval() # not sure if supposed to set this
                        dis_optimizer.zero_grad()

                        real_render_batch = real_render_batch.to(self.device)
                        pred_logits_real = semantic_dis_net(real_render_batch)

                        gen_batch_meshes = gen_batch["mesh"].to(self.device)
                        gen_batch_vertices = gen_batch["mesh_verts"].to(self.device)
                        gen_batch_images = gen_batch["image"].to(self.device)
                        gen_batch_poses = gen_batch["pose"].to(self.device)
                        _, deformed_meshes  = self.refine_mesh_batched(deform_net, semantic_dis_net, gen_batch_meshes, gen_batch_vertices, 
                                                                    gen_batch_images, gen_batch_poses, None, compute_losses=False)
                        pred_logits_fake, _ = compute_sem_dis_logits(deformed_meshes, self.semantic_dis_loss_num_render, semantic_dis_net, self.device)

                        batch_size = real_render_batch.shape[0]
                        real_labels = self.real_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)
                        fake_labels = self.fake_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)

                        dis_loss = F.binary_cross_entropy_with_logits(pred_logits_real, real_labels) + \
                            F.binary_cross_entropy_with_logits(pred_logits_fake, fake_labels)
                        
                        dis_loss.backward()
                        dis_optimizer.step()
                        curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "dis_loss": dis_loss.item()}
                        training_df["semantic_dis"] = training_df["semantic_dis"].append(curr_train_info, ignore_index = True)
                        pbar.update(1)
                    
                for param in semantic_dis_net.parameters(): param.requires_grad = False
                for param in deform_net.parameters(): param.requires_grad = True

            # training generator; discriminator weights are frozen
            ############################################################################################################################

            #with tqdm(total=self.num_batches_gen_train, file=self.tqdm_out, desc="Iteration {} Gen Batches".format(iter_idx)) as pbar:
            for batch_idx, gen_batch in enumerate(tqdm(generation_loader, file=self.tqdm_out, desc="Iteration {} Gen Batches".format(iter_idx))):
                #if batch_idx >= self.num_batches_gen_train: break

                gen_batch_meshes = gen_batch["mesh"].to(self.device)
                gen_batch_vertices = gen_batch["mesh_verts"].to(self.device)
                gen_batch_images = gen_batch["image"].to(self.device)
                gen_batch_poses = gen_batch["pose"].to(self.device)
                gen_batch_masks = gen_batch["mask"].to(self.device)

                deform_net.train()
                semantic_dis_net.eval()
                deform_optimizer.zero_grad()

                loss_dict, _ = self.refine_mesh_batched(deform_net, semantic_dis_net, gen_batch_meshes, gen_batch_vertices, 
                                                            gen_batch_images, gen_batch_poses, gen_batch_masks, compute_losses=True)
                total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])

                total_loss.backward()
                deform_optimizer.step()
                curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "total_loss": total_loss.item()}
                #curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "total_loss": total_loss.item(), "lr": deform_scheduler.get_last_lr().item()}
                curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
                training_df["deform_net_gen"] = training_df["deform_net_gen"].append(curr_train_info, ignore_index = True)
                #pbar.update(1)

            #deform_scheduler.step()

            # save model parameters and training info
            pickle.dump(training_df, open(os.path.join(self.training_output_dir, "training_df.p"),"wb"))
            if iter_idx != 0 and iter_idx % self.save_model_every == 0:
                torch.save(deform_net.state_dict(), os.path.join(self.training_output_dir, "deform_net_weights_{}.pt".format(iter_idx)))
                torch.save(semantic_dis_net.state_dict(), os.path.join(self.training_output_dir, "semantic_dis_net_weights_{}.pt".format(iter_idx)))

        torch.save(deform_net.state_dict(), os.path.join(self.training_output_dir, "deform_net_weights.pt"))
        torch.save(semantic_dis_net.state_dict(), os.path.join(self.training_output_dir, "semantic_dis_net_weights.pt"))
        pickle.dump(training_df, open(os.path.join(self.training_output_dir, "training_df.p"),"wb"))


    # given a batch of meshes, masks, and poses computes a forward pass through a given deformation network and semantic discriminator network
    # returns the deformed mesh and a (optionally) dict of (unweighed, raw) computed losses
    # Note that mask_batch is not necessary (can be None) when compute_losses=False, since it is only used to compute silhouette loss
    def refine_mesh_batched(self, deform_net, semantic_dis_net, mesh_batch, mesh_verts_batch, img_batch, pose_batch, mask_batch, compute_losses=True):

        # computing mesh deformation
        deformation_output = deform_net(pose_batch, img_batch, mesh_verts_batch)
        batch_size = deformation_output.shape[0]
        if self.cfg["model"]["output_delta_V"]:
            deformation_output = deformation_output.reshape((-1,3))
            deformed_meshes = mesh_batch.offset_verts(deformation_output)
        else:
            deformation_output = deformation_output.reshape((batch_size,-1,3))
            deformed_meshes = mesh_batch.update_padded(deformation_output)

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
            # TODO: fix this
            if self.cfg["model"]["output_delta_V"]:
                loss_dict["l2_loss"] = F.mse_loss(deformation_output, zero_deformation_tensor)
            else:
                loss_dict["l2_loss"] = torch.tensor(0).to(self.device)

            loss_dict["lap_smoothness_loss"] = mesh_laplacian_smoothing(deformed_meshes) # TODO: experiment with different variants (see pytorch3d docs)
            loss_dict["normal_consistency_loss"] = mesh_normal_consistency(deformed_meshes)

            if self.cfg["training"]["semantic_dis_lam"] > 0:
                sem_dis_logits, _ = compute_sem_dis_logits(deformed_meshes, self.semantic_dis_loss_num_render, semantic_dis_net, self.device)
                real_labels = self.real_labels_dist.sample((sem_dis_logits.shape[0],1)).squeeze(2).to(self.device)
                loss_dict["semantic_dis_loss"] = F.binary_cross_entropy_with_logits(sem_dis_logits, real_labels)
            else:
                loss_dict["semantic_dis_loss"] = torch.tensor(0).to(self.device)

            sym_plane_normal = [0,0,1] # TODO: make this generalizable to other classes
            if self.cfg["training"]["img_sym_lam"] > 0:
                loss_dict["img_sym_loss"] = def_losses.image_symmetry_loss_batched(deformed_meshes, sym_plane_normal, self.cfg["training"]["img_sym_num_azim"], self.device)
            else:
                loss_dict["img_sym_loss"] = torch.tensor(0).to(self.device)

            if self.cfg["training"]["vertex_sym_lam"] > 0:
                loss_dict["vertex_sym_loss"] = def_losses.vertex_symmetry_loss_batched(deformed_meshes, sym_plane_normal, self.device)
            else:
                loss_dict["vertex_sym_loss"] = torch.tensor(0).to(self.device)

        return loss_dict, deformed_meshes
            

# python adversarial_semantic_dis_trainer.py --exp_name test --light
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarially train a SemanticDiscriminatorNetwork.')
    parser.add_argument('--cfg_path', type=str, default="configs/default.yaml", help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--exp_name', type=str, default="adv_semantic_discrim", help='name of experiment')
    parser.add_argument('--light', action='store_true', help='run a lighter version of training w/ smaller batch size and num_workers')
    args = parser.parse_args()

    trainer = AdversarialDiscriminatorTrainer(args.cfg_path, args.gpu, args.exp_name, args.light)
    training_df = trainer.train()
