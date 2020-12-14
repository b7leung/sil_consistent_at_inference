import argparse
import os
import pickle
import time
import shutil
import pprint

import torch
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms
from pytorch3d.io import save_obj
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np

from deformation.deformation_net import DeformationNetwork
from deformation.deformation_net_graph_convolutional_full import DeformationNetworkGraphConvolutionalFull

from utils import general_utils
from utils.gradient_penalty import GradientPenalty
from utils.datasets import GenerationDataset, RealDataset, BenchmarkDataset
#from deformation.semantic_discriminator_net_renders import RendersSemanticDiscriminatorNetwork
from deformation.semantic_discriminator_net_points import PointsSemanticDiscriminatorNetwork
from deformation.multiview_semantic_discriminator_network import MultiviewSemanticDiscriminatorNetwork
from deformation.deformation_net_fc_vertex_aligned import DeformationNetworkFcVertexAligned
from deformation.deformation_net_gcn_hybrid import DeformationNetworkGcnHybrid
from utils.datasets import gen_data_collate
from deformation.forward_pass import batched_forward_pass, compute_sem_dis_logits


class AdversarialDiscriminatorTrainer():

    def __init__(self, cfg_path, gpu_num, exp_name, num_workers):
        self.num_workers = num_workers
        self.device = torch.device("cuda:"+str(gpu_num))
        self.cfg = general_utils.load_config(cfg_path, "configs/default.yaml")
        self.batch_size = self.cfg["semantic_dis_training"]["batch_size"]
        self.dis_type = self.cfg['semantic_dis_training']['dis_type']
        self.deform_net_type = self.cfg["semantic_dis_training"]["deform_net_type"]
        self.dis_weight_path = self.cfg["semantic_dis_training"]["dis_weight_path"]
        self.gen_weight_path = self.cfg["semantic_dis_training"]["gen_weight_path"]
        self.total_training_iters = self.cfg["semantic_dis_training"]["adv_iterations"]
        self.dis_steps_per_iteration = self.cfg["semantic_dis_training"]["dis_steps_per_iteration"]
        self.gen_steps_per_iteration = self.cfg["semantic_dis_training"]["gen_steps_per_iteration"]
        self.save_model_every = self.cfg["semantic_dis_training"]["save_model_every"]
        self.save_samples_every = self.cfg["semantic_dis_training"]["save_samples_every"]
        self.beta1 = self.cfg["semantic_dis_training"]["beta1"]

        # creating output dir and recursively copy inherited configs
        self.training_output_dir = os.path.join(self.cfg['semantic_dis_training']['output_dir'], "{}_{}".format(time.strftime("%Y_%m_%d--%H_%M_%S"), exp_name))
        if not os.path.exists(self.training_output_dir): os.makedirs(self.training_output_dir)
        all_inherited_cfg_paths = general_utils.get_all_inherited_cfgs(cfg_path) + ["configs/default.yaml"]
        for inherited_cfg_path in all_inherited_cfg_paths:
            shutil.copyfile(inherited_cfg_path, os.path.join(self.training_output_dir, inherited_cfg_path.split("/")[-1]))


    def eval(self, deform_net, semantic_dis_net, infinite_dataloader, output_dir_name):

        eval_output_path = os.path.join(self.training_output_dir, output_dir_name)
        if not os.path.exists(eval_output_path): 
            os.makedirs(eval_output_path)

        with torch.no_grad():
            num_mesh_batch_eval = self.cfg['semantic_dis_training']['num_mesh_batch_eval']
            for _ in range(num_mesh_batch_eval):
                gen_batch = infinite_dataloader.get_batch()
                instance_names = gen_batch["instance_name"]
                _, batch_deformed_meshes, _ = batched_forward_pass(self.cfg, self.device, deform_net, semantic_dis_net, gen_batch, compute_losses=True)
                verts_list = batch_deformed_meshes.verts_list()
                faces_list = batch_deformed_meshes.faces_list()
                for i in range(len(faces_list)):
                    save_obj(os.path.join(eval_output_path, "{}.obj".format(instance_names[i])), verts_list[i], faces_list[i])


    def setup_generator(self, deform_net_type):
        # setting generator input dataset loader
        generation_dataset = GenerationDataset(self.cfg)
        generation_loader = torch.utils.data.DataLoader(generation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=gen_data_collate, drop_last=True)
        
        # setting generator deformation network
        if deform_net_type == "pointnet":
            deform_net = DeformationNetwork(self.cfg, self.cfg["semantic_dis_training"]["mesh_num_verts"], self.device)
            gen_lr = self.cfg["semantic_dis_training"]["gen_pointnet_lr"]
        elif deform_net_type == "gcn_full":
            deform_net = DeformationNetworkGraphConvolutionalFull(self.cfg, self.device)
            gen_lr = self.cfg["semantic_dis_training"]["gen_gcn_lr"]
        elif deform_net_type == "fc_vert_aligned":
            deform_net = DeformationNetworkFcVertexAligned(self.cfg, self.device)
            gen_lr = self.cfg["semantic_dis_training"]["gen_gcn_lr"]

        elif deform_net_type == "gcn_hybrid":
            deform_net = DeformationNetworkGcnHybrid(self.cfg, self.device)
            gen_lr = self.cfg["semantic_dis_training"]["gen_gcn_lr"]
        else:
            raise ValueError("generator deform net type not recognized")

        if self.gen_weight_path != "":
            deform_net.load_state_dict(torch.load(self.gen_weight_path))
        deform_net.to(self.device)
        gen_decay = self.cfg["semantic_dis_training"]["gen_decay"]
        deform_optimizer = optim.Adam(deform_net.parameters(), lr=gen_lr, weight_decay=gen_decay, betas=(self.beta1, 0.99))

        return generation_loader, deform_net, deform_optimizer


    def setup_discriminator(self, dis_type):
        if dis_type == "renders":
            # dataloader
            num_render = self.cfg["semantic_dis_training"]["semantic_dis_num_render"]
            shapenet_renders_dataset = ShapenetRendersDataset(self.cfg)
            semantic_dis_loader = torch.utils.data.DataLoader(shapenet_renders_dataset, batch_size=self.batch_size * num_render, num_workers=self.num_workers, shuffle=True, drop_last=True)
            # network
            semantic_dis_net = RendersSemanticDiscriminatorNetwork(self.cfg)
            if self.dis_weight_path != "":
                semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
            semantic_dis_net.to(self.device)
            #optimizer dis_points_lr
            lr = self.cfg["semantic_dis_training"]["dis_renders_lr"]
            decay = self.cfg["semantic_dis_training"]["dis_renders_decay"]
            semantic_dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=lr, weight_decay=decay, betas=(self.beta1, 0.999))
        
        elif dis_type == "points":
            # dataloader
            #TODO: normalize + data aug on point sets?
            real_dataset = BenchmarkDataset(self.cfg)
            # TODO: not sure about pin_memory
            semantic_dis_loader = torch.utils.data.DataLoader(real_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True)
            #network
            semantic_dis_net = PointsSemanticDiscriminatorNetwork(self.cfg)
            if self.dis_weight_path != "":
                semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
            semantic_dis_net.to(self.device)
            #optimizer
            lr = self.cfg["semantic_dis_training"]["dis_points_lr"]
            decay = self.cfg["semantic_dis_training"]["dis_points_decay"]
            semantic_dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=lr, weight_decay=decay, betas=(self.beta1, 0.99))

        elif dis_type == "multiview":
            # dataloader
            real_dataset = RealDataset(self.cfg, self.device)
            semantic_dis_loader = torch.utils.data.DataLoader(real_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, drop_last=True)
            # network
            semantic_dis_net = MultiviewSemanticDiscriminatorNetwork(self.cfg)
            if self.dis_weight_path != "":
                semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
            semantic_dis_net.to(self.device)
            # optimizer
            lr = self.cfg["semantic_dis_training"]["dis_mv_lr"]
            decay = self.cfg["semantic_dis_training"]["dis_mv_decay"]
            semantic_dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=lr, weight_decay=decay, betas=(self.beta1, 0.999))

        else:
            raise ValueError("dis_type must be renders or pointnet")

        return semantic_dis_loader, semantic_dis_net, semantic_dis_optimizer


    def train(self):

        # setting up generator components 
        generation_loader, deform_net, deform_optimizer = self.setup_generator(self.deform_net_type)
        infinite_gen_loader = general_utils.InfiniteDataLoader(generation_loader)
        G = deform_net
        optimizerG = deform_optimizer

        # setting up discriminator components
        semantic_dis_loader, semantic_dis_net, semantic_dis_optimizer = self.setup_discriminator(self.dis_type)
        D = semantic_dis_net
        optimizerD = semantic_dis_optimizer

        GP = GradientPenalty(self.cfg["semantic_dis_training"]["lambdaGP"], gamma=1, device=self.device)
        num_epochs = self.cfg["semantic_dis_training"]["epochs"]

        # TreeGAN training loop
        for epoch_i in tqdm(range(num_epochs), desc=" >>> Training <<< ", file=general_utils.TqdmPrintEvery()):
            for _iter, point in enumerate(tqdm(semantic_dis_loader, desc="Epoch {}".format(epoch_i), file=general_utils.TqdmPrintEvery())):
                point = point.to(self.device)

                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.cfg["semantic_dis_training"]["D_iter"]):
                    D.zero_grad()
                    
                    gen_batch = infinite_gen_loader.get_batch()
                    with torch.no_grad():
                        #fake_point = G(gen_batch)
                        _, deformed_meshes, _ = batched_forward_pass(self.cfg, self.device, G, D, gen_batch, compute_losses=False)
                        fake_point = deformed_meshes.verts_padded()

                    D_fake = D(fake_point)
                    D_fakem = D_fake.mean() 

                    D_real = D(point)
                    D_realm = D_real.mean()
                    
                    gp_loss = GP(D, point.data, fake_point.data)
                    
                    d_loss = -D_realm + D_fakem
                    d_loss_gp = d_loss + gp_loss
                    d_loss_gp.backward()
                    optimizerD.step()

                # ---------------------- Generator ---------------------- #
                G.zero_grad()
                
                gen_batch = infinite_gen_loader.get_batch()
                #fake_point = G(gen_batch)
                #G_fake = D(fake_point)
                #G_fakem = G_fake.mean()
                #total_loss = -G_fakem
                
                loss_dict, _, _ = batched_forward_pass(self.cfg, self.device, G, D, gen_batch, compute_losses=True)
                total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])

                total_loss.backward()
                optimizerG.step()


            # save network parameters and evaluate meshes using current network
            epoch_i_str = "{num:0{pad}}".format(num=epoch_i, pad=len(str(num_epochs)))
            if epoch_i % self.save_model_every == 0 or epoch_i == num_epochs-1:
                curr_gen_weights_path = os.path.join(self.training_output_dir, "deform_net_weights_{}.pt".format(epoch_i_str))
                curr_dis_weights_path = os.path.join(self.training_output_dir, "semantic_dis_net_weights_{}.pt".format(epoch_i_str))
                torch.save(deform_net.state_dict(), curr_gen_weights_path)
                torch.save(semantic_dis_net.state_dict(), curr_dis_weights_path)
            if epoch_i % self.save_samples_every == 0 or epoch_i == num_epochs-1:
                self.eval(deform_net, semantic_dis_net, infinite_gen_loader, "eval_{}".format(epoch_i_str))



# python trainer_adversarial_semantic_dis.py --cfg configs/test.yaml --exp_name test --num_workers 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarially train a SemanticDiscriminatorNetwork.')
    parser.add_argument('--cfg', type=str, default="configs/default.yaml", help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--exp_name', type=str, default="adv_semantic_discrim", help='name of experiment')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers to use')
    args = parser.parse_args()

    trainer = AdversarialDiscriminatorTrainer(args.cfg, args.gpu, args.exp_name, args.num_workers)
    training_df = trainer.train()
