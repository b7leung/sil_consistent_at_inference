import argparse
import os
import pickle
import time
import shutil

import torch
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from tqdm.autonotebook import tqdm
import pandas as pd
import numpy as np

from utils import utils
from deformation.deformation_net_extended import DeformationNetworkExtended
import deformation.losses as def_losses
from deformation.semantic_discriminator_loss import compute_sem_dis_logits
from adversarial.datasets import GenerationDataset, ShapenetRendersDataset
from deformation.semantic_discriminator_net import SemanticDiscriminatorNetwork
from adversarial.datasets import gen_data_collate
from eval_adversarial_semantic_dis import AdversarialDiscriminatorEval


class AdversarialDiscriminatorTrainer():

    def __init__(self, cfg_path, gpu_num, exp_name, light):
        self.cfg = utils.load_config(cfg_path, "configs/default.yaml")
        self.device = torch.device("cuda:"+str(gpu_num))

        self.batch_size = self.cfg["semantic_dis_training"]["batch_size"]
        self.mesh_num_vertices = self.cfg["semantic_dis_training"]["mesh_num_verts"]
        self.semantic_dis_loss_num_render = self.cfg["training"]["semantic_dis_num_render"]
        self.dis_weight_path = self.cfg["semantic_dis_training"]["dis_weight_path"]
        self.gen_weight_path = self.cfg["semantic_dis_training"]["gen_weight_path"]

        self.total_training_iters = self.cfg["semantic_dis_training"]["adv_iterations"]
        self.dis_epochs_per_iteration = self.cfg["semantic_dis_training"]["dis_epochs_per_iteration"]
        self.gen_epochs_per_iteration = self.cfg["semantic_dis_training"]["gen_epochs_per_iteration"]
        self.save_model_every = self.cfg["semantic_dis_training"]["save_model_every"]
        self.num_mesh_eval = self.cfg['semantic_dis_training']['num_mesh_eval']

        self.training_output_dir = os.path.join(self.cfg['semantic_dis_training']['output_dir'], "{}_{}".format(time.strftime("%Y_%m_%d--%H_%M_%S"), exp_name))
        if not os.path.exists(self.training_output_dir):
            os.makedirs(self.training_output_dir)
        shutil.copyfile(cfg_path, os.path.join(self.training_output_dir, cfg_path.split("/")[-1]))

        # for adding noise to training labels. Real images have label 1, fake images has label 0
        self.label_noise = self.cfg["semantic_dis_training"]["label_noise"]
        self.real_label_offset = self.cfg["semantic_dis_training"]["real_label_offset"]
        self.real_labels_dist = torch.distributions.Uniform(torch.tensor([(1.0-self.real_label_offset)-self.label_noise]), torch.tensor([(1.0-self.real_label_offset)]))
        self.fake_labels_dist = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([0.0+self.label_noise]))

        if light:
            self.num_workers = 4
        else:
            self.num_workers = 8
        self.tqdm_out = utils.TqdmPrintEvery()


    # given a tensor of batches of images, dimensions (b x w x h x c) or channel-first (b x c x w x h)
    # saves a specified amount of them
    def save_tensor_img(self, tensor, channel_first, name_prefix, output_dir, save_num=5):
        img_transforms = transforms.Compose([
            transforms.ToPILImage()
        ])

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        tensor = tensor.detach().cpu()
        if not channel_first:
            tensor = tensor.permute(0, 3, 1, 2)
        if save_num != -1:
            tensor = tensor[:save_num]
        for i, img_tensor in enumerate(tensor):
            img_transforms(img_tensor).save(os.path.join(output_dir, name_prefix+"_{}.jpg".format(i)))


    def train(self):

        # setting up dataloaders
        generation_dataset = GenerationDataset(self.cfg, self.device)
        generation_loader = torch.utils.data.DataLoader(generation_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=gen_data_collate, drop_last=True)
        shapenet_renders_dataset = ShapenetRendersDataset(self.cfg)
        shapenet_renders_loader = torch.utils.data.DataLoader(shapenet_renders_dataset, batch_size=self.batch_size * self.semantic_dis_loss_num_render, num_workers=self.num_workers, shuffle=True)

        # setting up networks and optimizers
        deform_net = DeformationNetworkExtended(self.cfg, self.mesh_num_vertices, self.device)
        if self.gen_weight_path != "":
            deform_net.load_state_dict(torch.load(self.gen_weight_path))
        deform_net.to(self.device)
        deform_optimizer = optim.Adam(deform_net.parameters(), lr=self.cfg["training"]["learning_rate"])

        semantic_dis_net = SemanticDiscriminatorNetwork(self.cfg)
        if self.dis_weight_path != "":
            semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
        semantic_dis_net.to(self.device)
        dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=self.cfg["semantic_dis_training"]["dis_learning_rate"], weight_decay=1e-2)

        # training generative deformation network and discriminator in an alternating, GAN style
        training_df = {"deform_net_gen": pd.DataFrame(), "semantic_dis": pd.DataFrame()}
        for iter_idx in tqdm(range(self.total_training_iters), desc="Alternating MiniMax Iterations"):

            # training discriminator; generator weights are frozen
            ############################################################################################################################
            for param in semantic_dis_net.parameters(): param.requires_grad = True
            for param in deform_net.parameters(): param.requires_grad = False

            for dis_epoch in tqdm(range(self.dis_epochs_per_iteration), desc="Iteration {} Discriminator Trainings".format(iter_idx)):
                for batch_idx, (gen_batch, real_render_batch) in enumerate(tqdm(zip(generation_loader, shapenet_renders_loader),
                                                                           total=len(generation_loader), desc="Discriminator Epoch {} Batches".format(dis_epoch))):

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
                    pred_logits_fake, processed_renders = compute_sem_dis_logits(deformed_meshes, semantic_dis_net, self.device, self.cfg)

                    batch_size = real_render_batch.shape[0]
                    real_labels = self.real_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)
                    fake_labels = self.fake_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)

                    dis_loss = F.binary_cross_entropy_with_logits(pred_logits_real, real_labels) + \
                        F.binary_cross_entropy_with_logits(pred_logits_fake, fake_labels)

                    dis_loss.backward()
                    dis_optimizer.step()

                    # compute accuracy
                    batch_accuracies = []
                    real_correct_vec = (torch.sigmoid(pred_logits_real) > 0.5) == real_labels.byte()
                    fake_correct_vec = (torch.sigmoid(pred_logits_fake) > 0.5) == fake_labels.byte()
                    batch_accuracies.append(real_correct_vec.cpu().numpy())
                    batch_accuracies.append(fake_correct_vec.cpu().numpy())
                    batch_accuracy = np.mean(np.concatenate(batch_accuracies, axis=0)).item()
                    curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "semantic_dis_loss": dis_loss.item(), "batch_avg_dis_acc": batch_accuracy}
                    training_df["semantic_dis"] = training_df["semantic_dis"].append(curr_train_info, ignore_index=True)
                    # save some example inputs to discriminator from the first batch
                    if dis_epoch == 0 and batch_idx == 0:
                        img_output_dir = os.path.join(self.training_output_dir, "training_saved_images", "iter_{}".format(iter_idx))
                        self.save_tensor_img(real_render_batch, True, "iter_{}_real".format(iter_idx), img_output_dir, 32)
                        self.save_tensor_img(processed_renders, False, "iter_{}_fake".format(iter_idx), img_output_dir, 32)


            # training generator; discriminator weights are frozen
            ############################################################################################################################
            for param in semantic_dis_net.parameters(): param.requires_grad = False
            for param in deform_net.parameters(): param.requires_grad = True

            for gen_epoch in tqdm(range(self.gen_epochs_per_iteration), desc="Iteration {} Generator Trainings".format(iter_idx)):
                for batch_idx, gen_batch in enumerate(tqdm(generation_loader, desc="Generator Epoch {} Batches".format(gen_epoch))):

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
                    curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
                    training_df["deform_net_gen"] = training_df["deform_net_gen"].append(curr_train_info, ignore_index=True)

                pickle.dump(training_df, open(os.path.join(self.training_output_dir, "training_df.p"),"wb"))

            if iter_idx % self.save_model_every == 0 or iter_idx == self.total_training_iters-1:
                curr_gen_weights_path = os.path.join(self.training_output_dir, "deform_net_weights_{}.pt".format(iter_idx))
                curr_dis_weights_path = os.path.join(self.training_output_dir, "semantic_dis_net_weights_{}.pt".format(iter_idx))
                torch.save(deform_net.state_dict(), curr_gen_weights_path)
                torch.save(semantic_dis_net.state_dict(), curr_dis_weights_path)
                # show mesh results from current network
                if self.num_mesh_eval > 0:
                    evaluator = AdversarialDiscriminatorEval(self.training_output_dir, curr_gen_weights_path, curr_dis_weights_path,
                                                             "eval_{}".format(iter_idx), self.num_mesh_eval, self.device)
                    evaluator.eval()


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
            # TODO: fix this, or remove del v + v option entirely
            if self.cfg["model"]["output_delta_V"]:
                loss_dict["l2_loss"] = F.mse_loss(deformation_output, zero_deformation_tensor)
            else:
                loss_dict["l2_loss"] = torch.tensor(0).to(self.device)

            loss_dict["lap_smoothness_loss"] = mesh_laplacian_smoothing(deformed_meshes) # TODO: experiment with different variants (see pytorch3d docs)
            loss_dict["normal_consistency_loss"] = mesh_normal_consistency(deformed_meshes)

            if self.cfg["training"]["semantic_dis_lam"] > 0:
                sem_dis_logits, _ = compute_sem_dis_logits(deformed_meshes, semantic_dis_net, self.device, self.cfg)
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
