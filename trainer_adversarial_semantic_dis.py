import argparse
import os
import pickle
import time
import shutil

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
from utils.datasets import GenerationDataset, ShapenetRendersDataset, ShapenetPointsDataset, RealMultiviewDataset
#from deformation.semantic_discriminator_net_renders import RendersSemanticDiscriminatorNetwork
from deformation.semantic_discriminator_net_points import PointsSemanticDiscriminatorNetwork
from deformation.multiview_semantic_discriminator_network import MultiviewSemanticDiscriminatorNetwork
from utils.datasets import gen_data_collate
from deformation.forward_pass import batched_forward_pass, compute_sem_dis_logits
from utils.visualization_tools import save_tensor_img


class AdversarialDiscriminatorTrainer():

    def __init__(self, cfg_path, gpu_num, exp_name, num_workers):
        self.cfg = general_utils.load_config(cfg_path, "configs/default.yaml")
        self.device = torch.device("cuda:"+str(gpu_num))
        self.batch_size = self.cfg["semantic_dis_training"]["batch_size"]
        self.dis_type = self.cfg['semantic_dis_training']['dis_type']
        self.deform_net_type = self.cfg["semantic_dis_training"]["deform_net_type"]
        self.dis_weight_path = self.cfg["semantic_dis_training"]["dis_weight_path"]
        self.gen_weight_path = self.cfg["semantic_dis_training"]["gen_weight_path"]
        self.total_training_iters = self.cfg["semantic_dis_training"]["adv_iterations"]
        self.dis_epochs_per_iteration = self.cfg["semantic_dis_training"]["dis_epochs_per_iteration"]
        self.gen_epochs_per_iteration = self.cfg["semantic_dis_training"]["gen_epochs_per_iteration"]
        self.early_stop_dis_acc = self.cfg["semantic_dis_training"]["early_stop_dis_acc"]
        self.save_model_every = self.cfg["semantic_dis_training"]["save_model_every"]

        # creating output dir
        self.training_output_dir = os.path.join(self.cfg['semantic_dis_training']['output_dir'], "{}_{}".format(time.strftime("%Y_%m_%d--%H_%M_%S"), exp_name))
        if not os.path.exists(self.training_output_dir): os.makedirs(self.training_output_dir)
        shutil.copyfile(cfg_path, os.path.join(self.training_output_dir, cfg_path.split("/")[-1]))

        # for adding noise to training labels. Real images have label 1, fake images has label 0
        self.label_noise = self.cfg["semantic_dis_training"]["label_noise"]
        self.real_label_offset = self.cfg["semantic_dis_training"]["real_label_offset"]
        self.real_labels_dist_gen = torch.distributions.Uniform(torch.tensor([1.0]), torch.tensor([1.0]))
        self.real_labels_dist = torch.distributions.Uniform(torch.tensor([(1.0-self.real_label_offset)-self.label_noise]), torch.tensor([(1.0-self.real_label_offset)]))
        self.fake_labels_dist = torch.distributions.Uniform(torch.tensor([0.0]), torch.tensor([0.0+self.label_noise]))

        self.num_workers = num_workers
        self.tqdm_out = general_utils.TqdmPrintEvery()


    # TODO: write description of what this does
    def eval(self, deform_net, semantic_dis_net, output_dir_name):
        num_mesh_eval = self.cfg['semantic_dis_training']['num_mesh_eval']
        if num_mesh_eval == 0:
            pass

        eval_output_path = os.path.join(self.training_output_dir, output_dir_name)
        if not os.path.exists(eval_output_path): os.makedirs(eval_output_path)

        eval_generation_dataset = GenerationDataset(self.cfg)
        eval_generation_loader = torch.utils.data.DataLoader(eval_generation_dataset, batch_size=1, num_workers=1, shuffle=False, collate_fn=gen_data_collate)
        if num_mesh_eval == -1: num_mesh_eval = len(eval_generation_loader) + 1

        loss_info = {}
        with tqdm(total=num_mesh_eval, desc="Evaluating Meshes", leave=False) as pbar:
            for batch_idx, gen_batch in enumerate(eval_generation_loader):
                if batch_idx >= num_mesh_eval: break

                deform_net.eval()
                semantic_dis_net.eval()
                loss_dict, deformed_mesh, _ = batched_forward_pass(self.cfg, self.device, deform_net, semantic_dis_net, gen_batch, compute_losses=True)

                curr_instance_name = gen_batch["instance_name"][0]
                save_obj(os.path.join(eval_output_path, "{}.obj".format(curr_instance_name)), deformed_mesh.verts_packed(), deformed_mesh.faces_packed())

                total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])
                curr_train_info = {"total_loss": total_loss.item()}
                curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
                loss_info[curr_instance_name] = curr_train_info
                pbar.update(1)

        pickle.dump(loss_info, open(os.path.join(eval_output_path, "eval_loss_info.p"),"wb"))


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
        else:
            raise ValueError("generator deform net type not recognized")

        if self.gen_weight_path != "":
            deform_net.load_state_dict(torch.load(self.gen_weight_path))
        deform_net.to(self.device)
        gen_decay = self.cfg["semantic_dis_training"]["gen_decay"]
        deform_optimizer = optim.Adam(deform_net.parameters(), lr=gen_lr, weight_decay=gen_decay)

        return generation_loader, deform_net, deform_optimizer


    def setup_discriminator(self, dis_type):
        if dis_type == "renders":
            # dataloader
            num_render = self.cfg["semantic_dis_training"]["semantic_dis_num_render"]
            shapenet_renders_dataset = ShapenetRendersDataset(self.cfg)
            semantic_dis_loader = torch.utils.data.DataLoader(shapenet_renders_dataset, batch_size=self.batch_size * num_render, num_workers=self.num_workers, shuffle=True)
            # network
            semantic_dis_net = RendersSemanticDiscriminatorNetwork(self.cfg)
            if self.dis_weight_path != "":
                semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
            semantic_dis_net.to(self.device)
            #optimizer dis_points_lr
            lr = self.cfg["semantic_dis_training"]["dis_renders_lr"]
            decay = self.cfg["semantic_dis_training"]["dis_renders_decay"]
            semantic_dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=lr, weight_decay=decay)
        
        elif dis_type == "points":
            # dataloader
            #TODO: normalize + data aug on point sets?
            shapenet_points_dataset = ShapenetPointsDataset(self.cfg)
            semantic_dis_loader = torch.utils.data.DataLoader(shapenet_points_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            #network
            semantic_dis_net = PointsSemanticDiscriminatorNetwork(self.cfg)
            if self.dis_weight_path != "":
                semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
            semantic_dis_net.to(self.device)
            #optimizer
            lr = self.cfg["semantic_dis_training"]["dis_points_lr"]
            decay = self.cfg["semantic_dis_training"]["dis_points_decay"]
            semantic_dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=lr, weight_decay=decay)

        elif dis_type == "multiview":
            # dataloader
            real_multiview_dataset = RealMultiviewDataset(self.cfg, self.device)
            semantic_dis_loader = torch.utils.data.DataLoader(real_multiview_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
            # network
            semantic_dis_net = MultiviewSemanticDiscriminatorNetwork(self.cfg)
            if self.dis_weight_path != "":
                semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
            semantic_dis_net.to(self.device)
            # optimizer
            lr = self.cfg["semantic_dis_training"]["dis_mv_lr"]
            decay = self.cfg["semantic_dis_training"]["dis_mv_decay"]
            semantic_dis_optimizer = optim.Adam(semantic_dis_net.parameters(), lr=lr, weight_decay=decay)

        else:
            raise ValueError("dis_type must be renders or pointnet")

        return semantic_dis_loader, semantic_dis_net, semantic_dis_optimizer


    def train(self):

        # setting up generator components 
        generation_loader, deform_net, deform_optimizer = self.setup_generator(self.deform_net_type)

        # setting up discriminator components
        semantic_dis_loader, semantic_dis_net, semantic_dis_optimizer = self.setup_discriminator(self.dis_type)

        # training generative deformation network and discriminator in an alternating, GAN style
        training_df = {"deform_net_gen": pd.DataFrame(), "semantic_dis": pd.DataFrame()}
        for iter_idx in tqdm(range(self.total_training_iters), desc="Alternating MiniMax Iterations", leave=False):

            # training discriminator; generator weights are frozen
            ############################################################################################################################
            for param in semantic_dis_net.parameters(): param.requires_grad = True
            for param in deform_net.parameters(): param.requires_grad = False
            dis_batch_acc_history = []
            early_stop = False
            for dis_epoch in tqdm(range(self.dis_epochs_per_iteration), desc="Iteration {} Discriminator Trainings".format(iter_idx), leave=False):
                for batch_idx, (gen_batch, real_batch) in enumerate(tqdm(zip(generation_loader, semantic_dis_loader), leave=False,
                                                                         total=min(len(generation_loader), len(semantic_dis_loader)), desc="Discriminator Epoch {} Batches".format(dis_epoch))):

                    # if early stop, just fill the rest of the dataframe with last recorded value
                    # TODO: make this faster by not going through dataloaders
                    if early_stop:
                        curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "semantic_dis_loss": dis_loss.item(), "batch_avg_dis_acc": batch_accuracy}
                        training_df["semantic_dis"] = training_df["semantic_dis"].append(curr_train_info, ignore_index=True)
                        continue

                    semantic_dis_net.train()
                    deform_net.eval() # not sure if supposed to set this
                    semantic_dis_optimizer.zero_grad()

                    # computing real discriminator logits
                    real_batch = real_batch.to(self.device)
                    pred_logits_real = semantic_dis_net(real_batch)

                    # computing fake discriminator logits
                    _, deformed_meshes, _ = batched_forward_pass(self.cfg, self.device, deform_net, semantic_dis_net, gen_batch, compute_losses=False)
                    pred_logits_fake, semantic_dis_debug_data = compute_sem_dis_logits(deformed_meshes, semantic_dis_net, self.device, self.cfg)

                    batch_size = real_batch.shape[0]
                    real_labels = self.real_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)
                    fake_labels = self.fake_labels_dist.sample((batch_size,1)).squeeze(2).to(self.device)
                    dis_loss = F.binary_cross_entropy_with_logits(pred_logits_real, real_labels) + \
                        F.binary_cross_entropy_with_logits(pred_logits_fake, fake_labels)

                    dis_loss.backward()
                    semantic_dis_optimizer.step()

                    # compute accuracy & save to dataframe
                    batch_accuracies = []
                    real_correct_vec = (torch.sigmoid(pred_logits_real) > 0.5) == (real_labels > 0.5)
                    fake_correct_vec = (torch.sigmoid(pred_logits_fake) > 0.5) == (fake_labels > 0.5)
                    batch_accuracies.append(real_correct_vec.cpu().numpy())
                    batch_accuracies.append(fake_correct_vec.cpu().numpy())
                    batch_accuracy = np.mean(np.concatenate(batch_accuracies, axis=0)).item()
                    curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "semantic_dis_loss": dis_loss.item(), "batch_avg_dis_acc": batch_accuracy}
                    training_df["semantic_dis"] = training_df["semantic_dis"].append(curr_train_info, ignore_index=True)

                    # keeping running average of past 10 discriminator accuracies, and early stop if needed
                    if self.early_stop_dis_acc != -1:
                        dis_batch_acc_history.insert(0,batch_accuracy)
                        #tqdm.write(str(dis_batch_acc_history))
                        if len(dis_batch_acc_history) > 10:
                            dis_batch_acc_history.pop()
                            dis_batch_avg_acc = np.average(dis_batch_acc_history)
                            if dis_batch_avg_acc >= self.early_stop_dis_acc:
                                early_stop = True
                                tqdm.write("Iter. {}: Early stopped dis. training at epoch {}, batch {}, with avg acc of {}.".format(iter_idx, dis_epoch, batch_idx, dis_batch_avg_acc))

                    # if discriminator is render based, save some example inputs to discriminator from the first batch
                    if self.cfg['semantic_dis_training']['dis_type'] in ["renders", "multiview"] and dis_epoch == 0 and batch_idx == 0:
                        img_output_dir = os.path.join(self.training_output_dir, "training_saved_images", "iter_{}".format(iter_idx))
                        # TODO: saving the real batch can be disabled, as long as everything looks right
                        save_tensor_img(real_batch, "iter_{}_real".format(iter_idx), img_output_dir, 32)
                        save_tensor_img(semantic_dis_debug_data, "iter_{}_fake".format(iter_idx), img_output_dir, 32)

                if training_df["semantic_dis"].isnull().values.any():
                    tqdm.write("WARNING: nan in dataframe.")
                    raise ZeroDivisionError("nan in dataframe")
                pickle.dump(training_df, open(os.path.join(self.training_output_dir, "training_df.p"),"wb"))


            # training generator; discriminator weights are frozen
            ############################################################################################################################
            for param in semantic_dis_net.parameters(): param.requires_grad = False
            for param in deform_net.parameters(): param.requires_grad = True
            for gen_epoch in tqdm(range(self.gen_epochs_per_iteration), desc="Iteration {} Generator Trainings".format(iter_idx), leave=False):
                for batch_idx, gen_batch in enumerate(tqdm(generation_loader, desc="Generator Epoch {} Batches".format(gen_epoch), leave=False)):

                    deform_net.train()
                    semantic_dis_net.eval()
                    deform_optimizer.zero_grad()

                    loss_dict, _, _ = batched_forward_pass(self.cfg, self.device, deform_net, semantic_dis_net, gen_batch, compute_losses=True)
                    total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])

                    total_loss.backward()
                    deform_optimizer.step()

                    curr_train_info = {"iteration": iter_idx, "batch": batch_idx, "total_loss": total_loss.item()}
                    curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
                    training_df["deform_net_gen"] = training_df["deform_net_gen"].append(curr_train_info, ignore_index=True)

                if training_df["deform_net_gen"].isnull().values.any():
                    tqdm.write("WARNING: nan in dataframe.")
                    raise ZeroDivisionError("nan in dataframe")
                pickle.dump(training_df, open(os.path.join(self.training_output_dir, "training_df.p"),"wb"))

            # save network parameters and evaluate meshes using current network
            if iter_idx % self.save_model_every == 0 or iter_idx == self.total_training_iters-1:
                curr_gen_weights_path = os.path.join(self.training_output_dir, "deform_net_weights_{}.pt".format(iter_idx))
                curr_dis_weights_path = os.path.join(self.training_output_dir, "semantic_dis_net_weights_{}.pt".format(iter_idx))
                torch.save(deform_net.state_dict(), curr_gen_weights_path)
                torch.save(semantic_dis_net.state_dict(), curr_dis_weights_path)
                self.eval(deform_net, semantic_dis_net, "eval_{}".format(iter_idx))



# python trainer_adversarial_semantic_dis.py --cfg configs/test.yaml --exp_name test --num_workers 2
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarially train a SemanticDiscriminatorNetwork.')
    parser.add_argument('--cfg', type=str, default="configs/default.yaml", help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--exp_name', type=str, default="adv_semantic_discrim", help='name of experiment')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers to use')
    args = parser.parse_args()

    start_time = time.time()
    trainer = AdversarialDiscriminatorTrainer(args.cfg, args.gpu, args.exp_name, args.num_workers)
    training_df = trainer.train()
    end_time = time.time()
    print("\n \n >>> Finished training in {} seconds. <<< \n".format(end_time - start_time))
