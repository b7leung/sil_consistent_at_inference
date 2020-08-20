import argparse
import os
import glob
import pprint
import pickle

from tqdm.autonotebook import tqdm
import torch
from PIL import Image
import numpy as np
import pandas as pd
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from .semantic_discriminator_net import SemanticDiscriminatorNetwork
from .semantic_discriminator_dataset import SemanticDiscriminatorDataset
from utils import utils, network_utils


# data should be shapenet renders, 224 x224 jpgs w/ white bg
def train(cfg_path, gpu_num):

    device = torch.device("cuda:"+str(gpu_num))
    cfg = utils.load_config(cfg_path)

    # setting up dataloader
    train_dataset = SemanticDiscriminatorDataset(cfg, "train")
    val_dataset = SemanticDiscriminatorDataset(cfg, "val")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, num_workers=8, shuffle=True,
        collate_fn=None, worker_init_fn=None)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=8, num_workers=8, shuffle=False,
        collate_fn=None, worker_init_fn=None)

    # setting up network and optimizer
    semantic_discriminator_net = SemanticDiscriminatorNetwork(cfg, device)
    semantic_discriminator_net.to(device)
    optimizer = optim.Adam(semantic_discriminator_net.parameters(), lr=0.0001)

    # training
    df_dict = {"train": pd.DataFrame(), "val": pd.DataFrame()}
    iteration_i = 0
    for epoch_i in tqdm(range(cfg['semantic_dis_training']['epochs'])):

        for batch in train_loader:
            semantic_discriminator_net.train()
            optimizer.zero_grad()
            batch_size = batch['real'].shape[0]
            real_imgs = batch['real'].to(device)
            fake_imgs = batch['fake'].to(device)
            # real images have label 0, fake images has label 1
            real_labels = torch.zeros((batch_size, 1)).to(device)
            fake_labels = torch.ones((batch_size, 1)).to(device)

            pred_logits_real = semantic_discriminator_net(real_imgs)
            pred_logits_fake = semantic_discriminator_net(fake_imgs)
            loss = F.binary_cross_entropy_with_logits(pred_logits_real, real_labels) + \
                F.binary_cross_entropy_with_logits(pred_logits_fake, fake_labels)
            loss.backward()
            optimizer.step()

            curr_train_info = {"epoch": epoch_i, "iteration": iteration_i, "train_loss": loss.item()}
            df_dict["train"] = df_dict["train"].append(curr_train_info, ignore_index = True)
            iteration_i += 1


        # computing validation set accuracy
        if epoch_i % cfg['semantic_dis_training']['eval_every'] == 0:
            val_accuracies = []
            for val_batch in val_loader:
                with torch.no_grad():
                    pred_logits_real = semantic_discriminator_net(val_batch['real'].to(device))
                    pred_logits_fake = semantic_discriminator_net(val_batch['fake'].to(device))
                    batch_size = val_batch['real'].shape[0]
                    real_labels = torch.zeros((batch_size, 1)).to(device)
                    fake_labels = torch.ones((batch_size, 1)).to(device)
                    real_correct_vec = (torch.sigmoid(pred_logits_real) > 0.5) == real_labels.byte()
                    fake_correct_vec = (torch.sigmoid(pred_logits_fake) > 0.5) == fake_labels.byte()
                    val_accuracies.append(real_correct_vec.cpu().numpy())
                    val_accuracies.append(fake_correct_vec.cpu().numpy())
            val_accuracy = np.mean(np.concatenate(val_accuracies, axis = 0))
            curr_val_info = {"epoch": epoch_i, "val_acc": val_accuracy.item()}
            df_dict["val"] = df_dict["val"].append(curr_val_info, ignore_index = True)
    
    # saving model
    torch.save(semantic_discriminator_net.state_dict(), os.path.join(cfg['semantic_dis_training']['weight_path']))


    return df_dict 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SemanticDiscriminatorNetwork.')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()

    training_df = train(args.cfg_path, args.gpu)


