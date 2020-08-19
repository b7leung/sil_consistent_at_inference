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

from .semantic_discriminator_net import SemanticDiscriminatorNetwork
from utils import utils, network_utils


# data should be shapenet renders, 224 x224 jpgs w/ white bg
def train(cfg_path, gpu_num):

    device = torch.device("cuda:"+str(gpu_num))
    cfg = utils.load_config(cfg_path)

    # setting up dataloader
    real_dataset_dir = cfg['semantic_dis_training']['real_dataset_dir']
    fake_dataset_dir = cfg['semantic_dis_training']['fake_dataset_dir']

    # setting up network and optimizer
    semantic_discriminator_net = SemanticDiscriminatorNetwork(cfg, device)
    semantic_discriminator_net.to(device)
    optimizer = optim.Adam(semantic_discriminator_net.parameters(), lr=0.0001)

    # training
    training_df = pd.DataFrame()
    for epoch_i in tqdm(range(cfg['semantic_dis_training']['epochs'])):
        semantic_discriminator_net.train()
        optimizer.zero_grad()

        epoch_loss_info = {"epoch": epoch_i}
        training_df = training_df.append(epoch_loss_info, ignore_index = True)
    

    return training_df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a SemanticDiscriminatorNetwork.')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()

    training_df = train(args.cfg_path, args.gpu)



