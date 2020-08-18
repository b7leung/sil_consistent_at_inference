import argparse
import os
import glob
import pprint
import pickle

from tqdm.autonotebook import tqdm
import torch
from PIL import Image
import numpy as np

from .semantic_discriminator_net import SemanticDiscriminatorNetwork
from utils import utils, network_utils


parser = argparse.ArgumentParser(
    description='Train a SemanticDiscriminatorNetwork.'
)
parser.add_argument('real_dataset_dir', type=str, help='Path to folder with rendered images of real 3d models.')
parser.add_argument('fake_dataset_dir', type=str, help='Path to folder with rendered images of fake 3d models.')
parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
args = parser.parse_args()

device = torch.device("cuda:"+str(args.gpu))
cfg = utils.load_config(args.cfg_path)


# setting up dataloader


# setting up network and optimizer
semantic_discriminator_net = SemanticDiscriminatorNetwork(cfg, device)
semantic_discriminator_net.to(device)
optimizer = optim.Adam(semantic_discriminator_net.parameters(), lr=0.0001)

for epoch_i in tqdm(range(cfg['semantic_dis_training']['epochs'])):
    semantic_discriminator_net.train()
    optimizer.zero_grad()





