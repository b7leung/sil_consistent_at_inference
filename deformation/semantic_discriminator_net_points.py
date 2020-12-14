import random

import torch
import torch.nn as nn
from torch.nn import functional as F


class PointsSemanticDiscriminatorNetwork(nn.Module):
    # based on TreeGAN
    def __init__(self, cfg):
        super().__init__()

        features = [3, 64, 128, 256, 512, 1024]
        self.layer_num = len(features)-1

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            self.fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
                                         nn.Linear(features[-1], features[-2]),
                                         nn.Linear(features[-2], features[-2]),
                                         nn.Linear(features[-2], 1))

    
    # assumes vertices is batch size 1
    # ensures vertices is >= target_vertex_num
    def adjust_num_vertices(self, vertices, target_vertex_num, seed=0):
        # TODO: seed?. Also this can be done in one swoop, w/o while loop
        # TODO: add warnings if num vertex is way off

        # if mesh has too few vertices, duplicate random mesh vertices (they are "floating", no edges are added) 
        # until target number of vertices is reached
        while vertices.shape[1] < target_vertex_num:
            random_vertex_idx = random.randint(0,vertices.shape[1]-1)
            vertices = torch.cat([vertices, vertices[:,random_vertex_idx,:].unsqueeze(0)], axis=1)

        return vertices


    # mesh_vertices (tensor): a [batch_size, num_vertices, 3] tensor of vertices (ie, a pointcloud)
    # output: a [batch_size, 1] tensor of raw logits
    def forward(self, mesh_vertices):


        feat = mesh_vertices.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
            feat = self.leaky_relu(feat)

        out = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(out) # (B, 1)

        return out