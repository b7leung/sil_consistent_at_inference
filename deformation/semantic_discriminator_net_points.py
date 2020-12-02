import random

import torch
import torch.nn as nn


#from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended

class PointsSemanticDiscriminatorNetwork(nn.Module):

    # A discriminator for point clouds
    # In particular, feeds the point set into a pointnet to obtain a latent vector, then FC layers classify it.
    def __init__(self, cfg):
        super().__init__()

        self.num_vertices = cfg["semantic_dis_training"]["mesh_num_verts"]
        self.pooling_type = cfg["semantic_dis_training"]["dis_points_pooling"]
        self.spectral_norm = cfg["semantic_dis_training"]["dis_points_spectral_norm"]

        if self.pooling_type == "max":
            pooling_func = nn.MaxPool1d(self.num_vertices)
        elif self.pooling_type == "avg":
            pooling_func = nn.AvgPool1d(self.num_vertices)
        else:
            raise ValueError("pooling type must be avg or max")

        if self.spectral_norm:
            self.pointnet_discriminator = nn.Sequential(
                nn.utils.spectral_norm(nn.Conv1d(3, 64, kernel_size=1, stride=1)),
                nn.LeakyReLU(),
                nn.utils.spectral_norm(nn.Conv1d(64, 128, kernel_size=1, stride=1)),
                nn.LeakyReLU(),
                nn.utils.spectral_norm(nn.Conv1d(128, 1024, kernel_size=1, stride=1)),
                nn.LeakyReLU(),
                pooling_func,
                nn.Flatten(),
                nn.utils.spectral_norm(nn.Linear(1024, 512)),
                nn.LeakyReLU(),
                nn.utils.spectral_norm(nn.Linear(512, 1))
            )
        else:
            self.pointnet_discriminator = nn.Sequential(
                nn.Conv1d(3, 64, kernel_size=1, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(64, 128, kernel_size=1, stride=1),
                nn.LeakyReLU(),
                nn.Conv1d(128, 1024, kernel_size=1, stride=1),
                nn.LeakyReLU(),
                pooling_func,
                nn.Flatten(),
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 1)
            )
        
    
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


    # mesh_vertices (tensor): a batch_size x num_vertices x 3 tensor of vertices (ie, a pointcloud)
    # output: a [batch_size, 1] tensor of raw logits
    # TODO: make this work with a variable number of mesh vertices
    def forward(self, mesh_vertices):

        mesh_vertices = torch.transpose(mesh_vertices, 1, 2)
        out = self.pointnet_discriminator(mesh_vertices)

        return out