import torch
import torch.nn as nn

from .pointnet import SimplePointnet, ResnetPointnet

class PointsSemanticDiscriminatorNetwork(nn.Module):

    # A discriminator for point clouds
    # In particular, feeds the point set into a pointnet to obtain a latent vector, then FC layers classify it.
    def __init__(self, cfg):
        super().__init__()

        pointnet_encoding_dim = cfg['model']['latent_dim_pointnet']
        dropout_p = cfg["semantic_dis_training"]["dropout_p"]

        #self.pointnet_encoder = SimplePointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)
        self.pointnet_encoder = ResnetPointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)
        # TODO: regularize by adding batchnorm, dropout?
        self.fully_connected = nn.Sequential(
            nn.Linear(pointnet_encoding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    # mesh_vertices (tensor): a batch_size x num_vertices x 3 tensor of vertices (ie, a pointcloud)
    def forward(self, mesh_vertices):
        verts_encodings = self.pointnet_encoder(mesh_vertices)
        logits = self.fully_connected(verts_encodings)

        return logits