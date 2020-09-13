import torch
import torch.nn as nn

from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended

class PointsSemanticDiscriminatorNetwork(nn.Module):

    # A discriminator for point clouds
    # In particular, feeds the point set into a pointnet to obtain a latent vector, then FC layers classify it.
    def __init__(self, cfg):
        super().__init__()

        dropout_p = cfg["semantic_dis_training"]["dropout_p"]
        pointnet_encoding_dim = cfg['model']['latent_dim_pointnet']
        dis_points_encoder = cfg["semantic_dis_training"]["dis_points_encoder"]

        # TODO: regularize by adding batchnorm, dropout?
        if dis_points_encoder == "pointnet":
            self.points_encoder = ResnetPointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)

        elif dis_points_encoder == "fc":
            num_vertices = cfg["semantic_dis_training"]["mesh_num_verts"]
            self.points_encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_vertices*3, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Dropout(p=dropout_p),
                nn.Linear(1024, pointnet_encoding_dim),
                nn.BatchNorm1d(pointnet_encoding_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError("dis_points_encoder not recognized")

        actvn = nn.LeakyReLU()
        self.fully_connected = nn.Sequential(
            nn.Linear(pointnet_encoding_dim, 1024),
            nn.BatchNorm1d(1024),
            actvn,
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            actvn,
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            actvn,
            nn.Dropout(p=dropout_p),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            actvn,
            nn.Dropout(p=dropout_p),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            actvn,
            nn.Linear(64, 1)
        )

    # mesh_vertices (tensor): a batch_size x num_vertices x 3 tensor of vertices (ie, a pointcloud)
    def forward(self, mesh_vertices):
        verts_encodings = self.points_encoder(mesh_vertices)
        logits = self.fully_connected(verts_encodings)

        return logits