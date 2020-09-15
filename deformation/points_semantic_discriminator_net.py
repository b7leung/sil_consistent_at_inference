import torch
import torch.nn as nn

from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended

class PointsSemanticDiscriminatorNetwork(nn.Module):

    # A discriminator for point clouds
    # In particular, feeds the point set into a pointnet to obtain a latent vector, then FC layers classify it.
    def __init__(self, cfg):
        super().__init__()

        self.dis_points_encoder = cfg["semantic_dis_training"]["dis_points_encoder"]
        num_vertices = cfg["semantic_dis_training"]["mesh_num_verts"]

        if self.dis_points_encoder in ["pointnet", "fc"]:
            dropout_p = cfg["semantic_dis_training"]["dropout_p"]
            pointnet_encoding_dim = cfg['model']['latent_dim_pointnet']
            actvn = nn.LeakyReLU()

            # setting up encoder
            if self.dis_points_encoder == "pointnet":
                self.points_encoder = ResnetPointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)
            else:
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

            # setting up FC layers
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

            self.dis_net = nn.Sequential(self.points_encoder, self.fully_connected)

        elif self.dis_points_encoder == "rgan":
            # borrowed from https://github.com/seowok/TreeGAN
            features = [3,  64,  128, 256, 512, 1024]
            layer_num = len(features)-1
            fc_layers = []
            for inx in range(layer_num):
                fc_layers.append(nn.Conv1d(features[inx], features[inx+1], kernel_size=1, stride=1))
            leaky_relu = nn.LeakyReLU(negative_slope=0.2)
            final_layer = nn.Sequential(nn.Linear(features[-1], features[-1]),
                                        nn.Linear(features[-1], features[-2]),
                                        nn.Linear(features[-2], features[-2]),
                                        nn.Linear(features[-2], 1))
            # putting network together
            rgan_net_list = []
            for i in range(layer_num):
                rgan_net_list.append(fc_layers[i])
                rgan_net_list.append(leaky_relu)
            rgan_net_list.append(nn.MaxPool1d(num_vertices))
            rgan_net_list.append(nn.Flatten())
            rgan_net_list.append(final_layer)

            self.dis_net = nn.Sequential(*rgan_net_list)

        else:
            raise ValueError("dis_points_encoder not recognized")
        

    # mesh_vertices (tensor): a batch_size x num_vertices x 3 tensor of vertices (ie, a pointcloud)
    def forward(self, mesh_vertices):
        # TODO: make this more elegant
        if self.dis_points_encoder == "rgan":
            mesh_vertices = torch.transpose(mesh_vertices, 1, 2)

        logits = self.dis_net(mesh_vertices)

        return logits