
import torch
import torch.nn as nn
from .pointnet import SimplePointnet, ResnetPointnet
from .resnet import Resnet18
from utils import network_utils

class PointwiseDeformationNetwork(nn.Module):

    def __init__(self, cfg, num_vertices, device):
        super().__init__()

        pointnet_encoding_dim = cfg['model']['latent_dim_pointnet']
        resnet_encoding_dim = cfg['model']['latent_dim_resnet']

        self.pointnet_encoder = ResnetPointnet(c_dim=pointnet_encoding_dim, 
                                               hidden_dim=pointnet_encoding_dim)
        self.resnet_encoder = Resnet18(c_dim=resnet_encoding_dim)

        self.point_deform_net = nn.Sequential(
            nn.Linear(3+pointnet_encoding_dim+resnet_encoding_dim+3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )
        #self.point_deform_net.apply(network_utils.weights_init_normal)
        self.num_vertices = num_vertices

    
    def forward(self, pose, image, mesh_vertices):
        '''
        Args (b is batch size):
            pose (tensor): a b x 3 tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a b x 3 x 224 x 224 image which is segmented.
            mesh_vertices (tensor): a b x num_vertices x 3 tensor of vertices (ie, a pointcloud)
        '''
        if mesh_vertices.shape[1] != self.num_vertices:
            raise ValueError("num_vertices does not match number of vertices of input mesh")
        
        image_encoding = self.resnet_encoder(image)
        verts_encoding = self.pointnet_encoder(mesh_vertices)
        combined_encoding = torch.cat((pose, image_encoding, verts_encoding), 1)

        repeated_combined_encoding = combined_encoding.repeat(self.num_vertices,1)
        raw_point_features = torch.cat((mesh_vertices.squeeze(0), repeated_combined_encoding), 1)

        delta_v = self.point_deform_net(raw_point_features)
        return delta_v

