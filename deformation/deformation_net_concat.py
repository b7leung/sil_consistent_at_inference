
import torch
import torch.nn as nn
from .pointnet import SimplePointnet, ResnetPointnet
from .resnet import Resnet18
from utils import network_utils

class DeformationNetworkConcat(nn.Module):

    def __init__(self, cfg, num_vertices, device):
        super().__init__()

        pointnet_encoding_dim = cfg['model']['latent_dim_pointnet']
        resnet_encoding_dim = cfg['model']['latent_dim_resnet']

        self.pointnet_encoder = SimplePointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)
        self.resnet_encoder = Resnet18(c_dim=resnet_encoding_dim)
        self.deform_net = nn.Sequential(
            nn.Linear((num_vertices*3)+(pointnet_encoding_dim+resnet_encoding_dim+3), 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_vertices*3)
        )
        #self.deform_net.apply(network_utils.weights_init_normal)
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

        mesh_vertices_combined_encoding = torch.cat((mesh_vertices.reshape(1,-1), combined_encoding), 1)

        delta_v = self.deform_net(mesh_vertices_combined_encoding)
        return delta_v

