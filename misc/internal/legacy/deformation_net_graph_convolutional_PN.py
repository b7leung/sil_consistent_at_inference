
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d

from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended
from .resnet import Resnet18, Resnet34
from utils import network_utils

class DeformationNetworkGraphConvolutionalPN(nn.Module):

    def __init__(self, cfg, num_vertices, device):
        super().__init__()
        self.device = device
        self.num_vertices = num_vertices

        pointnet_encoding_dim = 256
        self.pointnet_encoder = ResnetPointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)

        gconvs_hidden_dim = 256
        self.gconvs = nn.ModuleList()
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=3+pointnet_encoding_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=gconvs_hidden_dim, output_dim=gconvs_hidden_dim))

        vert_offset_hidden_dim = 512
        self.vert_offset = nn.Sequential(
            nn.Linear(gconvs_hidden_dim, vert_offset_hidden_dim),
            nn.ReLU(),
            nn.Linear(vert_offset_hidden_dim, vert_offset_hidden_dim),
            nn.ReLU(),
            nn.Linear(vert_offset_hidden_dim, vert_offset_hidden_dim),
            nn.ReLU(),
            nn.Linear(vert_offset_hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3))

    
    def forward(self, input_batch):
        '''
        Args (b is batch size):
            pose (tensor): a b x 3 tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a b x 3 x 224 x 224 image which is segmented.
            mesh_vertices (tensor): a b x num_vertices x 3 tensor of vertices (ie, a pointcloud)
        '''
        # based on https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/modeling/heads/mesh_head.py
        # TODO: make sure this also works for batched cases

        mesh_batch = input_batch["mesh"].to(self.device)
        mesh_vertices = input_batch["mesh_verts"].to(self.device)

        # appending pointnet latent vector to each vertex
        verts_encodings = self.pointnet_encoder(mesh_vertices)
        mesh_verts_packed = mesh_batch.verts_packed()
        batch_vertex_features = torch.cat([mesh_verts_packed, torch.repeat_interleave(verts_encodings, self.num_vertices, dim=0)], 1)

        # Graph conv layers
        for i in range(len(self.gconvs)):
            batch_vertex_features = F.relu(self.gconvs[i](batch_vertex_features, mesh_batch.edges_packed()))

        # FC layers 
        delta_v = (self.vert_offset(batch_vertex_features))

        return delta_v

