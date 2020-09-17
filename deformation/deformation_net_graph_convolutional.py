
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d

from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended
from .resnet import Resnet18, Resnet34
from utils import network_utils

class DeformationNetworkGraphConvolutional(nn.Module):

    def __init__(self, cfg, num_vertices, device):
        super().__init__()
        self.device = device
        self.num_vertices = num_vertices

        hidden_dim = 128
        self.gconvs = nn.ModuleList()
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=3, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))

        self.vert_offset = nn.Sequential(
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            #nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Linear(hidden_dim, 3))

    
    def forward(self, input_batch):
        '''
        Args (b is batch size):
            pose (tensor): a b x 3 tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a b x 3 x 224 x 224 image which is segmented.
            mesh_vertices (tensor): a b x num_vertices x 3 tensor of vertices (ie, a pointcloud)
        '''
        # based on https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/modeling/heads/mesh_head.py
        # TODO: need to find a way to incorporate silhouette/image data back into network

        mesh_batch = input_batch["mesh"].to(self.device)

        batch_vertex_features = mesh_batch.verts_packed()
        for i in range(len(self.gconvs)):
            batch_vertex_features = F.relu(self.gconvs[i](batch_vertex_features, mesh_batch.edges_packed()))
            # TODO: also add original coordinate?

        delta_v = torch.tanh(self.vert_offset(batch_vertex_features))
        #delta_v = (self.vert_offset(batch_vertex_features))

        return delta_v

