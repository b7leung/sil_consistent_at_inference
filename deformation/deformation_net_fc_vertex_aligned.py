
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d

from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended
from .resnet import Resnet18, Resnet34
from .resnet_backbone import build_backbone
from utils import coords

# based on https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/modeling/heads/mesh_head.py
class DeformationNetworkFcVertexAligned(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.device = device
        self.asym = cfg["training"]["vertex_asym"]
        self.num_vertices = cfg["semantic_dis_training"]["mesh_num_verts"]

        hidden_dim = 256
        self.backbone, self.feat_dims = build_backbone("resnet50")
        img_feat_dim = sum(self.feat_dims)
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.activation = nn.LeakyReLU()
        self.vert_offset = nn.Sequential(
            nn.Linear((3+hidden_dim)*self.num_vertices, 3*self.num_vertices),
            self.activation,
            nn.Linear(3*self.num_vertices, 3*self.num_vertices),
            self.activation,
            nn.Linear(3*self.num_vertices, 3*self.num_vertices),
            self.activation,
            nn.Linear(3*self.num_vertices, 3*self.num_vertices),
            self.activation,
            nn.Linear(3*self.num_vertices, 3*self.num_vertices),
            self.activation,
            nn.Linear(3*self.num_vertices, 3*self.num_vertices),
            self.activation,
            nn.Linear(3*self.num_vertices, 3*self.num_vertices)
        )

        self.asym_conf = nn.Sequential(
            nn.Linear(3+hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )


    def forward(self, input_batch):
        '''
        Args (b is batch size):
            pose (tensor): a b x 3 tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a b x 3 x 224 x 224 image which is segmented.
            mesh_vertices (tensor): a b x num_vertices x 3 tensor of vertices (ie, a pointcloud)
        '''
        
        images = input_batch["image"].to(self.device)
        mesh_batch = input_batch["mesh"].to(self.device)
        poses = input_batch["pose"].to(self.device)
        batch_size = poses.shape[0]

        # returns 4 feature maps from image, of size [1, 256, 56, 56], [1, 512, 28, 28], [1, 1024, 14, 14], [1, 2048, 7, 7]
        feat_maps = self.backbone(images)

        # aligning and normalizing vertex positions so (-1,-1) is the top left, (1,1) is the bottom right relative to the feature map
        verts_padded = mesh_batch.verts_padded()
        aligned_verts_padded = coords.align_and_normalize_verts(verts_padded, poses, self.device)

        # computing vert_align features
        vert_align_feats = pytorch3d.ops.vert_align(feat_maps, aligned_verts_padded, return_packed=False)
        vert_align_feats = self.activation(self.bottleneck(vert_align_feats))

        # appending original cordinates to vert_align features
        # not sure if original vertices or aligned/normalized vertices should be appended
        batch_vertex_features = torch.cat([vert_align_feats, mesh_batch.verts_padded()], dim=2)
        batch_vertex_features = torch.reshape(batch_vertex_features, (batch_size, -1))
        delta_v = self.vert_offset(batch_vertex_features)
        delta_v = torch.reshape(delta_v, (-1,3))

        if self.asym:
            #asym_conf_scores = F.softplus(self.asym_conf(batch_vertex_features))
            #asym_conf_scores = F.relu(self.asym_conf(batch_vertex_features))
            asym_conf_scores = torch.sigmoid(self.asym_conf(batch_vertex_features))
            asym_conf_scores = torch.reshape(asym_conf_scores, (-1,1))

            return [delta_v, asym_conf_scores]
        else:
            return delta_v

