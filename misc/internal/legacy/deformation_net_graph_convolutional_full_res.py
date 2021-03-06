
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d

from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended
from .resnet import Resnet18, Resnet34
from .resnet_backbone import build_backbone
from utils import network_utils
from utils import coords

# based on https://github.com/facebookresearch/meshrcnn/blob/89b59e6df2eb09b8798eae16e204f75bb8dc92a7/shapenet/modeling/heads/mesh_head.py
class DeformationNetworkGraphConvolutionalFullRes(nn.Module):

    def __init__(self, cfg, num_vertices, device):
        super().__init__()
        self.device = device
        self.num_vertices = num_vertices
        hidden_dim = 128

        resnet_encoding_dim = 256
        self.resnet_encoder = Resnet18(c_dim=resnet_encoding_dim)

        self.backbone, self.feat_dims = build_backbone("resnet50")
        img_feat_dim = sum(self.feat_dims)
        self.bottleneck = nn.Linear(img_feat_dim, hidden_dim)

        self.gconvs = nn.ModuleList()
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=3+resnet_encoding_dim+hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=hidden_dim, output_dim=hidden_dim))

        self.vert_offset = nn.Linear(hidden_dim, 3)


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

        # returns 4 feature maps from image, of size [1, 256, 56, 56], [1, 512, 28, 28], [1, 1024, 14, 14], [1, 2048, 7, 7]
        feat_maps = self.backbone(images)

        # aligning and normalizing vertex positions so (-1,-1) is the top left, (1,1) is the bottom right relative to the feature map
        verts_padded = mesh_batch.verts_padded()
        aligned_verts_padded = coords.align_and_normalize_verts(verts_padded, poses, self.device)

        # computing vert_align features
        vert_align_feats = pytorch3d.ops.vert_align(feat_maps, aligned_verts_padded, return_packed=True)
        vert_align_feats = F.relu(self.bottleneck(vert_align_feats))

        # appending original cordinates to vert_align features
        # not sure if original vertices or aligned/normalized vertices should be appended
        batch_vertex_features = torch.cat([vert_align_feats, mesh_batch.verts_packed()], dim=1)

        # appending resnet encoding to each vertex
        image_encodings = self.resnet_encoder(images)
        batch_vertex_features = torch.cat([batch_vertex_features, torch.repeat_interleave(image_encodings, self.num_vertices, dim=0)], 1)

        for i in range(len(self.gconvs)):
            batch_vertex_features = F.relu(self.gconvs[i](batch_vertex_features, mesh_batch.edges_packed()))
            # TODO: also add original coordinate?

        delta_v = (self.vert_offset(batch_vertex_features))
        #delta_v = torch.tanh(self.vert_offset(batch_vertex_features))

        return delta_v



#class MeshRefinementStage(nn.Module):
