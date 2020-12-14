import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch3d


class DeformationNetworkGcnHybrid(nn.Module):
    # based on https://arxiv.org/pdf/2006.07029.pdf on page 10
    # bn ordering based on https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    def __init__(self, cfg, device):
        super().__init__()

        self.device = device
        self.activation = nn.LeakyReLU()
        self.hidden_dim = 256
        indi_fc_dims_1 = [3, 512, 512, self.hidden_dim, self.hidden_dim]
        indi_fc_dims_3 = [self.hidden_dim, 128, 64, 32, 3]

        indi_mlp_1_modules = []
        for i in range(len(indi_fc_dims_1)-1):
            indi_mlp_1_modules.append(nn.Linear(indi_fc_dims_1[i], indi_fc_dims_1[i+1]))
            indi_mlp_1_modules.append(self.activation)
        self.indi_mlp_1 = nn.Sequential(*indi_mlp_1_modules)

        self.gconvs = nn.ModuleList()
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=3+self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))
        self.gconvs.append(pytorch3d.ops.GraphConv(input_dim=self.hidden_dim, output_dim=self.hidden_dim))

        indi_mlp_3_modules = []
        for i in range(len(indi_fc_dims_3)-1):
            indi_mlp_3_modules.append(nn.Linear(indi_fc_dims_3[i], indi_fc_dims_3[i+1]))
            if i < len(indi_fc_dims_3)-2:
                indi_mlp_3_modules.append(self.activation)
        self.indi_mlp_3 = nn.Sequential(*indi_mlp_3_modules)


    def forward(self, input_batch):

        #images = input_batch["image"].to(self.device)
        #poses = input_batch["pose"].to(self.device)
        original_meshes = input_batch["mesh"].to(self.device)
        input_points = original_meshes.verts_padded() # [b, n_points, 3]
        batch_size = input_points.shape[0]

        # passing into first indepent fc layer
        indi_1 = self.indi_mlp_1(input_points) #[b, n_points, hidden_dim]

        # passing into gcn layer
        # for homogenius n_points, reshaping verts_padded with (-1, feat_dim) is the same as verts_packed
        batch_vertex_features = torch.cat([torch.reshape(indi_1, (-1,self.hidden_dim)), original_meshes.verts_packed()], dim=1) #[b*n_points, hidden_dim+3]
        for i in range(len(self.gconvs)):
            batch_vertex_features = F.relu(self.gconvs[i](batch_vertex_features, original_meshes.edges_packed()))
            # TODO: also add original coordinate?
        batch_vertex_features = torch.reshape(batch_vertex_features, (batch_size, -1, self.hidden_dim)) #[b*n_points, hidden_dim] -> #[b, n_points, hidden_dim]

        # passing into second indepent fc layer to get deltas
        deltas = self.indi_mlp_3(batch_vertex_features)  # [b, n_points, 3]

        # TODO: add asym conf
        deltas = torch.reshape(deltas, (-1,3)) # [b*n_points, 3]
        return deltas


