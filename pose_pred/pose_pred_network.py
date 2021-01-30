import torch
import torch.nn as nn
from deformation.resnet import Resnet18, Resnet34


class PoseNetwork(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()
        self.device = device

        resnet_encoding_dim = 1024
        self.resnet_encoder = Resnet34(c_dim=resnet_encoding_dim)

        self.activation = nn.ReLU()
        pose_pred_dims = [resnet_encoding_dim, 512, 512, 256, 64, 32, 1]

        pose_pred_layers_1 = []
        for i in range(len(pose_pred_dims)-1):
            pose_pred_layers_1.append(nn.Linear(pose_pred_dims[i], pose_pred_dims[i+1]))
            if i < len(pose_pred_dims)-2:
                pose_pred_layers_1.append(self.activation)
        self.pose_pred_mlp_1 = nn.Sequential(*pose_pred_layers_1)

        pose_pred_layers_2 = []
        for i in range(len(pose_pred_dims)-1):
            pose_pred_layers_2.append(nn.Linear(pose_pred_dims[i], pose_pred_dims[i+1]))
            if i < len(pose_pred_dims)-2:
                pose_pred_layers_2.append(self.activation)
        self.pose_pred_mlp_2 = nn.Sequential(*pose_pred_layers_2)

        pose_pred_layers_3 = []
        for i in range(len(pose_pred_dims)-1):
            pose_pred_layers_3.append(nn.Linear(pose_pred_dims[i], pose_pred_dims[i+1]))
            if i < len(pose_pred_dims)-2:
                pose_pred_layers_3.append(self.activation)
        self.pose_pred_mlp_3 = nn.Sequential(*pose_pred_layers_3)


    # outputs pose preds of order azim, elev, dist, [b, 3]
    def forward(self, input_batch):
        image = input_batch["image"].to(self.device)

        image_encoding = self.resnet_encoder(image)
        pred_poses_1 = self.pose_pred_mlp_1(image_encoding)
        pred_poses_2 = self.pose_pred_mlp_2(image_encoding)
        pred_poses_3 = self.pose_pred_mlp_3(image_encoding)

        pred_poses = torch.cat([pred_poses_1, pred_poses_2, pred_poses_3], dim=1)

        return pred_poses

