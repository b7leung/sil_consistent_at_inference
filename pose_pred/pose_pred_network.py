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
        
        pose_pred_layers = []
        for i in range(len(pose_pred_dims)-1):
            pose_pred_layers.append(nn.Linear(pose_pred_dims[i], pose_pred_dims[i+1]))
            if i < len(pose_pred_dims)-2:
                pose_pred_layers.append(self.activation)
        self.pose_pred_mlp = nn.Sequential(*pose_pred_layers)


    # outputs pose preds in shape [b, 3]
    def forward(self, input_batch):
        image = input_batch["image"].to(self.device)
        image_encoding = self.resnet_encoder(image)
        pred_poses = self.pose_pred_mlp(image_encoding)

        return pred_poses

