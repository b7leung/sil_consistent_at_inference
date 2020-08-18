
import torch
import torch.nn as nn
from .resnet import Resnet18

class SemanticDiscriminatorNetwork(nn.Module):

    def __init__(self, cfg, device):
        super().__init__()

        resnet_encoding_dim = cfg['model']['latent_dim_resnet']
        self.resnet_encoder = Resnet18(c_dim=resnet_encoding_dim)

        self.linear_layers = nn.Sequential(
            nn.Linear(resnet_encoding_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    # assumes a batch of 3x224x224 image in [0,1]
    def forward(self, image):

        image_encoding = self.resnet_encoder(image)
        binary_classification_logits = self.linear_layers(image_encoding)
        return binary_classification_logits