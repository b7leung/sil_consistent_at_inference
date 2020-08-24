
import torch
import pytorch3d
from pytorch3d.renderer import look_at_view_transform

from .semantic_discriminator_net import SemanticDiscriminatorNetwork
from utils import utils

class SemanticDiscriminatorLoss():

    def __init__(self, cfg, device):

        self.device = device
        self.cfg = cfg
        self.num_render = cfg['training']['semantic_dis_num_render']

        self.semantic_discriminator = SemanticDiscriminatorNetwork(cfg)
        self.semantic_discriminator.load_state_dict(torch.load(cfg["training"]["semantic_dis_weight_path"]))
        self.semantic_discriminator.to(device)
        for param in self.semantic_discriminator.parameters():
            param.requires_grad = False


    def compute_loss(self, mesh):
        # need to make sure this matches render settings for discriminator training set
        # TODO: alternatively, randomize the angles each time?
        # 0.,  45.,  90., 135., 180., 225., 270., 315. 
        azims = torch.linspace(0, 360, self.num_render+1)[:-1]
        elevs = torch.Tensor([25 for i in range(self.num_render)])
        dists = torch.ones(self.num_render) * 1.7
        R, T = look_at_view_transform(dists, elevs, azims)

        meshes = mesh.extend(self.num_render)
        renders = utils.render_mesh(meshes, R, T, self.device, img_size=224, silhouette=True)
        # converting from [num_render, 224, 224, 4] silhouette render (only channel 4 has info) 
        # to [num_render, 224, 224, 3] rgb image (black/white)
        renders_binary_rgb = torch.unsqueeze(renders[...,3], 3).repeat(1,1,1,3)

        self.semantic_discriminator.eval()
        loss = torch.sigmoid(self.semantic_discriminator(renders_binary_rgb.permute(0,3,1,2)))
        loss = torch.mean(loss)

        return loss, renders_binary_rgb
