
import torch
import pytorch3d
from pytorch3d.renderer import look_at_view_transform

from .semantic_discriminator_net import SemanticDiscriminatorNetwork
from utils import utils


class SemanticDiscriminatorLoss():


    def __init__(self, cfg, device):

        self.device = device
        self.semantic_discriminator = SemanticDiscriminatorNetwork(cfg, device)
        self.semantic_discriminator.to(device)

        self.cfg = cfg
        self.num_render = cfg['training']['semantic_dis_num_render']
        semantic_dis_weight_path = cfg["training"]["semantic_dis_weight_path"]
        # TODO: set weights
                
        # freeze weights
        for param in self.semantic_discriminator.parameters():
            param.requires_grad = False
    

    def compute_loss(self, mesh):
        self.semantic_discriminator.eval()

        azims = torch.linspace(0,360,self.num_render+2)[1:-1]
        elevs = torch.Tensor([45 for i in range(self.num_render)])
        dists = torch.ones(self.num_render) * 1.9
        R, T = look_at_view_transform(dists, elevs, azims)
        meshes = mesh.extend(self.num_render)
        renders = utils.render_mesh(meshes, R, T, self.device, img_size=224, silhouette=False)
        loss = self.semantic_discriminator(renders[...,:3].permute(0,3,1,2))
        loss = torch.mean(loss)
        return loss, renders