
import torch
import torch.nn as nn
from .pointnet import SimplePointnet, ResnetPointnet, ResnetPointnetExtended
from .resnet import Resnet18, Resnet34
from utils import network_utils

class DeformationNetworkMinimal(nn.Module):

    def __init__(self, cfg, num_vertices, device):
        super().__init__()
        self.device = device
        self.num_vertices = num_vertices
        point_encoder_name = cfg['model']['point_encoder']
        image_encoder_name = cfg['model']['image_encoder']
        deformation_decoder_name = cfg['model']['deformation_decoder']
        pointnet_encoding_dim = cfg['model']['latent_dim_pointnet']
        resnet_encoding_dim = cfg['model']['latent_dim_resnet']
        decoder_dim = cfg['model']['decoder_dim']

        if point_encoder_name == "ResnetPointnet":
            self.pointnet_encoder = ResnetPointnet(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)
        elif point_encoder_name == "ResnetPointnetExtended":
            self.pointnet_encoder = ResnetPointnetExtended(c_dim=pointnet_encoding_dim, hidden_dim=pointnet_encoding_dim)
        else:
            raise ValueError("Point encoder name not recognized.")
        
        if image_encoder_name == "Resnet18":
            self.resnet_encoder = Resnet18(c_dim=resnet_encoding_dim)
        elif image_encoder_name == "Resnet34":
            self.resnet_encoder = Resnet34(c_dim=resnet_encoding_dim)
        else: 
            raise ValueError("Image encoder name not recognized.")

        if deformation_decoder_name == "FCStandard":
            self.deform_net = nn.Sequential(
                #nn.Linear(pointnet_encoding_dim+resnet_encoding_dim+3, decoder_dim),
                nn.Linear(pointnet_encoding_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, num_vertices),
                nn.ReLU(),
                nn.Linear(num_vertices, num_vertices*3)
            )
        elif deformation_decoder_name == "FC_BN": 
            self.deform_net = nn.Sequential(
                nn.Linear(pointnet_encoding_dim+resnet_encoding_dim+3, decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim),
                nn.BatchNorm1d(decoder_dim),
                nn.ReLU(),
                nn.Linear(decoder_dim, num_vertices),
                nn.ReLU(),
                nn.Linear(num_vertices, num_vertices*3)
            )
        else:
            raise ValueError("Deformation decoder name not recognized.")

        if cfg["semantic_dis_training"]["gen_small_weights_init"]:
            # TODO: should this also be applied to pointnet encoder?
            self.deform_net.apply(network_utils.weights_init_normal)

    
    def forward(self, input_batch):
        '''
        Args (b is batch size):
            pose (tensor): a b x 3 tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a b x 3 x 224 x 224 image which is segmented.
            mesh_vertices (tensor): a b x num_vertices x 3 tensor of vertices (ie, a pointcloud)
        '''

        mesh_vertices = input_batch["mesh_verts"].to(self.device)
        #image = input_batch["image"].to(self.device)
        #pose = input_batch["pose"].to(self.device)

        if mesh_vertices.shape[1] != self.num_vertices:
            raise ValueError("num_vertices does not match number of vertices of input mesh")
        
        #image_encoding = self.resnet_encoder(image)
        verts_encoding = self.pointnet_encoder(mesh_vertices)
        #combined_encoding = torch.cat((pose, image_encoding, verts_encoding), 1)

        delta_v = self.deform_net(verts_encoding)
        #delta_v = self.deform_net(combined_encoding)
        return delta_v

