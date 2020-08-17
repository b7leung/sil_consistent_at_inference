import pprint

import torch
from torch.nn import functional as F
import torch.optim as optim
import pytorch3d.structures
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader
)
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from tqdm import tqdm
import pandas as pd

from utils import utils, network_utils
from deformation.deformation_net import DeformationNetwork
import deformation.losses as def_losses


class MeshRefiner():

    def __init__(self, cfg_yaml_path, device):
        self.cfg = utils.load_config(cfg_yaml_path)
        self.device = device

        self.num_iterations = self.cfg["training"]["num_iterations"]
        self.img_sym_num_azim = self.cfg["training"]["img_sym_num_azim"]

        self.sil_lam = self.cfg["training"]["sil_lam"]
        self.l2_lam = self.cfg["training"]["l2_lam"]
        self.lap_lam = self.cfg["training"]["lap_lam"]
        self.normals_lam= self.cfg["training"]["normals_lam"]
        self.img_sym_lam= self.cfg["training"]["img_sym_lam"]
        self.vertex_sym_lam= self.cfg["training"]["vertex_sym_lam"]


    # given a mesh, mask, and pose, solves an optimization problem which encourages
    # silhouette consistency on the mask at the given pose.
    # record_intermediate will return a list of meshes
    # TODO: fix mesh (currently, needs to already be in device)
    def refine_mesh(self, mesh, rgba_image, pred_dist, pred_elev, pred_azim, record_intermediate=False):
        '''
        Args:
        pred_dist (int)
        pred_elev (int)
        pred_azim (int)
        rgba_image (np int array, 224 x 224 x 4, rgba, 0-255)
        '''

        # prep inputs used during training

        image = rgba_image[:,:,:3]
        image_in = torch.unsqueeze(torch.tensor(image/255, dtype=torch.float).permute(2,0,1),0).to(self.device)
        mask = rgba_image[:,:,3] > 0
        mask_gt = torch.tensor(mask, dtype=torch.float).to(self.device)
        pose_in = torch.unsqueeze(torch.tensor([pred_dist, pred_elev, pred_azim]),0).to(self.device)
        verts_in = torch.unsqueeze(mesh.verts_packed(),0).to(self.device)

        R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim) 
        num_vertices = mesh.verts_packed().shape[0]
        zero_deformation_tensor = torch.zeros((num_vertices, 3)).to(self.device)

        # prep network & optimizer
        deform_net = DeformationNetwork(self.cfg, num_vertices, self.device)
        deform_net.to(self.device)
        optimizer = optim.Adam(deform_net.parameters(), lr=self.cfg["training"]["learning_rate"])

        # optimizing  
        loss_info = pd.DataFrame()
        deformed_meshes = []
        for i in tqdm(range(self.num_iterations)):
            deform_net.train()
            optimizer.zero_grad()
            
            # computing mesh deformation & its render at the input pose
            delta_v = deform_net(pose_in, image_in, verts_in)
            delta_v = delta_v.reshape((-1,3))
            deformed_mesh = mesh.offset_verts(delta_v)
            rendered_deformed_mesh = utils.render_mesh(deformed_mesh, R, T, self.device, img_size=224, silhouette=True)

            # computing losses
            l2_loss = F.mse_loss(delta_v, zero_deformation_tensor)
            lap_smoothness_loss = mesh_laplacian_smoothing(deformed_mesh)
            normal_consistency_loss = mesh_normal_consistency(deformed_mesh)

            sil_loss = F.binary_cross_entropy(rendered_deformed_mesh[0, :,:, 3], mask_gt)

            sym_plane_normal = [0,0,1]
            if self.img_sym_lam > 0:
                img_sym_loss, _ = def_losses.image_symmetry_loss(deformed_mesh, sym_plane_normal, self.img_sym_num_azim, self.device)
            else:
                img_sym_loss = torch.tensor(0).to(self.device)
            if self.vertex_sym_lam > 0:
                vertex_sym_loss = def_losses.vertex_symmetry_loss_fast(deformed_mesh, sym_plane_normal, self.device)
            else:
                vertex_sym_loss = torch.tensor(0).to(self.device)

            # optimization step on weighted losses
            total_loss = (sil_loss*self.sil_lam + l2_loss*self.l2_lam + lap_smoothness_loss*self.lap_lam +
                          normal_consistency_loss*self.normals_lam + img_sym_loss*self.img_sym_lam + 
                          vertex_sym_loss*self.vertex_sym_lam)
            total_loss.backward()
            optimizer.step()

            # saving info
            iter_loss_info = {"iter":i, "sil_loss": sil_loss.item(), "l2_loss": l2_loss.item(), 
                              "lap_smoothness_loss":lap_smoothness_loss.item(),
                              "normal_consistency_loss": normal_consistency_loss.item(),"img_sym_loss": img_sym_loss.item(),
                              "vertex_sym_loss": vertex_sym_loss.item(), 
                              "total_loss": total_loss.item()}
            loss_info = loss_info.append(iter_loss_info, ignore_index = True)
            if record_intermediate and (i % 100 == 0 or i == self.num_iterations-1):
                print(i)
                deformed_meshes.append(deformed_mesh)

        if record_intermediate:
            return deformed_meshes, loss_info
        else:
            return deformed_mesh, loss_info
            