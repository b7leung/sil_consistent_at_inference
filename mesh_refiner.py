import pprint

import torch
import torch.optim as optim
from pytorch3d.renderer import look_at_view_transform
from tqdm.autonotebook import tqdm
import pandas as pd

from deformation.deformation_net import DeformationNetwork
from deformation.deformation_net_graph_convolutional import DeformationNetworkGraphConvolutional
from deformation.deformation_net_graph_convolutional_PN import DeformationNetworkGraphConvolutionalPN
from deformation.deformation_net_graph_convolutional_full import DeformationNetworkGraphConvolutionalFull
from deformation.semantic_discriminator_net_renders import RendersSemanticDiscriminatorNetwork
from deformation.semantic_discriminator_net_points import PointsSemanticDiscriminatorNetwork
from deformation.forward_pass import batched_forward_pass


class MeshRefiner():

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        self.num_iterations = self.cfg["refinement"]["num_iterations"]
        self.lr = self.cfg["refinement"]["learning_rate"]
        self.gen_weight_path = self.cfg["semantic_dis_training"]["gen_weight_path"]
        self.dis_weight_path = self.cfg["semantic_dis_training"]["dis_weight_path"]

        self.img_sym_num_azim = self.cfg["training"]["img_sym_num_azim"]
        self.sil_lam = self.cfg["training"]["sil_lam"]
        self.l2_lam = self.cfg["training"]["l2_lam"]
        self.lap_lam = self.cfg["training"]["lap_smoothness_lam"]
        self.normals_lam = self.cfg["training"]["normal_consistency_lam"]
        self.img_sym_lam = self.cfg["training"]["img_sym_lam"]
        self.vertex_sym_lam = self.cfg["training"]["vertex_sym_lam"]
        self.semantic_dis_lam = self.cfg["training"]["semantic_dis_lam"]


    def setup_generator(self):
        # setting generator deformation network
        deform_net_type = self.cfg["semantic_dis_training"]["deform_net_type"]
        if deform_net_type == "pointnet":
            deform_net = DeformationNetwork(self.cfg, self.cfg["semantic_dis_training"]["mesh_num_verts"], self.device)
        elif deform_net_type == "gcn":
            deform_net = DeformationNetworkGraphConvolutional(self.cfg, self.cfg["semantic_dis_training"]["mesh_num_verts"], self.device)
        elif deform_net_type == "gcn_full":
            deform_net = DeformationNetworkGraphConvolutionalFull(self.cfg, self.cfg["semantic_dis_training"]["mesh_num_verts"], self.device)
        elif deform_net_type == "gcn_pn":
            deform_net = DeformationNetworkGraphConvolutionalPN(self.cfg, self.cfg["semantic_dis_training"]["mesh_num_verts"], self.device)
        else:
            raise ValueError("generator deform net type not recognized")
        if self.gen_weight_path != "":
            deform_net.load_state_dict(torch.load(self.gen_weight_path))
        deform_net.to(self.device)

        return deform_net


    def setup_discriminator(self):
        dis_type = self.cfg['semantic_dis_training']['dis_type']
        if dis_type == "renders":
            semantic_dis_net = RendersSemanticDiscriminatorNetwork(self.cfg)
        elif dis_type == "points":
            semantic_dis_net = PointsSemanticDiscriminatorNetwork(self.cfg)
        else:
            raise ValueError("dis_type must be renders or pointnet")
        if self.dis_weight_path != "":
            semantic_dis_net.load_state_dict(torch.load(self.dis_weight_path))
        semantic_dis_net.to(self.device)
        for param in semantic_dis_net.parameters():
            param.requires_grad = False

        return semantic_dis_net


    # given a mesh, mask, and pose, solves an optimization problem which encourages
    # silhouette consistency on the mask at the given pose.
    # record_intermediate will return a list of meshes
    # rgba_image (np int array, 224 x 224 x 4, rgba, 0-255)
    # TODO: fix mesh (currently, needs to already be in device)
    def refine_mesh(self, mesh, rgba_image, pred_dist, pred_elev, pred_azim, record_intermediate=False):

        # prep inputs used during training (making a batch of size one)
        image = rgba_image[:,:,:3]
        image_in = torch.unsqueeze(torch.tensor(image/255, dtype=torch.float).permute(2,0,1),0).to(self.device)
        mask = rgba_image[:,:,3] > 0
        mask_gt = torch.unsqueeze(torch.tensor(mask, dtype=torch.float), 0).to(self.device)
        pose_in = torch.unsqueeze(torch.tensor([pred_dist, pred_elev, pred_azim]),0).to(self.device)
        verts_in = torch.unsqueeze(mesh.verts_packed(),0).to(self.device)
        deform_net_input = {"mesh_verts": verts_in, "image":image_in, "pose": pose_in, "mesh": mesh, "mask": mask_gt}

        # prep networks & optimizer
        semantic_dis_net = self.setup_discriminator()
        deform_net = self.setup_generator()
        optimizer = optim.Adam(deform_net.parameters(), lr=self.lr)

        # optimizing  
        loss_info = pd.DataFrame()
        deformed_meshes = []
        lowest_loss = None
        best_deformed_mesh = None

        for i in tqdm(range(self.num_iterations)):
            deform_net.train()
            semantic_dis_net.eval()
            optimizer.zero_grad()

            loss_dict, deformed_mesh = batched_forward_pass(self.cfg, self.device, deform_net, semantic_dis_net, deform_net_input, compute_losses=True)

            # optimization step on weighted losses
            total_loss = sum([loss_dict[loss_name] * self.cfg['training'][loss_name.replace("loss", "lam")] for loss_name in loss_dict])
            total_loss.backward()
            optimizer.step()

            # saving info
            curr_train_info = {"iteration": i, "total_loss": total_loss.item()}
            curr_train_info = {**curr_train_info, **{loss_name:loss_dict[loss_name].item() for loss_name in loss_dict}}
            loss_info = loss_info.append(curr_train_info, ignore_index=True)

            #if record_intermediate and (i % 100 == 0 or i == self.num_iterations-1):
            if record_intermediate:
                deformed_meshes.append(deformed_mesh.detach().cpu())

            if lowest_loss is None or total_loss.item() < lowest_loss:
                lowest_loss = total_loss.item()
                best_deformed_mesh = deformed_mesh

        if record_intermediate:
            return deformed_meshes, loss_info
        else:
            return best_deformed_mesh, loss_info
