import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from glob import glob
import pickle
import numpy as np

from utils import utils, network_utils


class GenerationDataset(Dataset):
    """Dataset used for mesh deformation generator. Each datum is a tuple {mesh vertices, input img, predicted pose}"""
    # mesh is automatally sent to the device

    def __init__(self, cfg, device):
        self.device = device
        self.input_dir_img = cfg['dataset']['input_dir_img']
        self.input_dir_mesh = cfg['dataset']['input_dir_mesh']
        self.input_dir_pose = cfg['semantic_dis_training']['input_dir_pose']
        self.cached_pred_poses = pickle.load(open(self.input_dir_pose, "rb"))
        self.dataset_meshes_list_path = cfg['semantic_dis_training']['dataset_meshes_list_path']

        with open (self.dataset_meshes_list_path, 'r') as f:
            self.dataset_meshes_list = f.read().split('\n')
        
    
    def __len__(self):
        return len(self.dataset_meshes_list)


        '''
        Args:
            pose (tensor): a 3 element tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a 3 x 224 x 224 image which is segmented. 
            mesh_vertices (tensor): a num_vertices x 3 tensor of vertices (ie, a pointcloud). 
        '''
    def __getitem__(self, idx):
        data = {}

        instance_name = self.dataset_meshes_list[idx]
        data["instance_name"] = instance_name

        curr_obj_path = os.path.join(self.input_dir_mesh, instance_name+".obj")
        with torch.no_grad():
            # This can probably be done in a more efficent way (load a batch of meshes?)
            mesh = utils.load_untextured_mesh(curr_obj_path, self.device)
        data["mesh_verts"] = mesh.verts_packed()

        curr_img_path = os.path.join(self.input_dir_img, instance_name+".png")
        # TODO: use transforms.ToTensor() instead?
        rgba_image = np.asarray(Image.open(curr_img_path))
        image = rgba_image[:,:,:3]
        data["image"] = torch.tensor(image/255, dtype=torch.float).permute(2,0,1)

        pred_dist = self.cached_pred_poses[instance_name]['dist']
        pred_elev = self.cached_pred_poses[instance_name]['elev']
        pred_azim = self.cached_pred_poses[instance_name]['azim']
        data["pose"] = torch.tensor([pred_dist, pred_elev, pred_azim])

        return data



class ShapenetRendersDataset(Dataset):
    """Dataset used for shapenet renders. Each datum is a random shapenet render"""
    def __init__(self, cfg):

        self.real_render_dir = cfg["semantic_dis_training"]["real_dataset_dir"]

        self.real_image_paths = glob(os.path.join(self.real_render_dir, "*.jpg"))
        
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    
    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        data = self.img_transforms(Image.open(self.real_image_paths[idx]).convert("RGB"))
        return data