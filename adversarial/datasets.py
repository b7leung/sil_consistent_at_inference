import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from glob import glob
import pickle
import numpy as np
from torch.utils.data.dataloader import default_collate
import pytorch3d.datasets
import pytorch3d.structures

from utils import utils, network_utils


# input should be a list of dicts
def gen_data_collate(batch):
    elem = batch[0]
    out = {}
    for key in elem:
        if isinstance(elem[key], pytorch3d.structures.meshes.Meshes):
            out[key] = pytorch3d.structures.join_meshes_as_batch([d[key] for d in batch], include_textures=False)
            #out[key] = pytorch3d.datasets.utils.collate_batched_meshes([{d[key] for d in batch])
        else:
            out[key] = default_collate([d[key] for d in batch])

    #out =  {key: default_collate([d[key] for d in batch]) for key in elem}
    return out


class GenerationDataset(Dataset):
    """Dataset used for mesh deformation generator. Each datum is a tuple {mesh vertices, input img, predicted pose}"""

    def __init__(self, cfg, device):
        self.cpu_device = torch.device("cpu")
        self.input_dir_img = cfg['dataset']['input_dir_img']
        self.input_dir_mesh = cfg['dataset']['input_dir_mesh']
        self.input_dir_pose = cfg['semantic_dis_training']['input_dir_pose']
        self.cached_pred_poses = pickle.load(open(self.input_dir_pose, "rb"))
        self.dataset_meshes_list_path = cfg['semantic_dis_training']['dataset_meshes_list_path']

        with open (self.dataset_meshes_list_path, 'r') as f:
            self.dataset_meshes_list = f.read().split('\n')
        
    
    def __len__(self):
        return len(self.dataset_meshes_list)


    def __getitem__(self, idx):
        '''
            pose (tensor): a 3 element tensor specifying distance, elevation, azimuth (in that order)
            image (tensor): a 3 x 224 x 224 image which is segmented. 
            mesh_vertices (tensor): a num_vertices x 3 tensor of vertices (ie, a pointcloud). 
        '''
        data = {}

        instance_name = self.dataset_meshes_list[idx]
        data["instance_name"] = instance_name

        curr_obj_path = os.path.join(self.input_dir_mesh, instance_name+".obj")
        with torch.no_grad():
            # load mesh to cpu
            # This can probably be done in a more efficent way (load a batch of meshes?)
            mesh = utils.load_untextured_mesh(curr_obj_path, self.cpu_device)
        data["mesh"] = mesh
            
        data["mesh_verts"] = mesh.verts_packed()

        curr_img_path = os.path.join(self.input_dir_img, instance_name+".png")
        # TODO: use transforms.ToTensor() instead? (may fix mixed memory formats warning)
        rgba_image = np.asarray(Image.open(curr_img_path))
        image = rgba_image[:,:,:3]
        data["image"] = torch.tensor(image/255, dtype=torch.float).permute(2,0,1)

        data["mask"] = torch.tensor(rgba_image[:,:,3] > 0, dtype=torch.float)

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
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
    
    
    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        data = self.img_transforms(Image.open(self.real_image_paths[idx]).convert("RGB"))
        return data