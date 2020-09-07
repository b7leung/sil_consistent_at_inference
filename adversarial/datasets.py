import os
from pathlib import Path

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
            out[key] = pytorch3d.structures.join_meshes_as_batch([d[key] for d in batch], include_textures=True)
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

        self.num_batches_gen_train = cfg["semantic_dis_training"]["num_batches_gen_train"]
        self.batch_size = cfg["semantic_dis_training"]["batch_size"]

        with open (self.dataset_meshes_list_path, 'r') as f:
            self.dataset_meshes_list = f.read().split('\n')
        
        if self.num_batches_gen_train != -1:
            self.dataset_meshes_list = self.dataset_meshes_list[:self.num_batches_gen_train * self.batch_size]

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    
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
        rgba_image = Image.open(curr_img_path)
        data["image"] = self.img_transforms(rgba_image.convert("RGB"))

        data["mask"] = torch.tensor(np.asarray(rgba_image)[:,:,3] > 0, dtype=torch.float)

        pred_dist = self.cached_pred_poses[instance_name]['dist']
        pred_elev = self.cached_pred_poses[instance_name]['elev']
        pred_azim = self.cached_pred_poses[instance_name]['azim']
        data["pose"] = torch.tensor([pred_dist, pred_elev, pred_azim])

        return data


dis_input_PIL_transforms= transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0),
    transforms.RandomPerspective(distortion_scale=0.1, p=0.25, interpolation=3, fill=0),
    transforms.RandomAffine(10, translate=(0.1,0.1), scale=(0.70,1.3), shear=[-7,7,-7,7], fillcolor=(0,0,0)),
    transforms.RandomHorizontalFlip(p=0.5),
])


class ShapenetRendersDataset(Dataset):
    """Dataset used for shapenet renders. Each datum is a random shapenet render"""
    def __init__(self, cfg):

        sil_dis_input = cfg["semantic_dis_training"]["sil_dis_input"]
        if sil_dis_input:
            real_dataset_dir = cfg["semantic_dis_training"]["real_dataset_dir_sil"]
        else:
            real_dataset_dir = cfg["semantic_dis_training"]["real_dataset_dir"]

        self.real_image_paths = glob(os.path.join(real_dataset_dir, "*.jpg"))
        input_img_size = cfg["semantic_dis_training"]["dis_input_size"]

        transform_dis_inputs = cfg["semantic_dis_training"]["transform_dis_inputs"]
        if transform_dis_inputs:
            self.img_transforms = transforms.Compose([
                transforms.Resize((input_img_size,input_img_size)),
                dis_input_PIL_transforms,
                transforms.ToTensor(),
            ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.Resize((input_img_size,input_img_size)),
                transforms.ToTensor(),
            ])
            
    
    def __len__(self):
        return len(self.real_image_paths)

    def __getitem__(self, idx):
        data = self.img_transforms(Image.open(self.real_image_paths[idx]).convert("RGB"))
        return data


class ShapenetPointsDataset(Dataset):

    def __init__(self, cfg):
        
        real_image_paths_cache_path = "caches/real_shapes_paths.p"

        if cfg["semantic_dis_training"]["recompute_cache"] or not os.path.exists(real_image_paths_cache_path):
            real_shapes_dir = cfg["semantic_dis_training"]["real_shapes_dir"]
            self.real_shapes_paths = list(Path(real_shapes_dir).rglob('model_watertight.obj'))
            pickle.dump(self.real_shapes_paths, open(real_image_paths_cache_path, 'wb'))
        else:
            self.real_shapes_paths = pickle.load(open(real_image_paths_cache_path, 'rb'))

        self.cpu_device = torch.device("cpu")
        self.mesh_num_verts = cfg["semantic_dis_training"]["mesh_num_verts"]
        

    def __len__(self):
        return len(self.real_shapes_paths)

    def __getitem__(self, idx):

        mesh = utils.load_untextured_mesh(self.real_shapes_paths[idx], self.cpu_device)
        points = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=self.mesh_num_verts)
        # by remove batch dimension
        points = torch.squeeze(points, dim=0)
        # TODO: normalize, either here or at dis loss func?

        return points