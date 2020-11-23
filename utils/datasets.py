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
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
from tqdm.autonotebook import tqdm
import cv2

from utils import general_utils
from pytorch3d.renderer import (
    look_at_view_transform
)

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

    def __init__(self, cfg):
        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.input_dir_img = cfg['semantic_dis_training']['gen_dir_img']
        self.input_dir_mesh = cfg['semantic_dis_training']['gen_dir_mesh']
        self.dataset_meshes_list = general_utils.get_instances(self.input_dir_mesh, self.input_dir_img)
        self.gen_poses = cfg['semantic_dis_training']['gen_poses']
        self.cached_pred_poses = pickle.load(open(self.gen_poses, "rb"))

        self.num_batches_gen_train = cfg["semantic_dis_training"]["num_batches_gen_train"]
        self.batch_size = cfg["semantic_dis_training"]["batch_size"]
        
        if self.num_batches_gen_train != -1:
            self.dataset_meshes_list = self.dataset_meshes_list[:self.num_batches_gen_train * self.batch_size]

        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

        # computing cache for entire dataset
        self.recompute_cache = cfg['semantic_dis_training']['recompute_input_mesh_cache']
        self.use_cache = cfg['semantic_dis_training']['use_input_mesh_cache']
        # TODO: make this cache filename more detailed
        self.input_mesh_cache_path = os.path.join(cfg["semantic_dis_training"]["cache_dir"],"generation_dataset_cache_{}.pt".format(len(self.dataset_meshes_list)))
        if self.recompute_cache or (self.use_cache and not os.path.exists(self.input_mesh_cache_path)):
            print("Caching generation dataset...")
            self.cached_data = [self.getitem_scratch(i) for i in tqdm(range(self.__len__()))]
            if not os.path.exists(cfg["semantic_dis_training"]["cache_dir"]):
                os.makedirs(cfg["semantic_dis_training"]["cache_dir"])
            torch.save(self.cached_data, self.input_mesh_cache_path)
        elif self.use_cache:
            print("Loading cached generation dataset at {}...".format(self.input_mesh_cache_path))
            self.cached_data = torch.load(self.input_mesh_cache_path)
            # gotcha for case where num_batches_gen_train has changed
            if len(self.cached_data) != len(self.dataset_meshes_list):
                raise ValueError("Cache length {} != requested length {}. Try recomputing cache with current settings.".format(len(self.cached_data), len(self.dataset_meshes_list)))

    
    def __len__(self):
        return len(self.dataset_meshes_list)


    def __getitem__(self, idx):
        if self.use_cache:
            return self.cached_data[idx]
        else:
            return self.getitem_scratch(idx)


    def getitem_scratch(self, idx):
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
            mesh = general_utils.load_untextured_mesh(curr_obj_path, self.cpu_device)
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



class RealDataset(Dataset):

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        real_shapes_dir = cfg["semantic_dis_training"]["dis_real_shapes_dir"]
        self.instances = general_utils.get_instances(real_shapes_dir)
        self.dis_type = cfg["semantic_dis_training"]["dis_type"]
        if self.dis_type not in ["multiview", "points"]:
            raise ValueError("Dis type not recognized")
        self.cache_dir = os.path.join(cfg["semantic_dis_training"]["cache_dir"], "real_dataset", self.dis_type, real_shapes_dir.replace('/', '_'))

        # computing all meshes from real shapes dir
        recompute = cfg["semantic_dis_training"]["dis_data_recompute"]
        if recompute or not os.path.exists(self.cache_dir):
            print("Recomputing {} assets...".format(self.dis_type))
            os.makedirs(self.cache_dir, exist_ok=True)
            for instance in tqdm(self.instances):
                if self.dis_type == "multiview":
                    self.render_and_save(os.path.join(real_shapes_dir, "{}.obj".format(instance)), self.cache_dir)
                else:
                    self.extract_points(os.path.join(real_shapes_dir, "{}.obj".format(instance)), self.cache_dir)
        else:
            print("Reusing previusly computed {} assets...".format(self.dis_type))

        self.img_transforms = transforms.Compose([transforms.ToTensor()])
        # saving tensor cache
        self.cache_path = os.path.join(self.cache_dir, "cache.pt")
        self.recreate_cache = cfg['semantic_dis_training']['dis_data_recreate_cache']
        self.use_cache = cfg['semantic_dis_training']['dis_data_use_cache']
        if self.recreate_cache or (self.use_cache and not os.path.exists(self.cache_path)):
            print("Caching generation dataset...")
            self.cached_data = [self.getitem_scratch(i) for i in tqdm(range(self.__len__()))]
            torch.save(self.cached_data, self.cache_path)
        elif self.use_cache:
            print("Loading cached generation dataset at {}...".format(self.cache_path))
            self.cached_data = torch.load(self.cache_path)
            # check for case where num_batches_gen_train has changed
            if len(self.cached_data) != len(self.instances):
                raise ValueError("Cache length {} != requested length {}. Try recomputing cache with current settings.".format(len(self.cached_data), len(self.dataset_meshes_list)))
    

    def __len__(self):
        return len(self.instances)


    def __getitem__(self, idx):
        if self.use_cache:
            return self.cached_data[idx]
        else:
            return self.getitem_scratch(idx)


    def getitem_scratch(self, idx):
        instance = self.instances[idx]
        if self.dis_type == "multiview":
            mv_imgs = []
            for instance_mv_img in sorted([str(path) for path in list(Path(self.cache_dir).rglob('{}*'.format(instance)))]):
                mv_imgs.append(self.img_transforms(Image.open(instance_mv_img).convert("RGB")).unsqueeze(0))
            item = torch.cat(mv_imgs)

        else:
            item = torch.load(os.path.join(self.cache_dir, "{}.pt".format(instance)))

        return item

    
    def extract_points(self, mesh_path, output_dir):
        mesh = general_utils.load_untextured_mesh(mesh_path, self.device)

        mesh_num_verts = self.cfg["semantic_dis_training"]["mesh_num_verts"]
        points = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=mesh_num_verts)
        points = torch.squeeze(points, dim=0) # dimension is [num_sampled_points, 3]
        instance = mesh_path.split('/')[-1][:-4]
        torch.save(points, os.path.join(output_dir, "{}.pt".format(instance)))


    def render_and_save(self, mesh_path, output_dir):
        instance = mesh_path.split('/')[-1][:-4]
        azims = self.cfg["semantic_dis_training"]["dis_mv_azims"]
        num_azims = len(azims)
        dists = torch.ones(num_azims) * self.cfg["semantic_dis_training"]["dis_mv_dist"]
        elevs = torch.ones(num_azims) * self.cfg["semantic_dis_training"]["dis_mv_elev"]
        R, T = look_at_view_transform(dists, elevs, azims)
        mesh = general_utils.load_untextured_mesh(mesh_path, self.device)
        meshes = mesh.extend(num_azims)
        lighting_mode = self.cfg["semantic_dis_training"]["dis_mv_lighting_mode"]
        renders = general_utils.render_mesh(meshes, R, T, self.device, img_size=self.cfg["semantic_dis_training"]["dis_mv_img_size"], 
                                            silhouette=self.cfg["semantic_dis_training"]["dis_mv_render_sil"], custom_lights=lighting_mode)
        for i, render in enumerate(renders):
            img_render_rgb = (render[..., :3].cpu().numpy()*255).astype(int) 
            rgb_render_filename = "{}_{:03d}.jpg".format(instance,i)
            cv2.imwrite(os.path.join(output_dir, rgb_render_filename), img_render_rgb)
        
