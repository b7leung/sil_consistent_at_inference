import os
import pprint
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


class PosePredDataset(Dataset):
    """Dataset used for mesh deformation generator. Each datum is a tuple {mesh vertices, input img, predicted pose}"""

    def __init__(self, cfg, instances_to_use=None):
        self.cfg = cfg

        self.input_dir_img = cfg['input_img_dir']
        self.pose_path = cfg['pose_path']
        self.pose_dict = pickle.load(open(self.pose_path, "rb"))
        self.img_transforms = transforms.Compose([transforms.ToTensor()])

        # filtering based on specified instances
        self.instance_names = list(self.pose_dict.keys())
        if instances_to_use is not None:
            old_dataset_length = len(self.instance_names)
            self.instance_names = [x for x in self.instance_names if x in instances_to_use]
            print("Filtered based on specified instances, from {} -> {}.".format(old_dataset_length, len(self.instance_names)))
     

    def __len__(self):
        return len(self.instance_names)


    # poses are azim, elev, dist, [b, 3]
    # [0, 360], [40], ~1
    def __getitem__(self, idx):
        data = {}
        instance_name = self.instance_names[idx]
        data["instance_name"] = instance_name
        data["pose"] = torch.tensor([self.pose_dict[instance_name][pose_axis] for pose_axis in ["azim", "elev", "dist"]])

        curr_img_path = os.path.join(self.input_dir_img, instance_name+".png")
        rgba_image = Image.open(curr_img_path)
        data["image"] = self.img_transforms(rgba_image.convert("RGB"))

        return data