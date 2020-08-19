import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from glob import glob


class SemanticDiscriminatorDataset(Dataset):
    """Semantic Discriminator dataset. Assumes length of the real/fake image folder, and are 224x224 jpgs are the same"""

    def __init__(self, cfg, split):

        self.real_dataset_dir = cfg['semantic_dis_training']['real_dataset_dir']
        self.fake_dataset_dir = cfg['semantic_dis_training']['fake_dataset_dir']
        
        if split == "train":
            names_file = "train.lst"
        elif split == "val":
            names_file = "val.lst"

        with open (os.path.join(self.real_dataset_dir, names_file), 'r') as f:
            self.real_image_filenames = f.read().split('\n')
        with open (os.path.join(self.fake_dataset_dir, names_file), 'r') as f:
            self.fake_image_filenames = f.read().split('\n')

        if len(self.real_image_filenames) != len(self.fake_image_filenames):
            raise Error("Real dataset must have the same number of images as the fake dataset")
        
        # TODO: experiment with more transforms
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.real_image_filenames)


    def __getitem__(self, idx):
        data = {}
        data["real"] = self.img_transforms(Image.open(os.path.join(self.real_dataset_dir, self.real_image_filenames[idx])).convert("RGB"))
        data["fake"] = self.img_transforms(Image.open(os.path.join(self.fake_dataset_dir, self.fake_image_filenames[idx])).convert("RGB"))

        return data