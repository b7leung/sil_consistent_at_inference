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
            self.img_transforms = transforms.Compose([
                # TODO: remove the resize here and fix discriminator to allow for 224x224 instead of 64x64
                transforms.Resize((64,64)),
                transforms.ColorJitter(brightness=0.1, contrast=0, saturation=0, hue=0),
                transforms.RandomPerspective(distortion_scale=0.1, p=0.25, interpolation=3, fill=0),
                transforms.RandomAffine(10, translate=(0.1,0.1), scale=(0.70,1.3), shear=[-7,7,-7,7], fillcolor=(0,0,0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
            ])
        elif split == "val":
            names_file = "val.lst"
            self.img_transforms = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor(),
            ])


        with open (os.path.join(self.real_dataset_dir, names_file), 'r') as f:
            self.real_image_filenames = f.read().split('\n')
        with open (os.path.join(self.fake_dataset_dir, names_file), 'r') as f:
            self.fake_image_filenames = f.read().split('\n')

        if len(self.real_image_filenames) != len(self.fake_image_filenames):
            raise Error("Real dataset must have the same number of images as the fake dataset")
        

    def __len__(self):
        return len(self.real_image_filenames)


    def __getitem__(self, idx):
        data = {}
        data["real"] = self.img_transforms(Image.open(os.path.join(self.real_dataset_dir, self.real_image_filenames[idx])).convert("RGB"))
        data["real_path"] = os.path.join(self.real_dataset_dir, self.real_image_filenames[idx])
        data["fake"] = self.img_transforms(Image.open(os.path.join(self.fake_dataset_dir, self.fake_image_filenames[idx])).convert("RGB"))
        data["fake_path"] = os.path.join(self.fake_dataset_dir, self.fake_image_filenames[idx])

        return data