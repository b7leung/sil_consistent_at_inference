# imports
import pprint
import pickle
import glob
import random
from pathlib import Path
import math
import argparse
import os

import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt
import pandas as pd
import cv2

from utils import general_utils
from utils import eval_utils
from deformation import losses



# python check_symmetry.py /home/svcl-oowl/dataset/ShapeNetCore.v1/02933112 data_prep_tools/check_symmetry_results 02933112 --suffix model_watertight.obj
# python check_symmetry.py /home/svcl-oowl/dataset/ShapeNetCore.v1/test data_prep_tools/check_symmetry_results test --suffix model_watertight.obj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Goes through a folder of meshes, and checks how many of them are symmetrical')
    parser.add_argument('input_mesh_dir', type=str, help='Path to input shapenet dir with meshes')
    parser.add_argument('output_dir', type=str, help='Path to occnet folder')
    parser.add_argument('name', type=str, help='Path to occnet folder')
    parser.add_argument('--suffix', type=str, default="*.obj", help='Gpu number to use.')
    parser.add_argument('--recheck', action="store_true", help='rerender')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu))

    mesh_paths = [str(path) for path in list(Path(args.input_mesh_dir).rglob(args.suffix))]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    sym_plane = [0,0,1]
    num_azim = 3
    dist = 1.3

    results_path = os.path.join(args.output_dir, "{}.p".format(args.name))
    if os.path.exists(results_path):
        results = pickle.load(open(results_path, "rb"))
    else:
        results = {}
    
    instance_blacklist = ["178382bfcc33146dd141480e2c154d3", "198f9bc1c351c97948c9215ea29b906f", "58891b4cc3fcdd7622bad8a709de6e5"]

    for mesh_path in tqdm(mesh_paths, file=general_utils.TqdmPrintEvery()):
        instance = mesh_path.split('/')[-2]
        if (args.recheck or mesh_path not in results) and (instance not in instance_blacklist):
            with torch.no_grad():
                print(mesh_path)
                mesh = general_utils.load_untextured_mesh(mesh_path, device)
                sym_loss, sym_triples = losses.image_symmetry_loss(mesh, sym_plane, num_azim, device, render_silhouettes=False, dist=dist)
                print(sym_loss)
                results[mesh_path] = sym_loss.item()
            
            pickle.dump(results, open(results_path, "wb"))
