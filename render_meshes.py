import pprint
import glob
from pathlib import Path
import argparse
import os

from tqdm.autonotebook import tqdm
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import utils


def render(INPUT_MESH_DIR, OUTPUT_RENDER_DIR, silhouette, device):

    # render settings
    img_size = 224
    device = torch.device("cuda:0")
    batch_size = 8
    num_azims = 8
    # 0.,  45.,  90., 135., 180., 225., 270., 315.
    azims = torch.linspace(0, 360, num_azims+1)[:-1]
    elevs = torch.ones(num_azims) * 25
    dists = torch.ones(num_azims) * 1.7


    if not os.path.exists(OUTPUT_RENDER_DIR):
        os.makedirs(OUTPUT_RENDER_DIR)
        
    obj_paths = list(Path(INPUT_MESH_DIR).rglob('*.obj'))

    num_errors = 0
    tqdm_out = utils.TqdmPrintEvery()
    for model_path in tqdm(obj_paths, file=tqdm_out):
        print("number of errors so far: {}".format(num_errors))
        try:
            with torch.no_grad():
                mesh = utils.load_untextured_mesh(model_path, device)
                renders = utils.batched_render(mesh, azims, elevs, dists, batch_size, device, silhouette, img_size)
                for i, render in enumerate(renders):
                    
                    if silhouette:
                        # turn into hard 0/1 siluette
                        img_render = (render[ ..., 3].cpu().numpy() > 0).astype(int) * 255
                    else:
                        img_render = (render[ ..., :3].cpu().numpy()* 255).astype(int) 
                        
                    model_name = str(model_path).replace(INPUT_MESH_DIR,'').replace('/','_').replace(".obj","")
                    if model_name[0] == '_': model_name = model_name[1:]
                    
                    render_filename = "{}_{}.jpg".format(model_name, i)
                    cv2.imwrite(os.path.join(OUTPUT_RENDER_DIR,render_filename), img_render)
        except:
            num_errors += 1
            continue


    
# python render_meshes.py data/misc/example_shapenet data/semantic_dis/real_renders --silhouette
# python render_meshes.py data/test_dataset_processed data/semantic_dis/fake_renders --silhouette

# python render_meshes.py /home/svcl-oowl/dataset/ShapeNetCore.v1/03001627 data/semantic_dis/real_renders --silhouette
# python render_meshes.py data/onet_chair_pix3d_dann_simplified_processed data/semantic_dis/fake_renders --silhouette

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render a folder of meshes')
    parser.add_argument('input_mesh_dir', type=str, help='Path to input mesh dir to render')
    parser.add_argument('output_render_dir', type=str, help='Path to output render dir')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--silhouette', action='store_true', help='Render the silhouette instead of the shape.')
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu))
    render(args.input_mesh_dir, args.output_render_dir, args.silhouette, device)

