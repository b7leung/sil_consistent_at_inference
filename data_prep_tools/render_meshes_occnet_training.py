import glob
from pathlib import Path
import argparse
import os
import pickle
import numpy as np

from tqdm.autonotebook import tqdm
import cv2
import torch
from pytorch3d.io import load_objs_as_meshes

from utils import general_utils


def render(model_path, output_renders_dir, render_textureless, device, num_azims=24, batch_size=14):

    if not os.path.exists(output_renders_dir):
        os.makedirs(output_renders_dir)

    # renders settings
    img_size = 137
    azims = torch.rand(num_azims) * 360
    elevs = torch.from_numpy(np.random.uniform(low=30, high=50, size=(num_azims))).type(torch.float32)
    dists = torch.from_numpy(np.random.uniform(low=0.9, high=1.3, size=(num_azims))).type(torch.float32)

    # loading mesh and rendering
    if render_textureless:
        mesh = general_utils.load_untextured_mesh(model_path, device)
        renders = general_utils.batched_render(mesh, azims, elevs, dists, batch_size, device, False, img_size, False)
    else:
        raise ValueError("Not implemented yet")

    # saving renders and pose dict
    renders_camera_params = {}
    for i, render in enumerate(renders):
        
        img_render_rgb = (render[..., :3].cpu().numpy()*255).astype(int) 
        rgb_render_filename = "{:03d}.jpg".format(i)
        cv2.imwrite(os.path.join(output_renders_dir, rgb_render_filename), img_render_rgb)

        renders_camera_params[rgb_render_filename] = {"azim": azims[i].item(), "elev": elevs[i].item(), "dist": dists[i].item()}
    pickle.dump(renders_camera_params, open(os.path.join(output_renders_dir, "renders_camera_params.pt"), "wb"))


def instance_rendered(output_renders_dir):
    #print(os.path.join(output_renders_dir, "renders_camera_params.pt"))
    if os.path.exists(os.path.join(output_renders_dir, "renders_camera_params.pt")):
        return True
    else:
        return False

# python render_meshes_occnet_training.py /home/svcl-oowl/dataset/ShapeNetCore.v1/02933112 /home/svcl-oowl/brandon/research/occ_uda/data/ShapeNet/test --textureless
# python /home/svcl-oowl/dataset/ShapeNetCore.v1/02933112 /home/svcl-oowl/brandon/research/occ_uda/data/ShapeNet/02933112 --textureless

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render a folder of meshes')
    parser.add_argument('input_mesh_dir', type=str, help='Path to input shapenet dir with meshes')
    parser.add_argument('output_render_dir', type=str, help='Path to occnet folder')
    parser.add_argument('--textureless', action="store_true", help='Render textureless.')
    parser.add_argument('--rerender', action="store_true", help='rerender')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()
    
    
    device = torch.device("cuda:"+str(args.gpu))

    pix3d_class_dir = args.output_render_dir
    if pix3d_class_dir[-1] == '/':
        pix3d_class_dir = pix3d_class_dir[:-1]
    instances_list = [path.split('/')[-2] for path in glob.glob(args.output_render_dir+"/*/")]

    num_azims = 24
    for instance in tqdm(instances_list, file=general_utils.TqdmPrintEvery()):
        print(instance)
        mesh_path = os.path.join(args.input_mesh_dir, instance, "model_watertight.obj")
        output_renders_dir = os.path.join(pix3d_class_dir, instance, "img_pytorch3d")
        if args.rerender or not instance_rendered(output_renders_dir):
            render(mesh_path, output_renders_dir, args.textureless, device, num_azims=num_azims)

