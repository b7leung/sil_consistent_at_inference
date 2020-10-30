import glob
from pathlib import Path
import argparse
import os
import pickle

from tqdm.autonotebook import tqdm
import cv2
import torch
from pytorch3d.io import load_objs_as_meshes

from ..utils import general_utils


def render(INPUT_MESH_DIR, OUTPUT_RENDER_DIR, render_textureless, device, suffix=".obj", instances_list=None):

    # render settings
    img_size = 224
    device = torch.device("cuda:0")
    batch_size = 8
    num_azims = 1

    renders_camera_params = {}

    rgb_output_render_dir = os.path.join(OUTPUT_RENDER_DIR, "rgb")
    rgba_output_render_dir = os.path.join(OUTPUT_RENDER_DIR, "rgba")
    if not os.path.exists(rgb_output_render_dir):
        os.makedirs(rgb_output_render_dir)
    if not os.path.exists(rgba_output_render_dir):
        os.makedirs(rgba_output_render_dir)
        
    obj_paths = list(Path(INPUT_MESH_DIR).rglob('*{}'.format(suffix)))

    for model_path in tqdm(obj_paths, file=general_utils.TqdmPrintEvery()):
        model_name = str(model_path).split('/')[-2]
        if instances_list is None or model_name in instances_list:
            # 0.,  45.,  90., 135., 180., 225., 270., 315.
            #azims = torch.linspace(0, 360, num_azims+1)[:-1]
            azims = torch.rand(num_azims) * 360
            elevs = torch.ones(num_azims) * 40
            dists = torch.ones(num_azims) * 1

            if render_textureless:
                mesh = general_utils.load_untextured_mesh(model_path, device)
                renders = general_utils.batched_render(mesh, azims, elevs, dists, batch_size, device, False, img_size, False)
            else:
                try:
                    mesh = load_objs_as_meshes([model_path], device=device)
                    renders = general_utils.batched_render(mesh, azims, elevs, dists, batch_size, device, False, img_size, False)
                except ValueError:
                    mesh = general_utils.load_untextured_mesh(model_path, device)
                    renders = general_utils.batched_render(mesh, azims, elevs, dists, batch_size, device, False, img_size, False)

            for i, render in enumerate(renders):
                
                # assumes shapenet file hierarchy, where model name is the folder name
                if num_azims == 1:
                    render_name = model_name
                else:
                    render_name = "{}_{}".format(model_name, i)

                img_render_rgba = (render.cpu().numpy()*255).astype(int) 
                rgba_render_filename = "{}.png".format(render_name)
                cv2.imwrite(os.path.join(rgba_output_render_dir, rgba_render_filename), img_render_rgba)

                img_render_rgb = (render[ ..., :3].cpu().numpy()* 255).astype(int) 
                rgb_render_filename = "{}.jpg".format(render_name)
                cv2.imwrite(os.path.join(rgb_output_render_dir, rgb_render_filename), img_render_rgb)

                renders_camera_params[render_name] = {"azim": azims[i].item(), "elev": elevs[i].item(), "dist": dists[i].item()}
                pickle.dump(renders_camera_params, open(os.path.join(rgba_output_render_dir, "renders_camera_params.pt"), "wb"))


#  python -m sil_consistent_at_inference.data_prep_tools.render_meshes /home/svcl-oowl/dataset/shapenet_test/03001627 /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data/pytorch3d_shapenet_renders_test/03001627 --suffix model_watertight.obj --textureless --instances_list /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data_prep_tools/occnet_test_set_lists/03001627_test.lst

#  python -m sil_consistent_at_inference.data_prep_tools.render_meshes /home/svcl-oowl/dataset/ShapeNetCore.v1/03001627 /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data/pytorch3d_shapenet_renders/03001627 --suffix model_watertight.obj --textureless --instances_list /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data_prep_tools/occnet_test_set_lists/03001627_test.lst

#  python -m sil_consistent_at_inference.data_prep_tools.render_meshes /home/svcl-oowl/dataset/ShapeNetCore.v1/02933112 /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data/input_images/pytorch3d_shapenet_renders/02933112_sym --suffix model_watertight.obj --textureless --instances_list /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data_prep_tools/occnet_test_set_lists/02933112_sym.lst

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Render a folder of meshes')
    parser.add_argument('input_mesh_dir', type=str, help='Path to input mesh dir to render')
    parser.add_argument('output_render_dir', type=str, help='Path to output render dir')
    parser.add_argument('--instances_list', type=str, default=None, help='Path to list of instances to render')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--textureless', action="store_true", help='Gpu number to use.')
    parser.add_argument('--suffix', type=str, default=".obj", help='only render meshes with this suffix')
    args = parser.parse_args()

    device = torch.device("cuda:"+str(args.gpu))

    if args.instances_list is not None:
        with open(args.instances_list, "r") as f:
            instances_list = f.read().split('\n')

    render(args.input_mesh_dir, args.output_render_dir, args.textureless,device, args.suffix, instances_list)

