import pprint
import argparse
import os
import glob
import pprint
import pickle
import shutil
import itertools
import copy

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from pytorch3d.io import save_obj

from utils import utils
from postprocess_dataset import predict_pose
from mesh_refiner import MeshRefiner


def create_configs_list(param_search_dict, default_cfg_dict, input_folder_dir):

    param_axes_names = []
    param_axes = []
    for param_category in param_search_dict:
        for param_name in param_search_dict[param_category]:
            param_axes_names.append("{} {}".format(param_category, param_name))
            param_axes.append(param_search_dict[param_category][param_name])
            
    grid_parameters_list = list(itertools.product(*param_axes))
    gridsearch_configs_list = []
    for param_list in grid_parameters_list:
        curr_config = copy.deepcopy(default_cfg_dict)
        for param_i in range(len(param_axes_names)):
            param_entry = param_axes_names[param_i]
            param_category = param_entry.split(" ")[0]
            param_name = param_entry.split(" ")[1]
            curr_config[param_category][param_name] = param_list[param_i]

        curr_config["dataset"]["input_dir_mesh"] = input_folder_dir
        curr_config["dataset"]["input_dir_img"] = input_folder_dir
        gridsearch_configs_list.append(curr_config)

    return gridsearch_configs_list, grid_parameters_list, param_axes_names


def perform_refinement_gridsearch(instance_name, pose_dict, cfg_dict_list, input_dir, output_dir, device):

    instance_obj_path = os.path.join(input_dir, "{}.obj".format(instance_name))
    instance_img_path = os.path.join(input_dir, "{}.png".format(instance_name))
    pred_dist = pose_dict['dist']
    pred_elev = pose_dict['elev']
    pred_azim = pose_dict['azim']

    loss_info = {}
    for i, curr_cfg in enumerate(tqdm(cfg_dict_list, desc="Instance {} gridsearch".format(instance_name))):
        with torch.no_grad():
            input_mesh = utils.load_untextured_mesh(instance_obj_path, device)
        input_image = np.asarray(Image.open(instance_img_path))

        refiner = MeshRefiner(curr_cfg, device)
        curr_refined_mesh, curr_loss_info = refiner.refine_mesh(input_mesh, input_image, pred_dist, pred_elev, pred_azim)

        save_obj(os.path.join(output_dir, "{}.obj".format(i)), curr_refined_mesh.verts_packed(), curr_refined_mesh.faces_packed())
        loss_info[i] = curr_loss_info
        pickle.dump(loss_info, open(os.path.join(output_dir, "loss_info.p"), "wb"))


# run a gridsearch on folder. Folder should have a file called param_search_dict.yaml defining a parameter search dictionary
# it should also have the meshes and images which will be used for the gridsearch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gridsearch on a folder.')
    parser.add_argument('input_folder', type=str, help='Path to folder.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu))

    # loading list of gridsearch parameter config dicts
    param_search_dict = utils.load_config(os.path.join(args.input_folder, "param_search_dict.yaml"))
    default_cfg = utils.load_config("configs/default.yaml")
    gridsearch_configs_list, param_list, params_names = create_configs_list(param_search_dict, default_cfg, args.input_folder)
    num_configs = len(gridsearch_configs_list)
    with open(os.path.join(args.input_folder, "grid_parameters_info.lst"), "w") as f:
        f.write("{}\n".format(str(params_names)))
        for i, params in enumerate(param_list):
            f.write("{} -- {}\n".format(i, str(params)))
    print("Starting Gridsearch -- Number of configs: {}. Search will take ~{} hours per instance.".format(num_configs, (num_configs*1.25)/60))

    # predicting poses
    instance_names = [path.split('/')[-1].replace(".obj", "") for path in glob.glob(os.path.join(args.input_folder, "*.obj"))]
    cached_pred_poses = predict_pose(gridsearch_configs_list[0], device, instance_names)

    # performing gridsearch refinement
    for instance in tqdm(cached_pred_poses, desc="instances"):
        output_dir = os.path.join(args.input_folder, "{}_gridsearch".format(instance))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        perform_refinement_gridsearch(instance, cached_pred_poses[instance], gridsearch_configs_list, args.input_folder, output_dir, device)


