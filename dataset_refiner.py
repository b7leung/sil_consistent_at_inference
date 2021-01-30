import argparse
import os
import glob
import pprint
import pickle
import shutil

from tqdm.autonotebook import tqdm
import torch
from PIL import Image
import numpy as np
from pytorch3d.io import save_obj
import pytorch3d

from utils import general_utils
from mesh_refiner import MeshRefiner
from utils.brute_force_pose_est import brute_force_estimate_pose, brute_force_estimate_dist


# predicts poses of a dataset
def predict_pose(cfg, device, meshes_to_process):
    input_dir_img = cfg['dataset']['input_dir_img']
    input_dir_mesh = cfg['dataset']['input_dir_mesh']
    num_azims = cfg['brute_force_pose_est']['num_azims']
    num_elevs = cfg['brute_force_pose_est']['num_elevs']
    num_dists = cfg['brute_force_pose_est']['num_dists']

    cached_pred_poses = {}
    for instance_name in tqdm(meshes_to_process, file=general_utils.TqdmPrintEvery(), desc="Predicting Poses"):
        print(instance_name)
        img_path = os.path.join(input_dir_img, instance_name + ".png")
        mesh_path = os.path.join(input_dir_mesh, instance_name + ".obj")
        mask = np.asarray(Image.open(img_path))[:,:,3] > 0
        with torch.no_grad():
            mesh = general_utils.load_untextured_mesh(mesh_path, device)

            # this catches an error in occnet reconstructions where the output has no vertices
            if mesh.verts_packed().shape[0] == 0:
                print("skipped")
                continue

            pred_azim, pred_elev, pred_dist, render, iou = brute_force_estimate_pose(mesh, mask, num_azims, num_elevs, num_dists, device, 8)
            cached_pred_poses[instance_name] = {"azim": pred_azim.item(), "elev": pred_elev.item(), "dist": pred_dist.item()}

    return cached_pred_poses


# postprocesses imgs/meshes based on a dict of cached predicted poses (the output of predict_pose)
def postprocess_data(pred_poses_dict, output_dir_mesh, cfg, device, recompute_meshes):
    input_dir_img = cfg['dataset']['input_dir_img']
    input_dir_mesh = cfg['dataset']['input_dir_mesh']
    error_log_path = os.path.join(output_dir_mesh, "..", "error_log.txt")

    # postprocessing each mesh/img in dataset
    refiner = MeshRefiner(cfg, device)
    refinements_info = {}
    for instance_name in tqdm(pred_poses_dict, file=general_utils.TqdmPrintEvery()):
        curr_obj_path = os.path.join(output_dir_mesh, instance_name+".obj")
        if recompute_meshes or not os.path.exists(curr_obj_path):
            input_image = np.asarray(Image.open(os.path.join(input_dir_img, instance_name+".png")))
            with torch.no_grad():
                mesh = general_utils.load_untextured_mesh(os.path.join(input_dir_mesh, instance_name+".obj"), device)
                
            # this catches an error in occnet reconstructions where the output has no or very few vertices
            if mesh.verts_packed().shape[0] < 50:
                with open(error_log_path, "a") as f: 
                    f.write("skipped refinement, no vertices -- {}\n".format(os.path.join(input_dir_mesh, instance_name+".obj")))
                    continue

            pred_dist = pred_poses_dict[instance_name]['dist']
            pred_elev = pred_poses_dict[instance_name]['elev']
            pred_azim = pred_poses_dict[instance_name]['azim']

            curr_refined_mesh, refinement_info = refiner.refine_mesh(mesh, input_image, pred_dist, pred_elev, pred_azim)
            save_obj(curr_obj_path, curr_refined_mesh.verts_packed(), curr_refined_mesh.faces_packed())
            refinements_info[instance_name] = refinement_info

    return refinements_info


# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))


# given the path to a configuration file, and the path to a folder with meshes and segmented images, will postprocess all meshes in that folder.
# the folder needs to have .obj meshes, and for each mesh, a corresponding .png image (with segmented, transparent bg) of size 244 x 224, with the same filename.
# Upon completion, will save a dict with dataframes containing training information, and save all the processed meshes
# supports a batched mode to allow for processing across multiple GPUs
# example usages:
# python postprocess_dataset.py configs/default.yaml --batch_i 1 --num_batches 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Postprocess a folder of meshes and corresponding segmented images.')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('data_out_dir', type=str, help='Folder to output results into')
    parser.add_argument('--batch_i', type=int, default=1, help='which batch this is (1-indexed)')
    parser.add_argument('--num_batches', type=int, default=1, help='number of batches to split dataset into')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--use_given_poses', action='store_true', help='Perform refinements with the specified pose dict in config.')
    parser.add_argument('--recompute_poses', action='store_true', help='Recompute the poses, even if there is a precomputed pose cache.')
    parser.add_argument('--recompute_meshes', action='store_true', help='Recompute the meshes, even for ones which already exist.')
    parser.add_argument('--instances_list', nargs='+', help='instances to refine; if not set, refine all of them')

    args = parser.parse_args()

    if args.batch_i > args.num_batches or args.batch_i <= 0:
        raise ValueError("batch_i cannot be greater than num_batches nor less than 1")

    device = torch.device("cuda:"+str(args.gpu))
    cfg = general_utils.load_config(args.cfg_path, "configs/default.yaml")
    input_dir_img = cfg['dataset']['input_dir_img']
    input_dir_mesh = cfg['dataset']['input_dir_mesh']

    # making processed meshes output dir, and recursively copying all config yamls
    if input_dir_mesh[-1] == '/': input_dir_mesh = input_dir_mesh[:-1]
    output_dir_mesh = os.path.join(args.data_out_dir, "batch_{}_of_{}".format(args.batch_i, args.num_batches))
    if not os.path.exists(output_dir_mesh):
        os.makedirs(output_dir_mesh)
    #shutil.copyfile(args.cfg_path, os.path.join(args.data_out_dir, args.cfg_path.split("/")[-1]))
    all_inherited_cfg_paths = general_utils.get_all_inherited_cfgs(args.cfg_path) + ["configs/default.yaml"]
    for inherited_cfg_path in all_inherited_cfg_paths:
        shutil.copyfile(inherited_cfg_path, os.path.join(args.data_out_dir, inherited_cfg_path.split("/")[-1]))


    # finding which instances are in this batch
    instance_names = general_utils.get_instances(input_dir_mesh, input_dir_img)
    curr_batch_instances = split(instance_names, args.num_batches)[args.batch_i-1]
    if args.instances_list is not None:
        curr_batch_instances = [i for i in curr_batch_instances if i in args.instances_list]

    # precomputing poses for this batch if necessary
    #pred_poses_path = os.path.join(output_dir_mesh, "pred_poses.p")
    #if args.recompute_poses or not os.path.exists(pred_poses_path):
    #if args.use_given_poses:
    if cfg['dataset']['input_poses'] == "":
        print("\n Correcting Distances from GT Poses...\n")
        pred_poses_dict = pickle.load(open(os.path.join(input_dir_img, "renders_camera_params.pt"), "rb"))
        pred_poses_dict = {instance:pred_poses_dict[instance] for instance in pred_poses_dict if instance in curr_batch_instances}
        pred_poses_dict = general_utils.correct_dists(cfg['dataset']['input_dir_img'], cfg['dataset']['input_dir_mesh'], pred_poses_dict, device, num_dists=40)
    else:
        print("\n Loading Poses from Saved Dict File...\n")
        pred_poses_dict = pickle.load(open(cfg['dataset']['input_poses'], "rb"))
        pred_poses_dict = {instance:pred_poses_dict[instance] for instance in pred_poses_dict if instance in curr_batch_instances}
    #else:
    #    print("\nPredicting Poses by Brute Force...\n")
    #    pred_poses_dict = predict_pose(cfg, device, curr_batch_instances)
    #pickle.dump(pred_poses_dict, open(pred_poses_path,"wb"))
    #else:
    #    pred_poses_dict = pickle.load(open(pred_poses_path, "rb"))

    # postprocessing meshes
    print("\nPerforming optimization-based postprocessing on mesh reconstructions...\n")
    # TODO: loss info is not preserved if loaded twice
    refinements_info = postprocess_data(pred_poses_dict, output_dir_mesh, cfg, device, args.recompute_meshes)
    pickle.dump(refinements_info, open(os.path.join(output_dir_mesh, "refinements_info.p"), "wb"))
