import argparse
import os
import glob
import pprint
import pickle

from tqdm.autonotebook import tqdm
import torch
from PIL import Image
import numpy as np
from pytorch3d.io import save_obj

from utils import utils
from deformation.deformation_net import DeformationNetwork
from mesh_refiner import MeshRefiner
import deformation.losses as def_losses
from utils import brute_force_pose_est


# predicts poses of a dataset
def predict_pose(cfg, device, meshes_to_process):
    input_dir_img = cfg['dataset']['input_dir_img']
    input_dir_mesh = cfg['dataset']['input_dir_mesh']
    num_azims = cfg['brute_force_pose_est']['num_azims']
    num_elevs = cfg['brute_force_pose_est']['num_elevs']
    num_dists = cfg['brute_force_pose_est']['num_dists']

    cached_pred_poses = {}
    tqdm_out = utils.TqdmPrintEvery()
    for instance_name in tqdm(meshes_to_process, file=tqdm_out):
        img_path = os.path.join(input_dir_img, instance_name + ".png")
        mesh_path = os.path.join(input_dir_mesh, instance_name + ".obj")
        mask = np.asarray(Image.open(img_path))[:,:,3] > 0
        with torch.no_grad():
            mesh = utils.load_untextured_mesh(mesh_path, device)
            pred_azim, pred_elev, pred_dist, renders = brute_force_pose_est.brute_force_estimate_pose(mesh, mask, num_azims, num_elevs, num_dists, device, 8)
            cached_pred_poses[instance_name] = {"azim": pred_azim.item(), "elev": pred_elev.item(), "dist": pred_dist.item()}

    return cached_pred_poses


# postprocesses imgs/meshes based on a dict of cached predicted poses (the output of predict_pose)
def postprocess_data(cached_pred_poses, output_dir_mesh, cfg, device, recompute_meshes):
    input_dir_img = cfg['dataset']['input_dir_img']
    input_dir_mesh = cfg['dataset']['input_dir_mesh']

    # postprocessing each mesh/img in dataset
    refiner = MeshRefiner(cfg, device)
    loss_info = {}
    tqdm_out = utils.TqdmPrintEvery()
    for instance_name in tqdm(cached_pred_poses, file=tqdm_out):
        curr_obj_path = os.path.join(output_dir_mesh, instance_name+".obj")
        if recompute_meshes or not os.path.exists(curr_obj_path):
            input_image = np.asarray(Image.open(os.path.join(input_dir_img, instance_name+".png")))
            with torch.no_grad():
                mesh = utils.load_untextured_mesh(os.path.join(input_dir_mesh, instance_name+".obj"), device)
            pred_dist = cached_pred_poses[instance_name]['dist']
            pred_elev = cached_pred_poses[instance_name]['elev']
            pred_azim = cached_pred_poses[instance_name]['azim']

            curr_refined_mesh, curr_loss_info = refiner.refine_mesh(mesh, input_image, pred_dist, pred_elev, pred_azim)
            save_obj(curr_obj_path, curr_refined_mesh.verts_packed(), curr_refined_mesh.faces_packed())
            loss_info[instance_name] = curr_loss_info

    return loss_info


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
    parser.add_argument('--batch_i', type=int, default=1, help='which batch this is (1-indexed)')
    parser.add_argument('--num_batches', type=int, default=1, help='number of batches to split dataset into')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--name', type=str, default="processed", help='name of experiment')
    parser.add_argument('--recompute_poses', action='store_true', help='Recompute the poses, even if there is a precomputed pose cache.')
    parser.add_argument('--recompute_meshes', action='store_true', help='Recompute the meshes, even for ones which already exist.')
    args = parser.parse_args()

    if args.batch_i > args.num_batches or args.batch_i <= 0:
            raise ValueError("batch_i cannot be greater than num_batches nor less than 1")

    device = torch.device("cuda:"+str(args.gpu))
    cfg = utils.load_config(args.cfg_path, "configs/default.yaml")
    input_dir_img = cfg['dataset']['input_dir_img']
    input_dir_mesh = cfg['dataset']['input_dir_mesh']

    # making processed meshes output dir
    if input_dir_mesh[-1] == '/': input_dir_mesh = input_dir_mesh[:-1]
    output_dir_mesh = os.path.join("data", input_dir_mesh.split('/')[-1]+"_"+args.name, "batch_{}_of_{}".format(args.batch_i, args.num_batches))
    if not os.path.exists(output_dir_mesh):
        os.makedirs(output_dir_mesh)

    # finding which instances are in this batch
    instance_names = []
    for mesh_path in glob.glob(os.path.join(input_dir_mesh, "*.obj")):
        img_path = os.path.join(input_dir_img, mesh_path.split('/')[-1].replace("obj", "png"))
        if not os.path.exists(img_path):
            raise ValueError("Couldn't find image for mesh {}.".format(mesh_path))
        instance_names.append(mesh_path.split('/')[-1][:-4])
    instance_names = sorted(instance_names)
    curr_batch_instances = split(instance_names, args.num_batches)[args.batch_i-1]

    # precomputing poses for this batch if necessary
    pred_poses_path = os.path.join(output_dir_mesh, "pred_poses.p")
    if args.recompute_poses or not os.path.exists(pred_poses_path):
        print("\nPredicting Poses...\n")
        cached_pred_poses = predict_pose(cfg, device, curr_batch_instances)
        pickle.dump(cached_pred_poses, open(pred_poses_path,"wb"))
    else:
        cached_pred_poses = pickle.load(open(pred_poses_path, "rb"))

    # postprocessing meshes
    print("\nPerforming optimization-based postprocessing on mesh reconstructions...\n")
    loss_info = postprocess_data(cached_pred_poses, output_dir_mesh, cfg, device, args.recompute_meshes)
    pickle.dump(loss_info, open(os.path.join(output_dir_mesh, "loss_info.p"), "wb"))

