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
from pose_est import brute_force_pose_est


# given the path to a configuration file, and the path to a folder with meshes and segmented images, will postprocess all meshes in that folder.
# the folder needs to have .obj meshes, and for each mesh, a corresponding .png image (with segmented, transparent bg) of size 244 x 224, with the same filename.
# Upon completion, will return a pickle file called
# a dict with dataframes containing training information, and save all the processed meshes
# NOTE: Condition is that original iamges cannot have an underscore in the filename
# if meshes_to_render is not true, will only render meshes with instance names in that list (but all poses are still computed) 
def postprocess_data(input_dir_img, input_dir_mesh, cfg_path, gpu_num, recompute_poses=False, 
                     meshes_group_name="postprocessed", meshes_to_render=None):
    device = torch.device("cuda:"+str(gpu_num))

    data_paths = []
    for mesh_path in glob.glob(os.path.join(input_dir_mesh, "*.obj")):
        if "_" not in mesh_path.split('/')[-1]:
            img_path = os.path.join(input_dir_img, mesh_path.split('/')[-1].replace("obj", "png"))
            if not os.path.exists(img_path):
                raise ValueError("Couldn't find image for mesh {}.".format(mesh_path))
            data_paths.append({"mesh_path":mesh_path, "img_path":img_path})

    # predict poses if not previously done
    pred_poses_path = os.path.join(input_dir_mesh, "pred_poses.p")
    if recompute_poses or not os.path.exists(pred_poses_path):
        cached_pred_poses = {}
        print("Predicting Poses...")
        for data in tqdm(data_paths):
            img_path = data['img_path']
            mesh_path = data['mesh_path']
            mask = np.asarray(Image.open(img_path))[:,:,3] > 0
            with torch.no_grad():
                mesh = utils.load_untextured_mesh(mesh_path, device)
                pred_azim, pred_elev, pred_dist, renders = brute_force_pose_est.brute_force_estimate_pose(mesh, mask, 20, 20, 40, device, 8)
                cached_pred_poses[mesh_path.split('/')[-1][:-4]] = {"azim": pred_azim.item(), "elev": pred_elev.item(), "dist": pred_dist.item()}
        pickle.dump(cached_pred_poses, open(pred_poses_path,"wb"))
    else:
        cached_pred_poses = pickle.load(open(pred_poses_path, "rb"))
    
    # postprocessing each mesh/img in dataset
    refiner = MeshRefiner(cfg_path, device)
    loss_info = {}
    for instance_name in tqdm(cached_pred_poses):

        if meshes_to_render is None or instance_name in meshes_to_render:

            input_image = np.asarray(Image.open(os.path.join(input_dir_img, instance_name+".png")))
            with torch.no_grad():
                mesh = utils.load_untextured_mesh(os.path.join(input_dir_mesh, instance_name+".obj"), device)
            pred_dist = cached_pred_poses[instance_name]['dist']
            pred_elev = cached_pred_poses[instance_name]['elev']
            pred_azim = cached_pred_poses[instance_name]['azim']

            curr_refined_mesh, curr_loss_info = refiner.refine_mesh(mesh, input_image, pred_dist, pred_elev, pred_azim)
            loss_info[instance_name] = curr_loss_info

            save_obj(os.path.join(input_dir_mesh, instance_name + "_{}.obj".format(meshes_group_name)), curr_refined_mesh.verts_packed(), curr_refined_mesh.faces_packed())

    return loss_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Postprocess a folder of meshes and corresponding segmented images.'
    )
    parser.add_argument('input_dir_img', type=str, help='Path to folder with images for meshes.')
    parser.add_argument('input_dir_mesh', type=str, help='Path to folder with meshes to postprocess.')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()
    loss_info = postprocess_data(args.input_dir_img, args.input_dir_mesh, args.cfg_path, args.gpu)

    # TODO: Save loss info