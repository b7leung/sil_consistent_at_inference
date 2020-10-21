import pickle
import pprint
import glob
import argparse
import os
import io

import trimesh
import torch
import pandas as pd
from tqdm.autonotebook import tqdm

from utils import utils
from evaluation import compute_iou_2d, compute_iou_2d_given_pose, compute_iou_3d, compute_chamfer_L1


def process_get_shapes_lst(gt_shapes_lst_path):
    gt_shapes_dict = {}
    with open(gt_shapes_lst_path, 'r') as f:
        f = f.read().split('\n')
        for line in f:
            if line != "":
                gt_shapes_dict[line.split(" ")[0]] = line.split(" ")[1]
    return gt_shapes_dict


def load_gridsearch_params(params_list_path):
    gridsearch_params_dict = {}
    with open(params_list_path, 'r') as f:
        f = f.read().split('\n')
        params_list = f[0].replace("'", "").replace("[","").replace("]", "").split(',')
        params_list = [param_name[1:] if param_name[0]==' ' else param_name for param_name in params_list]
        for i in range(1, len(f)):
            line = f[i]
            if line != "":
                entry_num = line.split(' -- ')[0]
                entry_config_list = line.split(' -- ')[1].replace("'", "").replace("(","").replace(")", "").replace(' ','').split(',')
                entry_config_dict = {params_list[param_i]: entry_config_list[param_i] for param_i in range(len(params_list))}
            
            gridsearch_params_dict[entry_num] = entry_config_dict
    return gridsearch_params_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate results on gridsearch on a folder.')
    parser.add_argument('input_folder', type=str, help='Path to folder.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()
    device = torch.device("cuda:"+str(args.gpu))

    # TODO: allow this to change
    gt_shapes_lst_path = "/home/svcl-oowl/brandon/research/sil_consistent_at_inference/data_evaluation/pix3d_chair/pix3d_chair_gt_shapes.lst"
    gridsearch_folder = args.input_folder

    gt_shapes_dict = process_get_shapes_lst(gt_shapes_lst_path)
    instances = [i.split('/')[-1].replace('.obj','') for i in glob.glob(os.path.join(gridsearch_folder, "*.obj"))]

    # loading gridsearch param info into a dict
    gridsearch_param_dict = load_gridsearch_params(os.path.join(gridsearch_folder, "grid_parameters_info.lst"))

    for instance in instances:
        # loading original input mesh
        mesh_original_path = os.path.join(gridsearch_folder, "{}.obj".format(instance))
        with torch.no_grad():
            mesh_original = utils.load_untextured_mesh(mesh_original_path, device)
        mesh_original_trimesh = trimesh.load(mesh_original_path)
        
        # loading gt mesh 
        mesh_gt_path = gt_shapes_dict[instance]
        with torch.no_grad():
            mesh_gt = utils.load_untextured_mesh(mesh_gt_path, device)
        mesh_gt_trimesh = trimesh.load(mesh_gt_path)
        
        # computing metrics before refinement
        iou_3d_before = compute_iou_3d(mesh_original_trimesh, mesh_original, mesh_gt_trimesh, mesh_gt)
        iou_3d_norm_before = compute_iou_3d(mesh_original_trimesh, mesh_original, mesh_gt_trimesh, mesh_gt, full_unit_normalize=True)
        chamfer_before = compute_chamfer_L1(mesh_original_trimesh, mesh_original, mesh_gt_trimesh, mesh_gt)
        chamfer_norm_before = compute_chamfer_L1(mesh_original_trimesh, mesh_original, mesh_gt_trimesh, mesh_gt, full_unit_normalize=True)
        
        instance_result_dir = os.path.join(gridsearch_folder, "{}_gridsearch".format(instance))
        gridsearch_results_df = pd.DataFrame()
        for mesh_refined_path in tqdm(glob.glob(os.path.join(instance_result_dir, "*.obj")), file=utils.TqdmPrintEvery()):

            refined_number = mesh_refined_path.split('/')[-1].replace(".obj","")
            with torch.no_grad():
                mesh_refined = utils.load_untextured_mesh(mesh_refined_path, device)
            refinement_metrics_dict = {"refinement_number": refined_number,
                                       "iou_3d": -1, "delta_iou_3d": -1, "iou_3d_norm": -1, "delta_iou_3d_norm": -1,
                                       "chamfer": -1, "delta_chamfer": -1, "chamfer_norm": -1, "delta_chamfer_norm": -1}

            # IndexError is thrown by trimesh if meshes have nan vertices
            try:
                mesh_refined_trimesh = trimesh.load(mesh_refined_path)
            except IndexError:
                gridsearch_results_df = gridsearch_results_df.append({**gridsearch_param_dict[refined_number], **refinement_metrics_dict}, ignore_index=True)
                continue
            
            #computing metrics after refinement
            iou_3d_after = compute_iou_3d(mesh_refined_trimesh, mesh_refined, mesh_gt_trimesh, mesh_gt)
            iou_3d_norm_after = compute_iou_3d(mesh_refined_trimesh, mesh_refined, mesh_gt_trimesh, mesh_gt, full_unit_normalize=True)
            chamfer_after = compute_chamfer_L1(mesh_refined_trimesh, mesh_refined, mesh_gt_trimesh, mesh_gt)
            chamfer_norm_after = compute_chamfer_L1(mesh_refined_trimesh, mesh_refined, mesh_gt_trimesh, mesh_gt, full_unit_normalize=True)

            refinement_metrics_dict["iou_3d"] = iou_3d_after
            refinement_metrics_dict["delta_iou_3d"] = iou_3d_after - iou_3d_before
            refinement_metrics_dict["iou_3d_norm"] = iou_3d_norm_after
            refinement_metrics_dict["delta_iou_3d_norm"] = iou_3d_norm_after - iou_3d_norm_before
            refinement_metrics_dict["chamfer"] = chamfer_after
            refinement_metrics_dict["delta_chamfer"] = chamfer_after - chamfer_before
            refinement_metrics_dict["chamfer_norm"] = chamfer_norm_after
            refinement_metrics_dict["delta_chamfer_norm"] = chamfer_norm_after - chamfer_norm_before
            gridsearch_results_df = gridsearch_results_df.append({**gridsearch_param_dict[refined_number], **refinement_metrics_dict}, ignore_index=True)
            
    gridsearch_results_df.to_pickle(os.path.join(gridsearch_folder, "eval_df.pkl"))