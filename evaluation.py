import argparse
import os
import glob
import pprint
import pickle
from pathlib import Path
import io

import numpy as np
from tqdm.autonotebook import tqdm
import pandas as pd
import trimesh
import torch
import pytorch3d
import pytorch3d.loss
from PIL import Image
from pytorch3d.renderer import look_at_view_transform

from utils import utils
from utils import eval_utils
from utils.brute_force_pose_est import brute_force_estimate_pose, get_iou
import inside_mesh
# https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py


def sample_points(mesh_verts, num_points):
    max_vert_values = torch.max(mesh_verts, 0).values.cpu().numpy()
    min_vert_values = torch.min(mesh_verts, 0).values.cpu().numpy()
    points_uniform = np.random.uniform(low=min_vert_values,high=max_vert_values,size=(num_points,3))
    return points_uniform


def compute_iou_2d(rec_mesh_torch, input_img, device, num_azims=20, num_elevs=20, num_dists=20):
    mask = np.asarray(input_img)[:,:,3] > 0
    _, _, _, render, iou = brute_force_estimate_pose(rec_mesh_torch, mask, num_azims, num_elevs, num_dists, device)
    return iou


def compute_iou_2d_given_pose(rec_mesh_torch, input_img, device, azim, elev, dist):
    mask = np.asarray(input_img)[:,:,3] > 0
    R, T = look_at_view_transform(dist, elev, azim) 
    render = utils.render_mesh(rec_mesh_torch, R, T, device)[0]
    iou = get_iou(render.cpu().numpy(), mask)   
    return iou


def compute_iou_3d(rec_mesh_input, rec_mesh_torch, gt_mesh_input, gt_mesh_torch, num_sample_points=100000, full_unit_normalize=False):
    rec_mesh = rec_mesh_input.copy()
    gt_mesh = gt_mesh_input.copy()
    if full_unit_normalize:
        points_uniform = np.random.uniform(low=-1,high=1,size=(num_sample_points,3))
        rec_mesh.vertices = eval_utils.normalize_pointclouds_numpy_all_axes(rec_mesh.vertices)
        gt_mesh.vertices = eval_utils.normalize_pointclouds_numpy_all_axes(gt_mesh.vertices)
    else:
        rec_points_uniform = sample_points(rec_mesh_torch.verts_packed(),int(num_sample_points/2))
        gt_points_uniform = sample_points(gt_mesh_torch.verts_packed(), int(num_sample_points/2))
        points_uniform = np.concatenate([rec_points_uniform, gt_points_uniform], axis=0)
        rec_mesh.vertices = eval_utils.normalize_pointclouds_numpy(rec_mesh.vertices)
        gt_mesh.vertices = eval_utils.normalize_pointclouds_numpy(gt_mesh.vertices)

    rec_mesh_occupancies = inside_mesh.check_mesh_contains(rec_mesh, points_uniform)
    gt_mesh_occupancies = inside_mesh.check_mesh_contains(gt_mesh, points_uniform)

    iou_3d = eval_utils.compute_iou(rec_mesh_occupancies, gt_mesh_occupancies)

    return iou_3d


def compute_chamfer_L1(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, sample_method="sample_surface", num_sample_points=100000, full_unit_normalize=False):
    if sample_method == "sample_surface":
        rec_pointcloud = pytorch3d.ops.sample_points_from_meshes(rec_mesh_torch, num_samples=num_sample_points)[0]
        gt_pointcloud = pytorch3d.ops.sample_points_from_meshes(gt_mesh_torch, num_samples=num_sample_points)[0]
        if full_unit_normalize:
            rec_pointcloud_normalized = torch.tensor(eval_utils.normalize_pointclouds_numpy_all_axes(rec_pointcloud.cpu().numpy(), proxy_points=None))
            #rec_pointcloud_normalized = torch.tensor(eval_utils.normalize_pointclouds_numpy_all_axes(rec_pointcloud.cpu().numpy(), proxy_points=rec_mesh.vertices))
            gt_pointcloud_normalized = torch.tensor(eval_utils.normalize_pointclouds_numpy_all_axes(gt_pointcloud.cpu().numpy(), proxy_points=None))
            #gt_pointcloud_normalized = torch.tensor(eval_utils.normalize_pointclouds_numpy_all_axes(gt_pointcloud.cpu().numpy(), proxy_points=gt_mesh.vertices))
        else:
            rec_pointcloud_normalized = eval_utils.normalize_pointclouds(rec_pointcloud)
            gt_pointcloud_normalized = eval_utils.normalize_pointclouds(gt_pointcloud)

    elif sample_method == "sample_uniformly":
        rec_points_uniform = sample_points(rec_mesh_torch.verts_packed(),int(num_sample_points/2))
        gt_points_uniform = sample_points(gt_mesh_torch.verts_packed(), int(num_sample_points/2))
        points_uniform = np.concatenate([rec_points_uniform, gt_points_uniform], axis=0)

        rec_mesh_occupancies = (inside_mesh.check_mesh_contains(rec_mesh, points_uniform) >= 0.5)
        rec_pointcloud = points_uniform[rec_mesh_occupancies]
        rec_pointcloud = torch.tensor(rec_pointcloud, dtype=torch.float)
        rec_pointcloud_normalized = eval_utils.normalize_pointclouds(rec_pointcloud)
        
        gt_mesh_occupancies = (inside_mesh.check_mesh_contains(gt_mesh, points_uniform) >= 0.5)
        gt_pointcloud = points_uniform[gt_mesh_occupancies]
        gt_pointcloud = torch.tensor(gt_pointcloud, dtype=torch.float)
        gt_pointcloud_normalized = eval_utils.normalize_pointclouds(gt_pointcloud)
    
    else:
        raise ValueError("method not recognized")
    
    #chamfer_dist = pytorch3d.loss.chamfer_distance(rec_pointcloud_normalized.unsqueeze(0), gt_pointcloud_normalized.unsqueeze(0))
    chamfer_dist = eval_utils.chamfer_distance_kdtree(rec_pointcloud_normalized.unsqueeze(0), gt_pointcloud_normalized.unsqueeze(0))

    chamfer_dist = chamfer_dist[0].item()
    return chamfer_dist


# takes in 3 dicts, each keyed by instance name.
def evaluate(input_img_dict, reconstructions_dict, gt_shapes_dict, results_output_path, metrics_to_eval, device, evaluations_df=None):

    if evaluations_df is None:
        evaluations_df = pd.DataFrame()

    # TODO: can this loop be parallelized?
    for instance in tqdm(reconstructions_dict, file=utils.TqdmPrintEvery()):
        print(instance)

        rec_obj_path = reconstructions_dict[instance]
        input_img_path = input_img_dict[instance]
        gt_obj_path = gt_shapes_dict[instance]
        instance_record = {**{metric:-1 for metric in metrics_to_eval},
                           **{"instance": instance, "input_img_path": input_img_path, "rec_obj_path": rec_obj_path, "gt_obj_path": gt_obj_path}}

        # IndexError is thrown by trimesh if meshes have nan vertices
        try:
            rec_mesh = trimesh.load(rec_obj_path)
        except IndexError:
            print("WARNING: instance {} had NaN nodes.".format(instance))
            evaluations_df = evaluations_df.append(instance_record, ignore_index=True)
            evaluations_df.to_pickle(results_output_path)
            continue

        input_img = Image.open(input_img_path)
        with torch.no_grad():
            rec_mesh_torch = utils.load_untextured_mesh(rec_obj_path, device)

        gt_mesh = trimesh.load(gt_obj_path)
        with torch.no_grad():
            gt_mesh_torch = utils.load_untextured_mesh(gt_obj_path, device)

        # TODO: should both meshes be watertight?
        #print(rec_mesh.is_watertight)
        #print(gt_mesh.is_watertight)

        if "2d_iou" in metrics_to_eval:
            instance_record["2d_iou"] = compute_iou_2d(rec_mesh_torch, input_img, device)
        if "3d_iou" in metrics_to_eval:
            instance_record["3d_iou"] = compute_iou_3d(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch)
        if "3d_iou_norm" in metrics_to_eval:
            instance_record["3d_iou_norm"] = compute_iou_3d(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, full_unit_normalize=True)
        if "chamfer_L1" in metrics_to_eval:
            instance_record["chamfer_L1"] = compute_chamfer_L1(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch)
        if "chamfer_L1_norm":
            instance_record["chamfer_L1_norm"] = compute_chamfer_L1(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, full_unit_normalize=True)

        evaluations_df = evaluations_df.append(instance_record, ignore_index=True)
        evaluations_df.to_pickle(results_output_path)
    

# images need to be transparent (masked) .pngs
# meshes need to be watertight .objs (also in a unit box? )
# assumes gt meshes and reconstructions meshes are aligned and normalized 
# (i.e. no further alignment/processing is done before calculating performancemetrics)
if __name__ == "__main__":
    all_metrics_list = ["2d_iou", "3d_iou", "3d_iou_norm", "chamfer_L1", "chamfer_L1_norm"]

    parser = argparse.ArgumentParser(description='Evaluates according to a configuration file')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--metrics', nargs='+', default=all_metrics_list, help='Gpu number to use.')
    parser.add_argument('--recompute', action='store_true', help='Recompute entries, even for ones which already exist.')
    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in all_metrics_list:
            raise ValueError("Metric {} is unknown.".format(metric))
    print("Metrics to evaluate: {}".format(args.metrics))

    # processing cfg file
    cfg = utils.load_config(args.cfg_path)
    input_img_dict = {str(filename).split('/')[-1].split('.')[0]:str(filename) for filename in list(Path(cfg['input_images']).rglob('*.png'))}
    reconstructions_dict = {str(filename).split('/')[-1].split('.')[0]:str(filename) for filename in list(Path(cfg['reconstructions']).rglob('*.obj'))}
    gt_shapes_dict = {}
    with open(cfg['gt_shapes'], 'r') as f:
        f = f.read().split('\n')
        for line in f:
            if line != "":
                gt_shapes_dict[line.split(" ")[0]] = line.split(" ")[1]

    # TODO: seed doesn't seem to work properly
    np.random.seed(0)
    device = torch.device("cuda:"+str(args.gpu))
    output_results_path = args.cfg_path.replace(".yaml", "_eval_results.pkl")
    if not args.recompute and os.path.exists(output_results_path):
        previous_evaluations_df = pd.read_pickle(output_results_path)
        previous_instances = list(previous_evaluations_df['instance'])
        for instance in list(reconstructions_dict.keys()):
            if instance in previous_instances:
                del reconstructions_dict[instance]
    else:
        previous_evaluations_df = None
    

    evaluate(input_img_dict, reconstructions_dict, gt_shapes_dict, output_results_path, args.metrics, device, evaluations_df=previous_evaluations_df)
