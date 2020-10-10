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
from scipy import ndimage
import cv2

from utils import utils
from utils import eval_utils
from utils.brute_force_pose_est import brute_force_estimate_pose, get_iou
import inside_mesh
# https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/eval.py


def sample_points_old(mesh_verts, num_points, margin=0):
    max_vert_values = torch.max(mesh_verts, 0).values.cpu().numpy()
    min_vert_values = torch.min(mesh_verts, 0).values.cpu().numpy()
    points_uniform = np.random.uniform(low=min_vert_values-margin,high=max_vert_values+margin,size=(num_points,3))
    return points_uniform


# samples points uniformly in a bounding box of the union of 2 pytorch meshes
def sample_points_pytorch(mesh1, mesh2, num_points, margin=0.5):
    rec_max_vert_values = torch.max(mesh1.verts_packed(), 0).values.cpu().numpy()
    gt_max_vert_values = torch.max(mesh2.verts_packed(), 0).values.cpu().numpy()
    max_vert_values = np.maximum(rec_max_vert_values, gt_max_vert_values)

    rec_min_vert_values = torch.min(mesh1.verts_packed(), 0).values.cpu().numpy()
    gt_min_vert_values = torch.min(mesh2.verts_packed(), 0).values.cpu().numpy()
    min_vert_values = np.minimum(rec_min_vert_values, gt_min_vert_values)

    points = np.random.uniform(low=min_vert_values-margin,high=max_vert_values+margin,size=(num_points,3))
    return points


# samples points uniformly in a bounding box of the union of 2 pytorch meshes
def sample_points(mesh1, mesh2, num_points, margin=0.5):

    mesh1_max_vert_values = np.amax(mesh1.vertices, axis=0)
    mesh2_max_vert_values = np.amax(mesh2.vertices, axis=0)
    max_vert_values = np.maximum(mesh1_max_vert_values, mesh2_max_vert_values)

    mesh1_min_vert_values = np.amin(mesh1.vertices, axis=0)
    mesh2_min_vert_values = np.amin(mesh2.vertices, axis=0)
    min_vert_values = np.minimum(mesh1_min_vert_values, mesh2_min_vert_values)

    points = np.random.uniform(low=min_vert_values-margin,high=max_vert_values+margin,size=(num_points,3))
    return points


def compute_iou_2d(rec_mesh_torch, input_img, device, num_azims=20, num_elevs=20, num_dists=20):
    mask = np.asarray(input_img)[:,:,3] > 0
    _, _, _, render, iou = brute_force_estimate_pose(rec_mesh_torch, mask, num_azims, num_elevs, num_dists, device)
    return iou, []


def compute_iou_2d_given_pose(rec_mesh_torch, input_img, device, azim, elev, dist):

    warnings = []
    input_img_mask = torch.tensor(np.asarray(input_img))[:,:,3] > 0
    input_img_mask = center_and_resize_mask(input_img_mask)
    R, T = look_at_view_transform(dist, elev, azim) 
    render = utils.render_mesh(rec_mesh_torch, R, T, device, img_size=input_img_mask.shape[0])[0]
    render_mask = center_and_resize_mask(render[:,:,3] > 0)

    if render_mask is None:
        iou = 0
        warnings.append("iou_2d_input: no bbox found")
    else:
        iou = get_mask_iou(input_img_mask, render_mask)

    #iou = get_iou(render.cpu().numpy(), mask)   
    return iou, warnings, (input_img_mask, render_mask)


# assums mask is a tensor
# throws indexerror if no bbox found
def center_and_resize_mask(mask):

    try:
        mask = mask.cpu().numpy()
        mask = mask.astype(np.uint8)
        img_size = mask.shape[0]
        # Get the height and width of bbox
        objs = ndimage.find_objects(mask)
        # upper left, lower right
        img_bbox = [objs[0][0].start, objs[0][1].start, objs[0][0].stop, objs[0][1].stop]
        cropped_mask = mask[img_bbox[0]:img_bbox[2],img_bbox[1]:img_bbox[3]]
        cropped_height = cropped_mask.shape[0]
        cropped_width = cropped_mask.shape[1]
        if cropped_width > cropped_height:
            scale_factor = float(img_size)/cropped_width
            resized_mask = cv2.resize(cropped_mask, (int(cropped_width*scale_factor), int(cropped_height*scale_factor)))
        else:
            scale_factor = float(img_size)/cropped_height
            resized_mask = cv2.resize(cropped_mask, (int(cropped_width*scale_factor), int(cropped_height*scale_factor)))
        
        resized_height = resized_mask.shape[0]
        resized_width = resized_mask.shape[1]
        # https://stackoverflow.com/questions/59525640/how-to-center-the-content-object-of-a-binary-image-in-python
        normalized_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        x = img_size//2 - resized_width//2
        y = img_size//2 - resized_height//2
        normalized_mask[y:y+resized_height, x:x+resized_width] = resized_mask

        normalized_mask = torch.tensor(normalized_mask > 0)
        return normalized_mask
    except IndexError:
        print("no bbox found")
        return None



def get_mask_iou(render1_mask, render2_mask):
    #render_1_mask = center_and_resize_mask(render1_mask)
    #render_2_mask = center_and_resize_mask(render2_mask)
    intersection = torch.logical_and(render1_mask, render2_mask)
    union = torch.logical_or(render1_mask, render2_mask)
    IOU = torch.sum(intersection, dtype=torch.float) / torch.sum(union, dtype=torch.float)
    return IOU.item()


def compute_iou_2d_multi(rec_mesh_torch, gt_mesh_torch, device, num_azims=8, num_elevs=3):
    # 0.,  45.,  90., 135., 180., 225., 270., 315.
    azims = torch.linspace(0, 360, num_azims+1)[:-1].repeat(num_elevs)
    elevs = torch.repeat_interleave(torch.tensor([-45, 0, 45]), num_azims) # TODO: also add underneath elevs
    dists = torch.ones(num_azims*num_elevs) * 1

    gt_renders = utils.batched_render(gt_mesh_torch, azims, elevs, dists, 8, device, False, 224, False)
    rec_renders = utils.batched_render(rec_mesh_torch, azims, elevs, dists, 8, device, False, 224, False)


    iou_2d_scores = []
    avg_2d_iou = 0
    gt_masks = []
    rec_masks = []
    warnings = []
    for render_i in range(num_azims*num_elevs):
        gt_mask = center_and_resize_mask(gt_renders[render_i][:,:,3] > 0)
        rec_mask = center_and_resize_mask(rec_renders[render_i][:,:,3] > 0)
        if gt_mask is not None and rec_mask is not None:
            gt_masks.append(gt_mask)
            rec_masks.append(rec_mask)
            iou_2d_score = get_mask_iou(gt_mask, rec_mask)
        else:
            # no bbox found
            iou_2d_score = 0
            warnings.append("iou_2d_multi: no bbox found")
        avg_2d_iou += iou_2d_score
        iou_2d_scores.append(iou_2d_scores)
    avg_2d_iou = avg_2d_iou / (num_azims*num_elevs)

    return avg_2d_iou, warnings, (gt_masks, rec_masks, iou_2d_scores)



def compute_iou_3d(rec_mesh_input, rec_mesh_torch, gt_mesh_input, gt_mesh_torch, num_sample_points=900000, full_unit_normalize=False, points=None):
    status = []
    rec_mesh = rec_mesh_input.copy()
    gt_mesh = gt_mesh_input.copy()
    if full_unit_normalize:
        if points is None:
            points = np.random.uniform(low=-1,high=1,size=(num_sample_points,3))
        rec_mesh.vertices = eval_utils.normalize_pointclouds_numpy_all_axes(rec_mesh.vertices)
        gt_mesh.vertices = eval_utils.normalize_pointclouds_numpy_all_axes(gt_mesh.vertices)
    else:

        if points is None:
            #rec_points_uniform = sample_points(rec_mesh_torch.verts_packed(),int(num_sample_points/2), 0)
            #gt_points_uniform = sample_points(gt_mesh_torch.verts_packed(), int(num_sample_points/2), 0)
            #points = np.concatenate([rec_points_uniform, gt_points_uniform], axis=0)

            points = sample_points(gt_mesh, gt_mesh, num_sample_points)
            #points = sample_points(rec_mesh, gt_mesh, num_sample_points)
            #points = sample_points_pytorch(rec_mesh_torch, gt_mesh_torch, num_sample_points)


        rec_mesh.vertices = eval_utils.normalize_pointclouds_numpy(rec_mesh.vertices)
        gt_mesh.vertices = eval_utils.normalize_pointclouds_numpy(gt_mesh.vertices)

    rec_mesh_occupancies = inside_mesh.check_mesh_contains(rec_mesh, points)
    gt_mesh_occupancies = inside_mesh.check_mesh_contains(gt_mesh, points)
    if sum(rec_mesh_occupancies) < 200:
        status.append("compute_iou_3d: rec mesh < 200 occupancies")
    if sum(gt_mesh_occupancies) < 200:
        status.append("compute_iou_3d: gt mesh < 200 occupancies")

    iou_3d = eval_utils.compute_iou(rec_mesh_occupancies, gt_mesh_occupancies)

    return iou_3d, status


def compute_chamfer_L1(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, sample_method="sample_surface", num_sample_points=900000, full_unit_normalize=False):
    # NOTE: a inferring a pointcloud will not always work, for thin meshes like shapenet chair d2815e678f173616e6cfc789522bfbab
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
        #rec_points_uniform = sample_points(rec_mesh_torch.verts_packed(),int(num_sample_points/2), 0.5)
        #gt_points_uniform = sample_points(gt_mesh_torch.verts_packed(), int(num_sample_points/2), 0.5)
        #points_uniform = np.concatenate([rec_points_uniform, gt_points_uniform], axis=0)
        try:
            points_uniform = sample_points(rec_mesh, gt_mesh, num_sample_points)

            rec_mesh_occupancies = (inside_mesh.check_mesh_contains(rec_mesh, points_uniform) >= 0.5)
            rec_pointcloud = points_uniform[rec_mesh_occupancies]
            rec_pointcloud = torch.tensor(rec_pointcloud, dtype=torch.float)
            rec_pointcloud_normalized = eval_utils.normalize_pointclouds(rec_pointcloud)
            
            gt_mesh_occupancies = (inside_mesh.check_mesh_contains(gt_mesh, points_uniform) >= 0.5)
            gt_pointcloud = points_uniform[gt_mesh_occupancies]
            gt_pointcloud = torch.tensor(gt_pointcloud, dtype=torch.float)
            gt_pointcloud_normalized = eval_utils.normalize_pointclouds(gt_pointcloud)
        except RuntimeError:
            # this exception is meant to catch case where no occupancies could be found.
            return None
    
    else:
        raise ValueError("method not recognized")
    
    #chamfer_dist = pytorch3d.loss.chamfer_distance(rec_pointcloud_normalized.unsqueeze(0), gt_pointcloud_normalized.unsqueeze(0))
    chamfer_dist = eval_utils.chamfer_distance_kdtree(rec_pointcloud_normalized.unsqueeze(0), gt_pointcloud_normalized.unsqueeze(0))

    chamfer_dist = chamfer_dist[0].item()
    return chamfer_dist, []


# takes in 3 dicts, each keyed by instance name.
# TODO: should both meshes be watertight?
def evaluate(input_img_dict, reconstructions_dict, gt_shapes_dict, results_output_path, metrics_to_eval, cached_pred_poses, device, evaluations_df=None):

    if evaluations_df is None:
        evaluations_df = pd.DataFrame()

    #eval_log_path = results_output_path.replace(".pkl", ".log")

    # TODO: can this loop be parallelized?
    for instance in tqdm(reconstructions_dict, file=utils.TqdmPrintEvery()):
        print(instance)

        rec_obj_path = reconstructions_dict[instance]
        input_img_path = input_img_dict[instance]
        gt_obj_path = gt_shapes_dict[instance]
        instance_record = {**{metric:-1 for metric in metrics_to_eval},
                           **{"instance": instance, "input_img_path": input_img_path, "rec_obj_path": rec_obj_path, "gt_obj_path": gt_obj_path, "eval_warnings":[]}}

        # IndexError is thrown by trimesh if meshes have nan vertices
        try:
            # note: the process=False flag is important to keep the mesh watertight
            # https://github.com/hjwdzh/ManifoldPlus/issues/6
            rec_mesh = trimesh.load(rec_obj_path, process=False, force="mesh")
        except IndexError:
            #print("WARNING: instance {} had NaN nodes.".format(instance))
            #with open(eval_log_path, "a") as f:
            #    f.write("WARNING: instance {} had NaN nodes.\n".format(instance))
            instance_record["eval_warnings"].append("NaN nodes")
            evaluations_df = evaluations_df.append(instance_record, ignore_index=True)
            evaluations_df.to_pickle(results_output_path)
            continue

        input_img = Image.open(input_img_path)
        with torch.no_grad():
            rec_mesh_torch = utils.load_untextured_mesh(rec_obj_path, device)

        gt_mesh = trimesh.load(gt_obj_path, process=False, force="mesh")
        with torch.no_grad():
            gt_mesh_torch = utils.load_untextured_mesh(gt_obj_path, device)

        if not rec_mesh.is_watertight:
            instance_record["eval_warnings"].append("rec_obj_path is not watertight")
        if not gt_mesh.is_watertight:
            instance_record["eval_warnings"].append("gt_obj_path is not watertight")

        if "2d_iou_input" in metrics_to_eval:
            if cached_pred_poses == {}:
                iou_2d, warnings = compute_iou_2d(rec_mesh_torch, input_img, device)
                raise
            else:
                azim = cached_pred_poses[instance]["azim"]
                elev = cached_pred_poses[instance]["elev"]
                dist = cached_pred_poses[instance]["dist"]
                iou_2d, warnings, _ = compute_iou_2d_given_pose(rec_mesh_torch, input_img, device, azim, elev, dist)
            instance_record["2d_iou_input"] = iou_2d
            instance_record["eval_warnings"] += warnings

        if "2d_iou_multi" in metrics_to_eval:
            iou_2d_multi, warnings, _ = compute_iou_2d_multi(rec_mesh_torch, gt_mesh_torch, device)
            instance_record["2d_iou_multi"] = iou_2d_multi
            instance_record["eval_warnings"] += warnings

        if "3d_iou" in metrics_to_eval:
            iou_3d, warnings = compute_iou_3d(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch)
            instance_record["3d_iou"] = iou_3d
            instance_record["eval_warnings"] += warnings

        if "3d_iou_norm" in metrics_to_eval:
            iou_3d_norm, warnings = compute_iou_3d(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, full_unit_normalize=True)
            instance_record["3d_iou_norm"] = iou_3d_norm
            instance_record["eval_warnings"] += warnings

        if "chamfer_L1" in metrics_to_eval:
            chamfer_l1, warnings = compute_chamfer_L1(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, sample_method="sample_surface")
            instance_record["chamfer_L1"] = chamfer_l1
            instance_record["eval_warnings"] += warnings
            #if instance_record["chamfer_L1"] is None:
            #    instance_record = {**{metric:-1 for metric in metrics_to_eval},
            #        **{"instance": instance, "input_img_path": input_img_path, "rec_obj_path": rec_obj_path, "gt_obj_path": gt_obj_path}}
            #    print("WARNING: couldn't compute chamfer_L1 for instance {}".format(instance))
            #    with open(eval_log_path, "a") as f:
            #        f.write("WARNING: couldn't compute chamfer_L1 for instance {}\n".format(instance))

        if "chamfer_L1_norm" in metrics_to_eval:
            chamfer_l1_norm, warnings = compute_chamfer_L1(rec_mesh, rec_mesh_torch, gt_mesh, gt_mesh_torch, full_unit_normalize=True)
            instance_record["chamfer_L1_norm"] = chamfer_l1_norm
            instance_record["eval_warnings"] += warnings

        evaluations_df = evaluations_df.append(instance_record, ignore_index=True)
        evaluations_df.to_pickle(results_output_path)


    print(evaluations_df) 

# images need to be transparent (masked) .pngs
# meshes need to be watertight .objs (also in a unit box? )
# assumes gt meshes and reconstructions meshes are aligned and normalized 
# (i.e. no further alignment/processing is done before calculating performancemetrics)
# refined_mesh_dir needs to have a yaml file with dataset info filled out (input_dir_img, gt_shapes_lst_path)
if __name__ == "__main__":
    all_metrics_list = ["2d_iou_input", "2d_iou_multi", "3d_iou", "3d_iou_norm", "chamfer_L1", "chamfer_L1_norm"]

    parser = argparse.ArgumentParser(description='Evaluates meshes in a folder')
    parser.add_argument('refined_meshes_dir', type=str, help='Path to folder with refined meshes')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    parser.add_argument('--metrics', nargs='+', default=all_metrics_list, help='Gpu number to use.')
    parser.add_argument('--use_gt_poses', action='store_true', help='Perform refinements with the ground truth pose, located in the image dir.')
    parser.add_argument('--recompute', action='store_true', help='Recompute entries, even for ones which already exist.')
    parser.add_argument('--output_filename', type=str, default="eval_results", help='Name of output evaluation .pkl')
    args = parser.parse_args()

    for metric in args.metrics:
        if metric not in all_metrics_list:
            raise ValueError("Metric {} is unknown.".format(metric))
    print("Metrics to evaluate: {}".format(args.metrics))

    # processing cfg file
    #np.random.seed(0)
    device = torch.device("cuda:"+str(args.gpu))
    cfg = utils.load_config(glob.glob(os.path.join(args.refined_meshes_dir, "*.yaml"))[0])

    img_dir = cfg["dataset"]["input_dir_img"]
    input_img_dict = {str(filename).split('/')[-1].split('.')[0]:str(filename) for filename in list(Path(img_dir).rglob('*.png'))}
    reconstructions_dict = {str(filename).split('/')[-1].split('.')[0]:str(filename) for filename in list(Path(args.refined_meshes_dir).rglob('*.obj'))}
    gt_shapes_dict = {}
    gt_shapes_list_path = cfg["dataset"]["gt_shapes_lst_path"]
    with open(gt_shapes_list_path, 'r') as f:
        f = f.read().split('\n')
        for line in f:
            if line != "":
                gt_shapes_dict[line.split(" ")[0]] = line.split(" ")[1]

    # combining all cached predicted poses
    pred_poses_dict = {}
    if args.use_gt_poses:
        pred_poses_dict = pickle.load(open(os.path.join(img_dir, "renders_camera_params.pt"), "rb"))
        pred_poses_dict = {instance:pred_poses_dict[instance] for instance in pred_poses_dict}
    else:
        pred_pose_paths = list(Path(args.refined_meshes_dir).rglob('pred_poses.p'))
        for pred_pose_path in pred_pose_paths:
            curr_cache = pickle.load(open(pred_pose_path, "rb"))
            pred_poses_dict = {**pred_poses_dict, **curr_cache}

    output_results_path = os.path.join(args.refined_meshes_dir, "{}.pkl".format(args.output_filename))
    if not args.recompute and os.path.exists(output_results_path):
        previous_evaluations_df = pd.read_pickle(output_results_path)
        previous_instances = list(previous_evaluations_df['instance'])
        for instance in list(reconstructions_dict.keys()):
            if instance in previous_instances:
                del reconstructions_dict[instance]
    else:
        previous_evaluations_df = None
    
    evaluate(input_img_dict, reconstructions_dict, gt_shapes_dict, output_results_path, args.metrics, pred_poses_dict, device, evaluations_df=previous_evaluations_df)
