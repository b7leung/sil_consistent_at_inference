# https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/common.py
# https://gist.github.com/LMescheder/b5e03ffd1bf8a0dfbb984cacc8c99532

import torch
import numpy as np

from im2mesh.utils.libkdtree import KDTree



# points: a tensor of 3d points
def normalize_pointclouds(points):
    max_vert_values = torch.max(points, 0).values
    min_vert_values = torch.min(points, 0).values
    max_width = torch.max(max_vert_values-min_vert_values)
    normalized_points = points * (1/max_width)

    return normalized_points


# points: a tensor of 3d points
def normalize_pointclouds_numpy_all_axes_slow(points):
    # TODO: This assumes the pointcloud is centered on the origin and there are pos and neg points on for every axis
    # not sure if this will always hold; may need to center pointcloud first
    normalized_points = np.copy(points)
    max_vert_values = np.amax(points, axis=0)
    min_vert_values = np.amin(points, axis=0)
    for i in range(normalized_points.shape[0]):

        x = normalized_points[i][0]
        if x > 0:
            x = x / max_vert_values[0]
        else:
            x = x / abs(min_vert_values[0])

        y = normalized_points[i][1]
        if y > 0:
            y = y / max_vert_values[1]
        else:
            y = y / abs(min_vert_values[1])

        z = normalized_points[i][2]
        if z > 0:
            z = z / max_vert_values[2]
        else:
            z = z / abs(min_vert_values[2])

        normalized_points[i] = np.array([x,y,z])

    return normalized_points


# points: a tensor of 3d points
# Proxy_points is a alternative, smaller pointcloud used to find the max/min of the points in order to normalize
# it is optional, but useful if there are many points, to avoid a O(n) search to find the min/max
def normalize_pointclouds_numpy_all_axes(points_to_normalize, proxy_points=None):
    # TODO: This assumes the pointcloud is centered on the origin and there are pos and neg points on for every axis
    # not sure if this will always hold; may need to center pointcloud first
    normalized_points = points_to_normalize
    if proxy_points is None:
        max_vert_values = np.amax(normalized_points, axis=0)
        min_vert_values = np.amin(normalized_points, axis=0)
    else:
        max_vert_values = np.amax(proxy_points, axis=0)
        min_vert_values = np.amin(proxy_points, axis=0)

    for i in range(3):
        normalized_points[:,i][normalized_points[:,i] > 0] = normalized_points[:,i][normalized_points[:,i] > 0]/max_vert_values[i]
        normalized_points[:,i][normalized_points[:,i] < 0] = normalized_points[:,i][normalized_points[:,i] < 0]/abs(min_vert_values[i])

    return normalized_points


# points: a tensor of 3d points
def normalize_pointclouds_numpy(points):
    max_vert_values = np.amax(points, axis=0)
    min_vert_values = np.amin(points, axis=0)
    max_width = np.amax(max_vert_values-min_vert_values)
    normalized_points = points * (1/max_width)

    return normalized_points


def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Put all data in second dimension
    # Also works for 1-dimensional data
    if occ1.ndim >= 2:
        occ1 = occ1.reshape(occ1.shape[0], -1)
    if occ2.ndim >= 2:
        occ2 = occ2.reshape(occ2.shape[0], -1)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=-1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=-1)

    iou = (area_intersect / area_union)

    return iou


def chamfer_distance_kdtree(points1, points2, give_id=False):
    ''' KD-tree based implementation of the Chamfer distance.
    Args:
        points1 (numpy array): first point set
        points2 (numpy array): second point set
        give_id (bool): whether to return the IDs of the nearest points
    '''
    # Points have size batch_size x T x 3
    batch_size = points1.size(0)

    # First convert points to numpy
    points1_np = points1.detach().cpu().numpy()
    points2_np = points2.detach().cpu().numpy()

    # Get list of nearest neighbors indieces
    idx_nn_12, _ = get_nearest_neighbors_indices_batch(points1_np, points2_np)
    idx_nn_12 = torch.LongTensor(idx_nn_12).to(points1.device)
    # Expands it as batch_size x 1 x 3
    idx_nn_12_expand = idx_nn_12.view(batch_size, -1, 1).expand_as(points1)

    # Get list of nearest neighbors indieces
    idx_nn_21, _ = get_nearest_neighbors_indices_batch(points2_np, points1_np)
    idx_nn_21 = torch.LongTensor(idx_nn_21).to(points1.device)
    # Expands it as batch_size x T x 3
    idx_nn_21_expand = idx_nn_21.view(batch_size, -1, 1).expand_as(points2)

    # Compute nearest neighbors in points2 to points in points1
    # points_12[i, j, k] = points2[i, idx_nn_12_expand[i, j, k], k]
    points_12 = torch.gather(points2, dim=1, index=idx_nn_12_expand)

    # Compute nearest neighbors in points1 to points in points2
    # points_21[i, j, k] = points2[i, idx_nn_21_expand[i, j, k], k]
    points_21 = torch.gather(points1, dim=1, index=idx_nn_21_expand)

    # Compute chamfer distance
    chamfer1 = (points1 - points_12).pow(2).sum(2).mean(1)
    chamfer2 = (points2 - points_21).pow(2).sum(2).mean(1)

    # Take sum
    chamfer = chamfer1 + chamfer2

    # If required, also return nearest neighbors
    if give_id:
        return chamfer1, chamfer2, idx_nn_12, idx_nn_21

    return chamfer


def get_nearest_neighbors_indices_batch(points_src, points_tgt, k=1):
    ''' Returns the nearest neighbors for point sets batchwise.
    Args:
        points_src (numpy array): source points
        points_tgt (numpy array): target points
        k (int): number of nearest neighbors to return
    '''
    indices = []
    distances = []

    for (p1, p2) in zip(points_src, points_tgt):
        kdtree = KDTree(p2)
        dist, idx = kdtree.query(p1, k=k)
        indices.append(idx)
        distances.append(dist)

    return indices, distances