import os
from pathlib import Path
import pprint
import multiprocessing
import time
import io

import torch
from glob import glob
import pickle
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
import pytorch3d

class TqdmPrintEvery(io.StringIO):
    """
        Output stream for TQDM which will output to stdout. Used for nautilus jobs.
    """
    def __init__(self):
        super(TqdmPrintEvery, self).__init__()
        self.buf = None

    def write(self,buf):
        self.buf = buf

    def flush(self):
        print(self.buf)

tqdm_out = TqdmPrintEvery()

def load_mesh_as_points(mesh_path):
    mesh_num_verts = 1502
    cpu_device = torch.device("cpu")
    mesh = load_objs_as_meshes([mesh_path], device=cpu_device, load_textures = False)
    points = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=mesh_num_verts)
    return points

real_shapes_dir = "/home/svcl-oowl/dataset/ShapeNetCore.v1/03001627"
real_image_paths_cache_path = "caches/real_shapes_paths.p"
recompute_cache = False
output_dir = "data/real_sampled_points/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
parallel = True

# getting list of .objs
if recompute_cache or not os.path.exists(real_image_paths_cache_path):
    real_shapes_paths = list(Path(real_shapes_dir).rglob('model_watertight.obj'))
    pickle.dump(real_shapes_paths, open(real_image_paths_cache_path, 'wb'))
else:
    real_shapes_paths = pickle.load(open(real_image_paths_cache_path, 'rb'))


if parallel:
    total_points = process_map(load_mesh_as_points, real_shapes_paths, max_workers=8, chunksize=10, file=tqdm_out)

else:
    total_points = []
    cpu_device = torch.device("cpu")
    mesh_num_verts = 1502
    with tqdm(total=len(real_shapes_paths), file=tqdm_out) as pbar:
        for real_shape_path in real_shapes_paths:
            mesh = load_objs_as_meshes([real_shape_path], device=cpu_device, load_textures = False)
            points = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=mesh_num_verts)
            total_points.append(points)
            pbar.update(1)
            pbar.refresh()


total_points = torch.cat(total_points, dim=0)
torch.save(total_points, os.path.join(output_dir, "shapenet_chairs_1502_4.pt"))