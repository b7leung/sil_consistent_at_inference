import os
import sys
import yaml
import io
import glob
import pprint
import pickle
import random

import torch
from PIL import Image
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from skimage.transform import resize
from skimage import img_as_bool
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


# Util function for loading meshes
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardPhongShader,
    SoftSilhouetteShader,
    HardFlatShader,
    BlendParams,
    softmax_rgb_blend
)

from pytorch3d.renderer.mesh.shading import flat_shading

class SoftFlatShader(nn.Module):

    def __init__(self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
        texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = softmax_rgb_blend(colors, fragments, blend_params)
        return images



# General config
def load_config_dict(cfg_special, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself


    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg



# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f, Loader=yaml.FullLoader)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# based on https://github.com/facebookresearch/pytorch3d/issues/51
def load_untextured_mesh(mesh_path, device):
    mesh = load_objs_as_meshes([mesh_path], device=device, load_textures = False)
    verts, faces_idx, _ = load_obj(mesh_path)
    faces = faces_idx.verts_idx
    verts_rgb = torch.ones_like(verts)[None] # (1, V, 3)
    textures = Textures(verts_rgb=verts_rgb.to(device))
    mesh_no_texture = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
        )
    return mesh_no_texture


# for rendering a single image
#TODO: double check params for silhouette case
# https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb
def render_mesh(mesh, R, T, device, img_size=512, silhouette=False, soft_flat_shader=False, custom_lights=None):
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)


    if silhouette:
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )
    else:
        raster_settings = RasterizationSettings(
            image_size=img_size, 
            blur_radius=0.0, 
            faces_per_pixel=1, 
        )
        if custom_lights is None:
            lights = PointLights(device=device, location=[[0.0, 5.0, -10.0]])
        else:
            lights = custom_lights

        if soft_flat_shader:

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=HardFlatShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights
                )
            )
        
        else:

            renderer = MeshRenderer(
                rasterizer=MeshRasterizer(
                    cameras=cameras, 
                    raster_settings=raster_settings
                ),
                shader=SoftPhongShader(
                    device=device, 
                    cameras=cameras,
                    lights=lights
                )
            )

    rendered_images = renderer(mesh, cameras=cameras)
    return rendered_images


def render_mesh_HQ(mesh, R, T, device, img_size=512, aa_factor=4, silhouette=True):
    cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=img_size*aa_factor, 
        blur_radius=0.000, 
        faces_per_pixel=1, 
        cull_backfaces=True
    )
    ambient = 0.5
    diffuse = 0.4
    specular = 0.3
    lights = PointLights(device=device, ambient_color=((ambient, ambient, ambient), ), diffuse_color=((diffuse, diffuse, diffuse), ),
        specular_color=((specular, specular, specular), ), location=[[0.0, 5.0, -10.0]])
    #lights = DirectionalLights(device=device, ambient_color=((ambient, ambient, ambient), ), diffuse_color=((diffuse, diffuse, diffuse), ),
    #    specular_color=((specular, specular, specular), ), direction=[[0.0, 5.0, -10.0]])
    #lights = PointLights(device=device, ambient_color=((0.5, 0.5, 0.5), ), diffuse_color=((0.3, 0.3, 0.3), ), specular_color=((0.2, 0.2, 0.2), ), location=[[0.0, 5.0, -10.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images = renderer(mesh, cameras=cameras)
    images = images.permute(0, 3, 1, 2)  # NHWC -> NCHW
    images = F.avg_pool2d(images, kernel_size=aa_factor, stride=aa_factor)
    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC

    return images 

# for batched rendering of many images
# mesh parameter is assumed to be in the cpu.
def batched_render(mesh, azims, elevs, dists, batch_size, device, silhouette=False, img_size=512, soft_flat_shader=False):
    meshes = mesh.extend(batch_size)
    num_renders = azims.shape[0]
    renders = []
    for batch_i in (range(int(np.ceil(num_renders/batch_size)))):
        pose_idx_start = batch_i * batch_size
        pose_idx_end = min((batch_i+1) * batch_size, num_renders)
        batch_azims = azims[pose_idx_start:pose_idx_end]
        batch_elevs = elevs[pose_idx_start:pose_idx_end]
        batch_dists = dists[pose_idx_start:pose_idx_end]
        
        R, T = look_at_view_transform(batch_dists, batch_elevs, batch_azims) 
        if batch_azims.shape[0] != batch_size:
            meshes = mesh.extend(batch_azims.shape[0])
        batch_renders = render_mesh(meshes, R, T, device, silhouette=silhouette, img_size=img_size, soft_flat_shader=soft_flat_shader)
        renders.append(batch_renders.cpu())
    renders = torch.cat(renders)
    return renders


class BatchedRenderIterable:


    def __init__(self, mesh, azims, elevs, dists, batch_size, device, silhouette=False, img_size=512):
        self.mesh = mesh
        self.azims = azims
        self.elevs = elevs
        self.dists = dists
        self.batch_size = batch_size
        self.device = device
        self.silhouette = silhouette
        self.img_size = img_size

        self.meshes = mesh.extend(batch_size)
        self.num_renders = azims.shape[0]
        self.num_iterations_required = int(np.ceil(self.num_renders/self.batch_size))

        self.batch_i = 0


    def __iter__(self):
        return self
    

    def __next__(self):

        if self.batch_i < self.num_iterations_required:

            pose_idx_start = self.batch_i * self.batch_size
            pose_idx_end = min((self.batch_i+1) * self.batch_size, self.num_renders)
            batch_azims = self.azims[pose_idx_start:pose_idx_end]
            batch_elevs = self.elevs[pose_idx_start:pose_idx_end]
            batch_dists = self.dists[pose_idx_start:pose_idx_end]
            
            R, T = look_at_view_transform(batch_dists, batch_elevs, batch_azims) 
            if batch_azims.shape[0] != self.batch_size:
                meshes = self.mesh.extend(batch_azims.shape[0])
            else:
                meshes = self.meshes
            batch_renders = render_mesh(meshes, R, T, self.device, silhouette=self.silhouette, img_size=self.img_size)
            self.batch_i += 1

            return batch_renders.cpu()
        else:
            raise StopIteration


class TqdmPrintEvery(io.StringIO):
    #Output stream for TQDM which will output to stdout. Used for nautilus jobs.
    def __init__(self):
        super(TqdmPrintEvery, self).__init__()
        self.buf = None
    def write(self,buf):
        self.buf = buf
    def flush(self):
        print(self.buf)