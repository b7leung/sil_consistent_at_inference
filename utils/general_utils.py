import yaml
import glob
import io
import os

import torch
import numpy as np
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from pytorch3d.renderer import (
    look_at_view_transform,
    OpenGLPerspectiveCameras, 
    PointLights, 
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
import plotly.express as px
import plotly


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


# returns a list of all cfgs inherited
def get_all_inherited_cfgs(highest_cfg_path):
    curr_cfg_path = highest_cfg_path
    all_inherited_cfg_paths = [highest_cfg_path]
    curr_cfg_inherits = True
    while curr_cfg_inherits:
        with open(curr_cfg_path, 'r') as f:
            curr_cfg = yaml.load(f, Loader=yaml.FullLoader)
            if "inherit_from" in curr_cfg:
                parent_cfg_path = curr_cfg["inherit_from"]
                all_inherited_cfg_paths.append(parent_cfg_path)
                curr_cfg_path = parent_cfg_path
            else:
                curr_cfg_inherits = False
    return all_inherited_cfg_paths
                

# given the path of a dir with .obj meshes, and path of a dir with .png images
def get_instances(input_dir_mesh, input_dir_img=None):
    instance_names = []
    for mesh_path in glob.glob(os.path.join(input_dir_mesh, "*.obj")):
        if input_dir_img is not None:
            img_path = os.path.join(input_dir_img, mesh_path.split('/')[-1].replace("obj", "png"))
            if not os.path.exists(img_path):
                raise ValueError("Couldn't find image for mesh {}.".format(mesh_path))
        instance_names.append(mesh_path.split('/')[-1][:-4])
    instance_names = sorted(instance_names)
    return instance_names


# based on https://github.com/facebookresearch/pytorch3d/issues/51
def load_untextured_mesh(mesh_path, device):
    mesh = load_objs_as_meshes([mesh_path], device=device, load_textures=False)
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
def render_mesh(mesh, R, T, device, img_size=512, silhouette=False, custom_lights="", differentiable=True):
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
        if custom_lights == "":
            lights = PointLights(device=device, location=[[0.0, 5.0, -10.0]])
        elif type(custom_lights) == str and custom_lights == "ambient":
            ambient = 0.5
            diffuse = 0
            specular = 0
            lights = PointLights(device=device, ambient_color=((ambient, ambient, ambient), ), diffuse_color=((diffuse, diffuse, diffuse), ),
                                 specular_color=((specular, specular, specular), ), location=[[0.0, 5.0, -10.0]])
        else:
            lights = custom_lights

        if differentiable:
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
        else:
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

    rendered_images = renderer(mesh, cameras=cameras)
    return rendered_images


# for batched rendering of many images
# mesh parameter is assumed to be in the cpu.
def batched_render(mesh, azims, elevs, dists, batch_size, device, silhouette=False, img_size=512):
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
        batch_renders = render_mesh(meshes, R, T, device, silhouette=silhouette, img_size=img_size)
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


# https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch
def weights_init_normal(m, var=0.0001):
    '''Takes in a module and initializes all linear layers with weight
        values taken from a normal distribution.'''

    classname = m.__class__.__name__
    # for every Linear layer in a model
    if classname.find('Linear') != -1:
        y = m.in_features
         # m.weight.data shoud be taken from a normal distribution
        m.weight.data.normal_(0.0,0.0001)
        # m.bias.data should be 0
        m.bias.data.fill_(0)


class InfiniteDataLoader():
    # based on https://discuss.pytorch.org/t/infinite-dataloader/17903/11
    # a wrapper for a dataloader, which allows batches to be drawn from it infinitely.
    # the order will be random if the dataloder is set to shuffle=True; else it will not be random
    # it's also recommended that drop_last=True if multiple infinite dataloaders are being used together (eg for a GAN)
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataloader_iter = iter(self.dataloader)
    
    def get_batch(self):
        try:
            data = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            data = next(self.dataloader_iter)
        return data


def normalize_pointcloud(pc):
    # centering
    norm_pc = pc - np.mean(pc, axis=0)
    # scaling to unit
    max_vert_values = np.amax(pc, axis=0)
    min_vert_values = np.amin(pc, axis=0)
    max_width = np.amax(max_vert_values-min_vert_values)
    norm_pc = norm_pc * (1/max_width)
    return norm_pc


# does not work for batch; only [n_verts,3]
def normalize_pointcloud_tensor(pc):
    axis = 0
    # centering
    norm_pc = pc - torch.mean(pc, axis=axis)
    # scaling to unit
    max_vert_values = torch.max(pc, axis).values
    min_vert_values = torch.min(pc, axis).values
    max_width = torch.max(max_vert_values-min_vert_values)
    norm_pc = norm_pc * (1/max_width)
    return norm_pc


# plots a [n_verts, 3] numpy/tensor pointcloud
def plot_pointcloud(pointcloud, normalized=True):
    if normalized:
        fig = px.scatter_3d(x=pointcloud[:,0], y=pointcloud[:,1], z = pointcloud[:,2], opacity=0.7, range_x=[-0.5,0.5], range_y=[-0.5,0.5], range_z=[-0.5,0.5])
    else:
        fig = px.scatter_3d(x=pointcloud[:,0], y=pointcloud[:,1], z = pointcloud[:,2], opacity=0.7)

    # based on https://stackoverflow.com/questions/61371324/scatter-3d-in-plotly-express-colab-notebook-is-not-plotting-with-equal-axes
    fig.update_layout(scene=dict(aspectmode="cube"))
    fig.update_traces(marker=dict(size=5, line={"width":1}))
    fig.show()



