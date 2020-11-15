
import torch
from PIL import Image
import numpy as np
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt
import PIL

from utils import general_utils
# Util function for loading meshes
import pytorch3d
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
import torch.nn.functional as F

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
    softmax_rgb_blend,
    HardGouraudShader
)

from pytorch3d.renderer.mesh.shading import flat_shading


def show_renders(renders_batch, masks=True):
    if masks:
        num_renders = len(renders_batch)
    else:
        num_renders = renders_batch.shape[0]
    imgs_per_row = 8
    num_rows = int(np.ceil(num_renders/imgs_per_row))
    fig, ax = plt.subplots(nrows=num_rows, ncols=max(num_renders, imgs_per_row), squeeze=False, figsize=(2*imgs_per_row,2*num_rows))

    for i in range(num_renders):
        col_i = int(np.floor(i/imgs_per_row))
        if masks:
            ax[col_i][i%imgs_per_row].imshow(renders_batch[i].detach().cpu().numpy(), cmap="Greys")
        else:
            ax[col_i][i%imgs_per_row].imshow(renders_batch[i, ..., :3].detach().cpu().numpy())
        ax[col_i][i%imgs_per_row].xaxis.set_visible(False)
        ax[col_i][i%imgs_per_row].yaxis.set_visible(False)
    plt.pause(0.05)


from matplotlib.backends.backend_agg import FigureCanvasAgg


def render_mesh_HQ(mesh, R, T, device, img_size=512, aa_factor=10):
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

def reduce_aa_img_size(input_np_img, aa_factor=10):
    np_img = np.uint8(input_np_img*255)
    img = Image.fromarray(np_img)
    img = img.resize((224,224), resample=PIL.Image.LANCZOS)
    return img

# displays meshes at the predicted pose
def show_refinement_results(input_image, mesh_original, mesh_processed, pred_dist, pred_elev, pred_azim, device, num_novel_view=3, img_size=224):
  

    R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim)
    mesh_original_render = render_mesh_HQ(mesh_original, R, T, device, img_size=img_size)
    mesh_processed_render = render_mesh_HQ(mesh_processed, R, T, device, img_size=img_size)
    
    # rendering processed mesh at poses other than the predicted pose
    novel_view_renders = []
    for i in range(num_novel_view):
        R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim + ((i+1)*45))
        novel_view_renders.append(render_mesh_HQ(mesh_processed, R, T, device, img_size=img_size))
    
    # visualizing
    num_columns = 3 + num_novel_view
    fig, ax = plt.subplots(nrows=1, ncols=num_columns, squeeze=False, figsize=(16,3))

    col_i = 0
    # TODO: show on black bg
    ax[0][col_i].imshow(input_image)
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)

    col_i += 1
    #ax[0][col_i].imshow(reduce_aa_img_size(mesh_original_render[0, ..., :3].detach().cpu().numpy()))
    ax[0][col_i].imshow(np.clip(mesh_original_render[0, ..., :3].detach().cpu().numpy(),0,1))
    #ax[0][col_i].imshow(mesh_original_render[0, ..., :3].detach().cpu().numpy())
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)

    col_i += 1
    #ax[0][col_i].imshow(reduce_aa_img_size(mesh_processed_render[0, ..., :3].detach().cpu().numpy()))
    ax[0][col_i].imshow(np.clip(mesh_processed_render[0, ..., :3].detach().cpu().numpy(),0,1))
    #ax[0][col_i].imshow(mesh_processed_render[0, ..., :3].detach().cpu().numpy())
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)
    
    col_i += 1
    for i in range(num_novel_view):
        #ax[0][col_i+i].imshow(reduce_aa_img_size(novel_view_renders[i][0, ..., :3].detach().cpu().numpy()))
        ax[0][col_i+i].imshow(np.clip(novel_view_renders[i][0, ..., :3].detach().cpu().numpy(),0,1))
        #ax[0][col_i+i].imshow(novel_view_renders[i][0, ..., :3].detach().cpu().numpy())
        ax[0][col_i+i].xaxis.set_visible(False)
        ax[0][col_i+i].yaxis.set_visible(False)
    plt.pause(0.05)

    # https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    fig.tight_layout(pad=0.5)

    # To remove the huge white borders
    #for axis in ax.flatten():
    #    axis.margins(0)

    #fig.savefig("temp.jpg")
    #image_from_plot = Image.open("temp.jpg")
    

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image_from_plot