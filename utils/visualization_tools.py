
import torch
from PIL import Image
import numpy as np
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt

from utils import utils


# displays meshes at the predicted pose
def show_refinement_results(input_image, mesh_original, mesh_processed, pred_dist, pred_elev, pred_azim, device, num_novel_view=3):
  

    R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim)
    mesh_original_render = utils.render_mesh(mesh_original, R, T, device)
    mesh_processed_render = utils.render_mesh(mesh_processed, R, T, device)
    
    # rendering processed mesh at poses other than the predicted pose
    novel_view_renders = []
    for i in range(num_novel_view):
        R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim + ((i+1)*45))
        novel_view_renders.append(utils.render_mesh(mesh_processed, R, T, device))
    
    # visualizing
    num_columns = 3 + num_novel_view
    fig, ax = plt.subplots(nrows=1, ncols=num_columns, squeeze=False, figsize=(15,5))
    col_i = 0
    # TODO: show on black bg
    ax[0][col_i].imshow(input_image)
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)

    col_i += 1
    ax[0][col_i].imshow(mesh_original_render[0, ..., :3].cpu().numpy())
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)

    col_i += 1
    ax[0][col_i].imshow(mesh_processed_render[0, ..., :3].cpu().numpy())
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)
    
    col_i += 1
    for i in range(num_novel_view):
        ax[0][col_i+i].imshow(novel_view_renders[i][0, ..., :3].cpu().numpy())
        ax[0][col_i+i].xaxis.set_visible(False)
        ax[0][col_i+i].yaxis.set_visible(False)
    plt.pause(0.05)