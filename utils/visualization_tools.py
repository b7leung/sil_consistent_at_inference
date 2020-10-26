
import torch
from PIL import Image
import numpy as np
from pytorch3d.renderer import look_at_view_transform
import matplotlib.pyplot as plt

from utils import general_utils


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

# displays meshes at the predicted pose
def show_refinement_results(input_image, mesh_original, mesh_processed, pred_dist, pred_elev, pred_azim, device, num_novel_view=3, img_size=224):
  

    R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim)
    mesh_original_render = general_utils.render_mesh(mesh_original, R, T, device, img_size=img_size)
    mesh_processed_render = general_utils.render_mesh(mesh_processed, R, T, device, img_size=img_size)
    
    # rendering processed mesh at poses other than the predicted pose
    novel_view_renders = []
    for i in range(num_novel_view):
        R, T = look_at_view_transform(pred_dist, pred_elev, pred_azim + ((i+1)*45))
        novel_view_renders.append(general_utils.render_mesh(mesh_processed, R, T, device, img_size=img_size))
    
    # visualizing
    num_columns = 3 + num_novel_view
    fig, ax = plt.subplots(nrows=1, ncols=num_columns, squeeze=False, figsize=(16,3))

    col_i = 0
    # TODO: show on black bg
    ax[0][col_i].imshow(input_image)
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)

    col_i += 1
    ax[0][col_i].imshow(mesh_original_render[0, ..., :3].detach().cpu().numpy())
    #ax[0][col_i].imshow(mesh_original_render[0, ..., :3].cpu().numpy())
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)

    col_i += 1
    ax[0][col_i].imshow(mesh_processed_render[0, ..., :3].detach().cpu().numpy())
    #ax[0][col_i].imshow(mesh_processed_render[0, ..., :3].cpu().numpy())
    ax[0][col_i].xaxis.set_visible(False)
    ax[0][col_i].yaxis.set_visible(False)
    
    col_i += 1
    for i in range(num_novel_view):
        ax[0][col_i+i].imshow(novel_view_renders[i][0, ..., :3].detach().cpu().numpy())
        #ax[0][col_i+i].imshow(novel_view_renders[i][0, ..., :3].cpu().numpy())
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