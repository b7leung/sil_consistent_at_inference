import sys
import glob
import pprint
import pickle

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
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.io import load_obj
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
    SoftPhongShader
)

from utils import utils


# given a mesh and a mask, finds position of the camera by brute force
# there are two steps: 1) finding the azimuth and elevation and 2) find the distance
# the additional requirement that the entire mesh should fit in the image after rendered at the estimated pose is also enforced 
# (the distance chosen must satisfy this)
# inputs:
# - mesh: pytorch 3d mesh object; assumed to be in the cpu
# - mask: a boolean 2D array
def brute_force_estimate_pose(mesh, mask, num_azims, num_elevs, num_dists, device, batch_size=8):
    #torch.cuda.set_device(device)
    with torch.no_grad():
        # rendering at many different azimuth and elevation combinations on a sphere
        num_renders = num_elevs * num_azims
        azims = torch.linspace(0, 360, num_azims).repeat(num_elevs)
        elevs = torch.repeat_interleave(torch.linspace(0, 180, num_elevs), num_azims) # TODO: also add underneath elevs
        dists = torch.ones(num_renders) * 2.7

        # computing iou of mask and azimuth/elevation renders
        iou_calcs = []
        batched_renders_iterable = utils.BatchedRenderIterable(mesh, azims, elevs, dists, batch_size, device)
        for render_batch in iter(batched_renders_iterable):
            for render in render_batch:
                iou = get_normalized_iou(render.cpu().numpy(), mask, False)
                iou_calcs.append(iou)

        #renders = utils.batched_render(mesh, azims, elevs, dists, batch_size, device)
        #for i in range(renders.shape[0]):
        #    iou = get_normalized_iou(renders[i].cpu().numpy(), mask, False)
        #    iou_calcs.append(iou)
        #    #print("azim: {}, elev: {}, iou: {}".format(azims[i], elevs[i], iou))

        # selecting render pose with highest iou to find predicted azimuth and elevation
        iou_argsort = np.argsort(iou_calcs)[::-1]
        iou_highest_idx = iou_argsort[0]
        pred_azim = azims[iou_highest_idx]
        pred_elev = elevs[iou_highest_idx]
        #print("highest iou: {}, azim: {}, elev: {}".format(iou_calcs[iou_highest_idx], pred_azim, pred_elev))

        # interpolating between rendered distances to find best distance for predicted azimuth and elevation
        # TODO: add edge case where no render fits in frame
        azims = torch.ones(num_dists) * pred_azim
        elevs = torch.ones(num_dists) * pred_elev
        dists = torch.linspace(0.5, 3, num_dists)
        renders = utils.batched_render(mesh, azims, elevs, dists, batch_size, device)
        iou_calcs = []
        rendered_image_fits = []
        for i in range(renders.shape[0]):
            iou = get_iou(renders[i].cpu().numpy(), mask)
            iou_calcs.append(iou)
            rendered_image_fits.append(rgba_obj_in_frame(renders[i].cpu().numpy()))
        iou_argsort = np.argsort(iou_calcs)[::-1]
        rendered_image_fits = np.array(rendered_image_fits)[iou_argsort]

        # choose distance with highest iou, whose rendered image will fit completely in the frame
        i = 0
        while not rendered_image_fits[i]:
            i+=1
            # if nothing fits, just return the highest IoU even though it doesn't fit
            if i >= len(rendered_image_fits):
                print("Couldn't find a pose which fits entirely")
                i=0
                break

        pred_dist = dists[iou_argsort[i]]
        iou = iou_calcs[iou_argsort[i]]
        render = renders[iou_argsort[i]]
        
    return pred_azim, pred_elev, pred_dist, render, iou


def brute_force_estimate_dist(mesh, mask, azim, elev, num_dists, device, batch_size=8):
    with torch.no_grad():
        pred_azim = azim
        pred_elev = elev
        #print("highest iou: {}, azim: {}, elev: {}".format(iou_calcs[iou_highest_idx], pred_azim, pred_elev))

        # interpolating between rendered distances to find best distance for predicted azimuth and elevation
        # TODO: add edge case where no render fits in frame
        azims = torch.ones(num_dists) * pred_azim
        elevs = torch.ones(num_dists) * pred_elev
        dists = torch.linspace(0.5, 3, num_dists)
        renders = utils.batched_render(mesh, azims, elevs, dists, batch_size, device)
        iou_calcs = []
        rendered_image_fits = []
        for i in range(renders.shape[0]):
            iou = get_iou(renders[i].cpu().numpy(), mask)
            iou_calcs.append(iou)
            rendered_image_fits.append(rgba_obj_in_frame(renders[i].cpu().numpy()))
        iou_argsort = np.argsort(iou_calcs)[::-1]
        rendered_image_fits = np.array(rendered_image_fits)[iou_argsort]

        # choose distance with highest iou, whose rendered image will fit completely in the frame
        i = 0
        while not rendered_image_fits[i]:
            i+=1
            # if nothing fits, just return the highest IoU even though it doesn't fit
            if i >= len(rendered_image_fits):
                print("Couldn't find a pose which fits entirely")
                i=0
                break

        pred_dist = dists[iou_argsort[i]]
        iou = iou_calcs[iou_argsort[i]]
        render = renders[iou_argsort[i]]
        
    return pred_azim, pred_elev, pred_dist, render, iou


def crop_binary_mask(mask):
    try:
        # Get the height and width of bbox
        objs = ndimage.find_objects(mask)
        # upper left, lower right
        img_bbox = [objs[0][0].start, objs[0][1].start, objs[0][0].stop, objs[0][1].stop]
    except IndexError:
        # if no bounding box found, return None
        print("no bbox found")
        return None
    # crop
    cropped_mask = mask[img_bbox[0]:img_bbox[2],img_bbox[1]:img_bbox[3] ]
    return cropped_mask


# return IOU, given an rgba 4-channel image and a mask. They don't need to be the same size
# (mask will be resized)
def get_iou(img, mask):
    img_mask = img[:,:,3] > 0
    mask_resized = img_as_bool(resize(mask, img_mask.shape))
    overlap = img_mask * mask_resized # Logical AND
    union = img_mask + mask_resized # Logical OR
    IOU = overlap.sum()/float(union.sum())
    return IOU


# given an 4 channel image (w1 x h1 x rgba), assumed to be an object on a white background and a mask (w2 x h2 x 1)
# will crop the image and mask, resize them, and then compute the IoU
# (mask is the gt mask, img is rendered reconstruction)
def get_normalized_iou(img, mask, show_intermediate=False):
    img_mask = img[:,:,3] > 0
    img_mask_cropped = crop_binary_mask(img_mask)
    mask_cropped = crop_binary_mask(mask)

    # if there was an issue with cropping (ie, no bounding box was found, just return IoU zero)
    if (img_mask_cropped is None) or (mask_cropped is None):
        return 0

    # resize the image to the mask's dimensions
    img_mask_resized = img_as_bool(resize(img_mask_cropped, mask_cropped.shape))
    
    overlap = img_mask_resized*mask_cropped # Logical AND
    union = img_mask_resized + mask_cropped # Logical OR
    IOU = overlap.sum()/float(union.sum())
    
    if show_intermediate:
        #plt.imshow(img_mask_cropped)
        #plt.show()
        plt.imshow(mask_cropped)
        plt.show()
        plt.imshow(img_mask_resized)
        plt.show()
    return IOU
    

# assumes input is an rgba image (np array); returns a boolean, of if the image contained in the image
# is captured fully (there is an empty border around the image.)
# pix_border_req is the threshold num. of pixels on the border to be considered "captured fully"
def rgba_obj_in_frame(rgba_img, px_border_req=1):
    mask = (rgba_img[:,:,3] > 0)
    objs = ndimage.find_objects(mask)
    # TODO: check height, width
    img_height = mask.shape[0]
    img_width = mask.shape[1]
    if len(objs) == 0:
        return False
    # upper left, lower right
    upper_left = [objs[0][0].start, objs[0][1].start]
    lower_right = [objs[0][0].stop, objs[0][1].stop]
    in_frame = False
    if (upper_left[0] >= px_border_req and upper_left[1] >= px_border_req and 
        img_width - lower_right[0] >= px_border_req and img_height - lower_right[1] >= px_border_req):
        in_frame = True
    
    return in_frame