import argparse
import os

from tqdm import tqdm

from utils import utils
from deformation.deformation_net import DeformationNetwork
from mesh_refiner import MeshRefiner
import deformation.losses as def_losses

# given the path to a configuration file, and the path to a folder with meshes and segmented images, will postprocess all meshes in that folder.
# the folder needs to have .obj meshes, and for each mesh, a corresponding .png image (with segmented, transparent bg) of size 244 x 224, with the same filename.

# Upon completion, will save a pickle file called
# postprocess.p which is a dict with dataframes containing training information and also intermediate data for debugging

# Arguments
parser = argparse.ArgumentParser(
    description='Postprocess a folder of meshes and corresponding segmented images.'
)
parser.add_argument('input_dir_img', type=str, help='Path to folder with images for meshes.')
parser.add_argument('input_dir_mesh', type=str, help='Path to folder with meshes to postprocess.')
parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
parser.add_argument('--gpu', type=str, default=0, help='Gpu number to use.')


args = parser.parse_args()