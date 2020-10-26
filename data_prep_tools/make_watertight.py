from pathlib import Path
import pprint
import subprocess
import os
import argparse
from tqdm import tqdm
import io

#import trimesh



# python make_watertight.py /home/svcl-oowl/brandon/research/sil_consistent_at_inference/data/misc/example_shapenet
# python make_watertight.py /home/svcl-oowl/dataset/ShapeNetCore.v1/03211117
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='goes through a folder of .obj meshes and creates watertight versions of it')
    parser.add_argument('input_mesh_dir', type=str, help='Path to input mesh dir to convert o watertight')
    parser.add_argument('--manifold_plus_path', type=str, default="ManifoldPlus", help='Path to manifold plus root folder')
    parser.add_argument('--recompute', action="store_true")
    args = parser.parse_args()

    paths = [str(path) for path in list(Path(args.input_mesh_dir).rglob('*.obj'))]

    class TqdmPrintEvery(io.StringIO):
        # Output stream for TQDM which will output to stdout. Used for nautilus jobs.
        def __init__(self):
            super(TqdmPrintEvery, self).__init__()
            self.buf = None
        def write(self,buf):
            self.buf = buf
        def flush(self):
            print(self.buf)
    tqdm_out = TqdmPrintEvery()

    for input_mesh_path in tqdm(paths, file=tqdm_out):
        output_mesh_path = input_mesh_path.replace(".obj","_watertight.obj")
        if (args.recompute or not os.path.exists(output_mesh_path)) and "_watertight.obj" not in input_mesh_path:
            cmd = " ".join([os.path.join(args.manifold_plus_path, "build", "manifold"), "--input", input_mesh_path, "--output", output_mesh_path])
            os.system(cmd)
    
