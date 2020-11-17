import subprocess
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='run simplify-folder on a 2-nested folder'
)
parser.add_argument('dir', type=str, help='Path folder with meshes')
parser.add_argument('num_faces', type=int, help='Target number of faces.')
args = parser.parse_args()

for root, dirs, files in os.walk(args.dir):
    for d in tqdm(dirs, desc="multi_folders"):
        folder_to_simplify = os.path.join(root, d)
        if "simplified" not in folder_to_simplify:
            subprocess.run(["python3", "simplify_folder.py", folder_to_simplify, str(args.num_faces)], check=True)

