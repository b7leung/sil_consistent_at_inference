import glob
import subprocess
import os
import argparse
import platform
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Convert a folder of .off to .obj meshes.'
)
parser.add_argument('meshes_dir', type=str, help='Path folder with meshes')
parser.add_argument('--remove_old', action='store_true', help='Keep the original off files.')
args = parser.parse_args()

# add meshlabserver to path
if(platform.system()=='Windows'):
    meshlabserver_path = 'C:\\Program Files\\VCG\\MeshLab'
elif(platform.system()=='Linux'):
    meshlabserver_path = '/usr/bin/meshlabserver'
elif(platform.system()=='Darwin'):
    meshlabserver_path = '/Applications/meshlab.app/Contents/MacOS/meshlabserver'
else:
    print('\n Unknown OS please set the PATH manually ...')
    raise OSError()
os.environ['PATH'] = meshlabserver_path + os.pathsep + os.environ['PATH']

f = glob.glob(os.path.join(args.meshes_dir, "*.off"))

for off_file in tqdm(f):
    print(off_file)
    #"meshlabserver -i 01_mesh.off -o 01_mesh.obj"
    subprocess.run(["meshlabserver", "-i", off_file, "-o", off_file.replace("off","obj")], check=True)

    if args.remove_old:
        os.remove(off_file)