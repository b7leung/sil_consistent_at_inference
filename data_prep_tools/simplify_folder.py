import os
import argparse
import sys
import meshlabxml as mlx
import platform
from tqdm import tqdm

# based off https://github.com/HusseinBakri/3DMeshBulkSimplification
def perform_simplify_mesh(original_mesh_path, simplified_mesh_path, num_faces):

    #Check the input mesh number of faces (so that we do not decimate to a higher number of faces than original mesh)
    MetricsMeshDictionary = {}
    MetricsMeshDictionary = mlx.files.measure_topology(original_mesh_path)
    if(MetricsMeshDictionary['face_num'] <= num_faces):
        #exit the script and print a message about it
        print("\n SORRY your decimated mesh can not have higher number of faces that the input mesh.....")
        sys.exit()

    # simplify
    simplified_meshScript = mlx.FilterScript(file_in=original_mesh_path, file_out=simplified_mesh_path,
                                         ml_version='2016.12')  # Create FilterScript object
    mlx.remesh.simplify(simplified_meshScript, texture=False, faces=num_faces,
                        target_perc=0.0, quality_thr=1.0, preserve_boundary=True,
                        boundary_weight=1.0, preserve_normal=True,
                        optimal_placement=True, planar_quadric=True,
                        selected=False, extra_tex_coord_weight=1.0)
    simplified_meshScript.run_script()


# https://stackoverflow.com/questions/11968998/remove-lines-that-contain-certain-string
def remove_lines_with_substrings(file_path, substrings):
    with open(file_path) as oldfile, open(file_path+"_TEMP_", 'w') as newfile:
        for line in oldfile:
            if not any(bad_word in line for bad_word in substrings):
                newfile.write(line)
    os.remove(file_path)
    os.rename(file_path+"_TEMP_", file_path)


parser = argparse.ArgumentParser(
    description='Decimate a folder of meshes, and discard texture/materials.'
)
parser.add_argument('meshes_dir', type=str, help='Path folder with meshes')
parser.add_argument('num_faces', type=int, help='Target number of faces.')
parser.add_argument('--keep_texture', action='store_true', help='Keep the empty mtl files and their references in the obj files.')
parser.add_argument('--overwrite', action='store_true', help='Even if the simplified mesh already exists, still compute it again and overwrite it. \
    off by default so that user can continue progress in the case of an error, without starting over.')
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

models = sorted(os.listdir(args.meshes_dir))
output_folder = args.meshes_dir + '_simplified'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for model_name in tqdm(models):
    simplified_path = os.path.join(output_folder, model_name)
    original_path = os.path.join(args.meshes_dir, model_name)
    if args.overwrite or not os.path.exists(simplified_path):
        perform_simplify_mesh(original_path, simplified_path, args.num_faces)
        if not args.keep_texture:
            remove_lines_with_substrings(simplified_path, ['mtl'])
            os.remove(simplified_path+'.mtl')

