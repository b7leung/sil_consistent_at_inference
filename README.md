# sil_consistent_at_inference

To get started, take a look at:
* notebooks/Deformation_Net.ipynb

The test dataset used can be downloaded here:

https://drive.google.com/file/d/1_30nA9JKxA1uNXUXoCv8qzwhoGovWhgV/view?usp=sharing


# Tools for preparing inputs to postprocessing
The postprocessing algorithm expects segmented rgba 224x224 .png images, and .obj meshes with relatively few (few thousand) vertices. Otherwise, if there are too many vertices, it will take a long time per example (though, the meshes don't have to be exactly the same # of vertices). There are some scripts in this repo to help with preparing the img/mesh inputs:
### Converting Meshes from .off to .obj
* get meshlab 2016.12 on windows
https://github.com/cnr-isti-vclab/meshlab/releases?after=e50c7fc
* use cvt_off_obj.py inside the data_prep_tools dir
### Decimating Meshes (.objs only, .off meshes doesn't work)
* get meshlab 2016.12 on windows
https://github.com/cnr-isti-vclab/meshlab/releases?after=e50c7fc
* install meshlabxml on windows python using pip
https://github.com/3DLIRIOUS/MeshLabXML
* run in command prompt, inside the data_prep_tools dir: `python3 simplify_folder.py windows_path_to_meshes_folder num_faces_desired`
### Preparing Images
* There is some code to help with preparing images (eg resizing, removing white backgrounds) in data_prep_tools/image_prep.ipynb
