# sil_consistent_at_inference

To get started, take a look at:
* notebooks/Deformation_Net.ipynb

The test dataset used can be downloaded here:

https://drive.google.com/file/d/1_30nA9JKxA1uNXUXoCv8qzwhoGovWhgV/view?usp=sharing


# Tools for preparing inputs
### Converting Meshes from .off to .obj
* get meshlab 2016.12 on windows
https://github.com/cnr-isti-vclab/meshlab/releases?after=e50c7fc
* run cvt_off_obj.py inside the prep_tools dir
### Decimating Meshes
* get meshlab 2016.12 on windows
https://github.com/cnr-isti-vclab/meshlab/releases?after=e50c7fc
* install meshlabxml on windows python using pip
https://github.com/3DLIRIOUS/MeshLabXML
* run in command prompt, inside the prep_tools dir: `python3 simplify_folder.py windows_path_to_meshes_folder num_faces_desired`
### Preparing Images
* There is some code to help with preparing images in prep_tools/image_prep.ipynb
