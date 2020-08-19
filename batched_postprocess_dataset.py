import os
import pprint
import argparse
import pickle

from postprocess_dataset import postprocess_data

# https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
def split(a, n):
    k, m = divmod(len(a), n)
    return list((a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)))

# a wrapper for postprocess_dataset for postprocessing a large dataset of reconstructions across multiple GPUs
# requires that poses are already precomputed
# python batched_postprocess_dataset.py  data/test_dataset data/test_dataset configs/default.yaml 1 3
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Postprocess a folder of meshes and corresponding segmented images.'
    )
    parser.add_argument('input_dir_img', type=str, help='Path to folder with images for meshes.')
    parser.add_argument('input_dir_mesh', type=str, help='Path to folder with meshes to postprocess.')
    parser.add_argument('cfg_path', type=str, help='Path to yaml configuration file.')
    parser.add_argument('batch_i', type=int, help='which batch this is (1-indexed)')
    parser.add_argument('num_batches', type=int, help='number of batches to split dataset into')
    parser.add_argument('--gpu', type=int, default=0, help='Gpu number to use.')
    args = parser.parse_args()

    if args.batch_i > args.num_batches or args.batch_i <= 0:
        raise ValueError("batch_i cannot be greater than num_batches nor less than 1")

    pred_poses_path = os.path.join(args.input_dir_mesh, "pred_poses.p")

    if not os.path.exists(pred_poses_path):
        raise Error("poses need to be precomputed")

    cached_pred_poses = pickle.load(open(pred_poses_path, "rb"))
    instance_names = sorted(list(cached_pred_poses.keys()))
    curr_batch_meshes = split(instance_names, args.num_batches)[args.batch_i-1]

    #postprocess_data(args.input_dir_img, args.input_dir_mesh, args.cfg_path, args.gpu,
    #                 meshes_group_name="batch_{}_of_{}".format(args.batch_i, args.num_batches), meshes_to_render = curr_batch_meshes)

