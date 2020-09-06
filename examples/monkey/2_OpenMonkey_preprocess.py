import os
import numpy as np
import glob
import torch
import src.utils as utils
import pickle

# =============================================================================
# To be specified
# =============================================================================

TRAIN_SUBJECTS = ["7", "9a", "10", "11"]  # "9", "9a", "9b", "10", "11"] #["7"]
#  # , "9", "9a", "9b"]# "10", "11"]
TEST_SUBJECTS = ["9"]

data_dir = "/data/LiftFly3D/openmonkey/"
out_dir = "/data/LiftFly3D/openmonkey/output"
actions = [""]  #'MDN_CsCh', 'aDN_CsCh']
camera_matrices = "/data/LiftFly3D/openmonkey/cameras.pkl"

target_sets = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
]
ref_points = [2]

# =============================================================================


def main():
    rcams = pickle.load(open(camera_matrices, "rb"))

    # 3D ground truth
    train_set, test_set, data_mean, data_std = read_3d_data(
        actions, data_dir, target_sets, ref_points, rcams
    )

    torch.save(train_set, out_dir + "/train_3d.pth.tar")
    torch.save(test_set, out_dir + "/test_3d.pth.tar")
    torch.save(
        {"mean": data_mean, "std": data_std}, out_dir + "/stat_3d.pth.tar",
    )

    # HG prediction (i.e. deeplabcut or similar)
    train_set, test_set, data_mean, data_std = read_2d_predictions(
        actions, data_dir, rcams, target_sets, ref_points
    )

    torch.save(train_set, out_dir + "/train_2d.pth.tar")
    torch.save(test_set, out_dir + "/test_2d.pth.tar")
    torch.save(
        {"mean": data_mean, "std": data_std}, out_dir + "/stat_2d.pth.tar",
    )


def read_3d_data(actions, data_dir, target_sets, ref_points, rcams=None):
    """
    Pipeline for processing 3D ground-truth data
    """

    dim = 3
    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, rcams)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, rcams)

    # filter data
    # train_set = utils.filter_data(train_set)
    # test_set = utils.filter_data(test_set, window=5, order=2)

    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    train_set, _ = utils.anchor(train_set, ref_points, target_sets, dim)
    test_set, _ = utils.anchor(test_set, ref_points, target_sets, dim)

    # Compute mean, std
    data_mean, data_std = utils.normalization_stats(train_set)

    # Standardize each dimension independently
    train_set = utils.normalize_data(train_set, data_mean, data_std)
    test_set = utils.normalize_data(test_set, data_mean, data_std)

    # transform to camera coordinates
    # if rcams is not None:
    #    train_set, _ = transform_frame(train_set, rcams, cam_id)
    #    test_set, vis = transform_frame(test_set, rcams, cam_id)

    # select coordinates to be predicted and return them as 'targets_3d'
    # train_set, _ = utils.collapse(train_set, target_sets, dim)
    # test_set, targets_3d = utils.collapse(test_set, target_sets, dim)

    return train_set, test_set, data_mean, data_std


def read_2d_predictions(actions, data_dir, rcams, target_sets, ref_points):
    """
    Pipeline for processing 2D data (stacked hourglass predictions)
    """

    dim = 2
    # Load 2d data
    train_set = load_stacked_hourglass(data_dir, TRAIN_SUBJECTS, actions)
    test_set = load_stacked_hourglass(data_dir, TEST_SUBJECTS, actions)

    # filter data
    #  train_set = utils.filter_data(train_set)
    #    test_set = utils.filter_data(test_set, window=5, order=2)

    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    train_set, _ = utils.anchor(train_set, ref_points, target_sets, dim)
    test_set, _ = utils.anchor(test_set, ref_points, target_sets, dim)

    # Compute mean, std
    data_mean, data_std = utils.normalization_stats(train_set)

    # Standardize each dimension independently
    train_set = utils.normalize_data(train_set, data_mean, data_std)
    test_set = utils.normalize_data(test_set, data_mean, data_std)
    
    
    for k in list(train_set.keys()):
        train_set[k] = train_set[k] / np.linalg.norm(
            train_set[k], axis=1, keepdims=True
        )

    for k in list(test_set.keys()):
        test_set[k] = test_set[k] / np.linalg.norm(test_set[k], axis=1, keepdims=True)
    

    # select coordinates to be predicted and return them as 'targets_2d'
    # train_set, _ = utils.collapse(train_set, target_sets, dim)
    # test_set, targets_2d = utils.collapse(test_set, target_sets, dim)

    return train_set, test_set, data_mean, data_std


def load_data(path, flies, actions, cams):
    """
    Load 3d ground truth, put it in an easy-to-access dictionary

    Args:
        path: String. Path where to load the data from
        flies: List of integers. Flies whose data will be loaded
        actions: List of strings. The actions to load
    Returns:
        data: Dictionary with keys (fly, action, filename)
    """

    path = os.path.join(path, "pose_result_linear*")
    fnames = glob.glob(path)
    # print(fnames)
    data = {}
    for fly in flies:
        for action in actions:

            fnames_new = list()
            for n in fnames:
                if "Fly" + fly + "_" in n and "pose_result_linear" in n:
                    fnames_new.append(n)
            assert len(fnames_new)

            for fname in fnames_new:

                seqname = os.path.basename(fname)

                poses = pickle.load(open(fname, "rb"))
                poses3d = np.array(poses["points3d"])
                batch_id = np.array(poses["batch_id"])
                assert poses3d.shape[0] == batch_id.shape[0]

                for idx in range(poses3d.shape[0]):
                    cid = poses["cam_id"][idx]
                    R, T, intr, distort, vis_pts = cams[batch_id[idx]][cid]
                    poses3d[idx] = utils.world_to_camera(poses3d[idx].copy(), R, T)

                poses3d = np.reshape(
                    poses3d, (poses3d.shape[0], poses3d.shape[1] * poses3d.shape[2])
                )

                data[
                    (fly, action, seqname[:-4])
                ] = poses3d  # [:-4] is to get rid of .pkl extension

    return data


def load_stacked_hourglass(path, flies, actions):
    """
    Load 2d data, put it in an easy-to-acess dictionary.
    
    Args
        path: string. Directory where to load the data from,
        flies: list of integers. Subjects whose data will be loaded.
        actions: list of strings. The actions to load.
    Returns
        data: dictionary with keys k=(fly, action, filename)
    """

    path = os.path.join(path, "pose_result*")
    fnames = glob.glob(path)

    data = {}
    for fly in flies:
        for action in actions:

            fnames_new = list()
            for n in fnames:
                if "Fly" + fly + "_" in n and "pose_result_linear" in n:
                    fnames_new.append(n)

            for fname in fnames_new:

                seqname = os.path.basename(fname)
                poses = pickle.load(open(fname, "rb"))
                poses = np.array(poses["points2d"])

                poses_cam = np.reshape(
                    poses, (poses.shape[0], poses.shape[1] * poses.shape[2])
                )

                data[(fly, action, seqname[:-4])] = poses_cam

    return data


'''
def transform_frame(poses, cams, cam_id, project=False):
    """
    Affine transform 3D cooridinates to camera frame

    Args
        poses: dictionary with 3d poses
        cams: dictionary with camera parameters
        cam_ids: camera_ids to consider
    Returns
        Ptransf: dictionary with 3d poses or 2d poses if projection is True
        vis: boolean array with coordinates visible from the camera
    """
    Ptransf = {}
    vis = {}
    for fly, a, seqname in sorted(poses.keys()):

        Pworld = poses[(fly, a, seqname)]

        R, T, intr, distort, vis_pts = cams[cam_id]
        Pcam = utils.world_to_camera(Pworld, R, T)

        if project:
            Pcam = utils.project_to_camera(Pcam, intr)

        Ptransf[(fly, a, seqname + ".cam_" + str(cam_id))] = Pcam

        vis_pts = vis_pts[dims_to_consider]
        vis = np.array(vis_pts, dtype=bool)

    return Ptransf, vis
'''

if __name__ == "__main__":
    main()
