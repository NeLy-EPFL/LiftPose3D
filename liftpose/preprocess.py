import numpy as np
import logging
import sys
import os

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="[%(filename)s:%(lineno)d]:%(levelname)s:%(message)s",
    datefmt="%H:%M:%S",
)


def preprocess_2d(
    train: dict,
    test: dict,
    roots: list,
    target_sets: list,
    in_dim: int,
    mean=None,
    std=None,
):
    """ Preprocess 2D data
        1. Center points in target_sets sets around roots
        2. Normalizes data into zero mean and unit variance
        3. Remove root joints
        
        Args:
            train: Dict[Tuple:np.array[float]]
                A dictionary where keys correspond to experiment names and 
                values are numpy arrays in the shape of [T J], where T 
                corresponds to time and J is number of joints times in_dim. 
            test: Dict[Tuple:np.array[float]]
                test data
            roots: List[int]
                Single depth list consisting of root joints. Corresponding
                target set will predicted with respect to the root joint. 
                Cannot be empty.
            target_sets: List[List[Int]
                Joints to be predicted with respect to roots.
                if roots = [0, 1] and target_sets = [[2,3], [4,5]], then the
                network will predict the relative location Joint 2 and 3 with respect to Joint 0.
                Likewise Joint location 4 and 5 will be predicted with respect to Joint 1.
                Cannot be empty.
            in_dim: number of dimensions for the 2d data 
                (should be 2 if predicting depth from 2d pose).

        Return:
            train:  Zero-mean and unit variance training data
            test: Zero-mean and unit variance test data
            mean: mean parameter for each dimension of train dadta
            std: std parameter for eaach dimension of test data
            targets_3d: TODO
            offset: the root position for corresponding target_sets for each joint
    """

    train_cp = train.copy()
    test_cp = test.copy()

    # anchor points to body-coxa (to predict leg joints w.r.t. body-boxas)
    train_cp, _ = anchor_to_root(train_cp, roots, target_sets, in_dim)
    test_cp, offset = anchor_to_root(test_cp, roots, target_sets, in_dim)

    # Standardize each dimension independently
    if (mean is None) or (std is None):
        mean, std = normalization_stats(train_cp)
    train_cp = normalize(train_cp, mean, std)
    test_cp = normalize(test_cp, mean, std)

    # select coordinates to be predicted and return them as 'targets'
    train_cp, _ = remove_roots(train_cp, target_sets, in_dim)
    test_cp, targets = remove_roots(test_cp, target_sets, in_dim)

    return train_cp, test_cp, mean, std, targets, offset


def preprocess_3d(train, test, roots, target_sets, out_dim, mean=None, std=None):
    """ Preprocess 3D data
        1. Center points in target_sets sets around roots
        2. Normalizes data into zero mean and unit variance
        3. Remove root joints
        
        Args:
            train: Zero-mean and unit variance training data
            test: Zero-mean and unit variance test data
            roots: Single depth list consisting of root joints. Corresponding
                target set will predicted with respect to the root joint. 
                Cannot be empty.
            target_sets: List[List[Int]
                Joints to be predicted with respect to roots.
                if roots = [0, 1] and target_sets = [[2,3], [4,5]], then the
                network will predict the relative location Joint 2 and 3 with respect to Joint 0.
                Likewise Joint location 4 and 5 will be predicted with respect to Joint 1.
                Cannot be empty.

        Return:
            train: Zero-mean and unit variance training data
            test: Zero-mean and unit variance test data
            mean: mean parameter for each dimension of train dadta
            std: std parameter for eaach dimension of test data
            targets_3d: TODO
            offset: the root position for corresponding target_sets for each joint
    """

    train_cp = train.copy()
    test_cp = test.copy()

    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    train_cp, _ = anchor_to_root(train_cp, roots, target_sets, out_dim)
    test_cp, offset = anchor_to_root(test_cp, roots, target_sets, out_dim)

    # Standardize each dimension independently
    if (mean is None) or (std is None):
        mean, std = normalization_stats(train_cp)
    train_cp = normalize(train_cp, mean, std)
    test_cp = normalize(test_cp, mean, std)

    # select coordinates to be predicted and return them as 'targets_3d'
    train_cp, _ = remove_roots(train_cp, target_sets, out_dim)
    test_cp, targets_3d = remove_roots(test_cp, target_sets, out_dim)

    return train_cp, test_cp, mean, std, targets_3d, offset


def normalization_stats(d, replace_zeros=True):
    """ Computes mean and stdev
    
    Args
        d: dictionary containing data of all experiments
    Returns
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
    """

    if type(d) is dict:
        d = np.concatenate([v for k, v in d.items()], 0)

    # replace zeros by nans, so we ignore them during the mean, std calculation
    d = d.astype("float")[d == 0] = np.nan if replace_zeros else d

    mean = np.nanmean(d, axis=0)
    std = np.nanstd(d, axis=0)

    return mean, std


def center_poses(d):
    """move center of gravity to origin"""

    for k in d.keys():
        d[k] -= np.mean(d[k], axis=1, keepdims=True)

    return d


def normalize(d, mean, std):
    """ Normalizes a dictionary of poses
  
    Args
        d: dictionary containing data of all experiments
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
      
    Returns
        d: dictionary containing normalized data
    """

    np.seterr(divide="ignore", invalid="ignore")

    for k in d.keys():
        d[k] -= mean
        d[k] /= std

    return d


def unNormalize(d_norm, mean, std):
    """ Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation
  
    Args
        d_norm: dictionary containing normalized data of all experiments
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
      
    Returns
        data: dictionary containing normalized data
    """

    d_norm *= std
    d_norm += mean

    return d_norm


def anchor_to_root(poses, roots, target_sets, dim):
    """ Center points in targset sets around roots
    
    Args
        poses: dictionary of experiments each with array of size n_frames x n_dimensions
        roots: list of dimensions to be pulled to the origin
        target_sets: list of lists of indexes that are computer relative to respective roots
        dim: spatial dimension of data (1, 2 or 3)
  
    Returns
        poses: dictionary of anchored poses
        offset: offset of each root from origin
    """
    assert len(target_sets) == len(roots), "We need the same # of roots as target sets!"
    assert all([p.ndim == 2 for p in list(poses.values())])

    offset = {}
    for k in poses.keys():

        offset[k] = np.zeros_like(poses[k])
        for i, root in enumerate(roots):
            for j in [root] + target_sets[i]:
                offset[k][:, dim * j : dim * (j + 1)] += poses[k][
                    :, dim * root : dim * (root + 1)
                ]

    for k in poses.keys():
        poses[k] -= offset[k]

    return poses, offset


def add_roots(d, dim_to_use, n_dim, base="zeros"):
    """ Add back the root dimensions
    
    Args
        d: array of size n_frames x (n_dim-n_roots)
        dim_to_use: list of indices of dimenions that are not roots 
        n_dim: integer number of dimensions including roots
    
    Returns
        orig_data: array of size n_frames x n_dim
    """

    T = d.shape[0]
    D = n_dim
    if base == "zeros":
        orig_data = np.zeros((T, D), dtype=np.float32)
    elif base == "ones":
        orig_data = np.ones((T, D), dtype=np.float32)
    else:
        raise NotImplementedError

    orig_data[:, dim_to_use] = d

    return orig_data


def remove_roots(d, targets, n_dim, vis=None):
    """
    # TODO 
  
    Args
        d: dictionary of experiments each with array of size n_frames x n_dimensions
        targets: list of list of dimensions to be considered
        n_dim: number of spatial dimensions (e.g., 1,2 or 3)
        
    Returns
        data: dictionary of experiments with roots removed
        dim_to_use: list of dimensions in use for lifting
    """

    dim_to_use = np.squeeze(get_coords_in_dim(targets, n_dim))

    for k in d.keys():
        if vis is not None:
            d[k] = d[k][:, vis]
        d[k] = d[k][:, dim_to_use]

    return d, dim_to_use


def get_coords_in_dim(targets, dim):
    """ Get keypoint indices in spatial dimension 'dim'
    
    Args
        targets: list of lists of keypoints to be converted
        dim: spatial dimension of data (1, 2 or 3)
        
    Returns
        dim_to_use: list of keypoint indices in dimension dim
    """

    if len(targets) > 1:
        dim_to_use = []
        for i in targets:
            dim_to_use += i
    else:
        dim_to_use = targets

    dim_to_use = np.array(dim_to_use)
    if dim == 2:
        dim_to_use = np.sort(np.hstack((dim_to_use * 2, dim_to_use * 2 + 1)))

    elif dim == 3:
        dim_to_use = np.sort(
            np.hstack((dim_to_use * 3, dim_to_use * 3 + 1, dim_to_use * 3 + 2))
        )
    return dim_to_use


def init_keypts(train_3d):
    """create a new dictionary with the same (k,v) pairs. v has dtype bool"""
    d = {k: np.ones_like(v, dtype=bool) for (k, v) in train_3d.items()}

    return d


def flatten_dict(d):
    """reshapes each (N,T,C) value inside the dictionary into (N,T*C)"""
    assert isinstance(d, dict)
    assert all([v.ndim == 3 for v in d.values()])
    for (k, v) in d.items():
        d[k] = v.reshape(v.shape[0], v.shape[1] * v.shape[2])

    return d


def concat_dict(d):
    """concatenates dictionary vertically in the first dimension"""

    d_concat = np.concatenate([v for k, v in d.items()], 0)

    return d_concat


def total_frames(d):
    """Count the number of frames for all experiments"""
    count = 0
    for (k, v) in d.items():
        count += v.shape[0]

    return count


def remove_dimensions(d, dims_to_remove):
    """Remove dimensions specified in dims_to_remove"""
    for (k, v) in d.items():
        d[k] = np.delete(d[k], dims_to_remove, axis=1)

    return d


def get_visible_points(d, good_keypts):
    """ Restricts a dictionary of poses only to th visible points.
    

    Parameters
    ----------
    d : dict of np arrays
        Dict of poses.
    good_keypts : dict of boolean np arrays
        Visible points for each timestep.

    Returns
    -------
    d : dict of np arrays
        Dict of poses with only visible keypoints.

    """
    d = d.copy()
    for (k, v) in d.items():
        d_tmp = []
        for i in range(d[k].shape[0]):
            d_tmp.append(v[i, good_keypts[k][i, :, 0], :])

        d[k] = np.stack(d_tmp, axis=0)

    return d


from liftpose.vision_3d import project_to_random_eangle, process_dict

import pickle
from numpy import linalg


def obtain_projected_stats(
    poses, eangle, axsorder, intr, roots, target_sets, out_dir, th=0.05
):

    error = np.inf
    count = 0
    error_log = []

    # run until convergence
    while error > th:
        # obtain randomly projected points
        # print(poses)
        # import sys
        # sys.exit()
        pts_2d = process_dict(
            project_to_random_eangle,
            poses,
            eangle,
            axsorder=axsorder,
            project=True,
            intr=intr,
        )
        pts_3d = process_dict(
            project_to_random_eangle, poses, eangle, axsorder=axsorder, project=False
        )

        pts_2d = flatten_dict(pts_2d)
        pts_3d = flatten_dict(pts_3d)

        pts_2d, _ = anchor_to_root(pts_2d, roots, target_sets, 2)
        pts_3d, _ = anchor_to_root(pts_3d, roots, target_sets, 3)

        pts_2d = np.concatenate([v for k, v in pts_2d.items()], 0)
        pts_3d = np.concatenate([v for k, v in pts_3d.items()], 0)

        # bootstrap mean, std
        if count == 0:
            train_samples_2d = pts_2d
            mean_old_2d = np.zeros(pts_2d.shape[1])
            std_old_2d = np.zeros(pts_2d.shape[1])
            train_samples_3d = pts_3d
            mean_old_3d = np.zeros(pts_3d.shape[1])
            std_old_3d = np.zeros(pts_3d.shape[1])
        else:
            train_samples_2d = np.vstack((train_samples_2d, pts_2d))
            train_samples_3d = np.vstack((train_samples_3d, pts_3d))

        mean_2d, std_2d = normalization_stats(train_samples_2d, replace_zeros=False)
        mean_3d, std_3d = normalization_stats(train_samples_3d, replace_zeros=False)

        error = (
            linalg.norm(mean_2d - mean_old_2d)
            + linalg.norm(std_2d - std_old_2d)
            + linalg.norm(mean_3d - mean_old_3d)
            + linalg.norm(std_3d - std_old_3d)
        )
        error_log.append(error)

        logger.info(f"Expected error for obtaining projection stats: {error}")
        mean_old_2d = mean_2d
        std_old_2d = std_2d
        mean_old_3d = mean_3d
        std_old_3d = std_3d
        count += 1

        if not os.path.exists(out_dir):
            logger.info(f"Creating directory {os.path.abspath(out_dir)}")
            os.makedirs(out_dir)

        # save
        pickle.dump(
            error_log,
            open(os.path.abspath(os.path.join(out_dir, "error_log.pkl")), "wb"),
        )

        # TODO remove the absolute path, this wouldn't work on other computers
        pickle.dump(
            [mean_2d, std_2d, mean_3d, std_3d],
            open(
                os.path.abspath(
                    os.path.join(
                        out_dir,
                        "/data/LiftPose3D/fly_tether/angle_inv_network/stats.pkl",
                    )
                ),
                "wb",
            ),
        )

    return mean_2d, std_2d, mean_3d, std_3d
