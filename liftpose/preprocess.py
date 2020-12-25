import numpy as np
from liftpose.vision_3d import transform_frame


def preprocess_2d(
    train: np.array, test: np.array, roots: list, target_sets: list, in_dim: int
):
    """
    Preprocess 2D data
    """
    # anchor points to body-coxa (to predict leg joints w.r.t. body-boxas)
    train, _ = anchor_to_root(train, roots, target_sets, in_dim)
    test, offset = anchor_to_root(test, roots, target_sets, in_dim)

    # Standardize each dimension independently
    mean, std = normalization_stats(train)
    train = normalize(train, mean, std)
    test = normalize(test, mean, std)

    # select coordinates to be predicted and return them as 'targets'
    train, _ = remove_roots(train, target_sets, in_dim)
    test, targets = remove_roots(test, target_sets, in_dim)

    return train, test, mean, std, targets, offset


def preprocess_3d(
    train, test, roots, target_sets, out_dim, rcams_train, rcams_test,
):
    """
    Preprocess 3D data
    """
    # transform to camera coordinates
    if rcams_train is not None and rcams_test is not None:
        train = transform_frame(train, rcams_train)
        test = transform_frame(test, rcams_test)

    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    train, _ = anchor_to_root(train, roots, target_sets, out_dim)
    test, offset = anchor_to_root(test, roots, target_sets, out_dim)

    # Standardize each dimension independently
    mean, std = normalization_stats(train)
    train = normalize(train, mean, std)
    test = normalize(test, mean, std)

    # select coordinates to be predicted and return them as 'targets_3d'
    train, _ = remove_roots(train, target_sets, out_dim)
    test, targets_3d = remove_roots(test, target_sets, out_dim)

    return train, test, mean, std, targets_3d, rcams_test, offset


def normalization_stats(data):
    """
    Computes mean and stdev
    
    Args
        data: dictionary containing data of all experiments
    Returns
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
    """

    complete_data = np.concatenate([v for k, v in data.items()], 0)
    mean = np.nanmean(complete_data, axis=0)
    std = np.nanstd(complete_data, axis=0)

    return mean, std


def normalize(data, mean, std):
    """
    Normalizes a dictionary of poses
  
    Args
        data: dictionary containing data of all experiments
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
      
    Returns
        data: dictionary containing normalized data
    """

    np.seterr(divide="ignore", invalid="ignore")

    for key in data.keys():
        data[key] -= mean
        data[key] /= std

    return data


def unNormalize(data_norm, mean, std):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation
  
    Args
        data: dictionary containing normalized data of all experiments
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
      
    Returns
        data: dictionary containing normalized data
    """

    data_norm *= std
    data_norm += mean

    return data_norm


def anchor_to_root(poses, roots, target_sets, dim):
    """
    Center points in targset sets around roots
    
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


def add_roots(data, dim_to_use, n_dim):
    """
    Add back the root dimensions
    
    Args
        data: array of size n_frames x (n_dim-n_roots)
        dim_to_use: list of indices of dimenions that are not roots 
        n_dim: integer number of dimensions including roots
    
    Returns
        orig_data: array of size n_frames x n_dim
    """

    T = data.shape[0]
    D = n_dim
    orig_data = np.zeros((T, D), dtype=np.float32)
    orig_data[:, dim_to_use] = data

    return orig_data


def remove_roots(data, targets, n_dim, vis=None):
    """
    Normalizes a dictionary of poses
  
    Args
        data: dictionary of experiments each with array of size n_frames x n_dimensions
        targets: list of list of dimensions to be considered
        n_dim: number of spatial dimensions (e.g., 1,2 or 3)
        
    Returns
        data: dictionary of experiments with roots removed
        dim_to_use: list of dimensions in use for lifting
    """

    dim_to_use = get_coords_in_dim(targets, n_dim)

    for key in data.keys():
        if vis is not None:
            data[key] = data[key][:, vis]
        data[key] = data[key][:, dim_to_use]

    return data, dim_to_use


def get_coords_in_dim(targets, dim):
    """
    Get keypoint indices in spatial dimension 'dim'
    
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
