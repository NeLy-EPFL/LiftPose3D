import numpy as np

def preprocess_2d(train: dict, 
                  test: dict, 
                  roots: list, 
                  target_sets: list, 
                  in_dim: int,
                  mean=None,
                  std=None):
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
    
    # anchor points to body-coxa (to predict leg joints w.r.t. body-boxas)
    train, _ = anchor_to_root(train, roots, target_sets, in_dim)
    test, offset = anchor_to_root(test, roots, target_sets, in_dim)

    # Standardize each dimension independently
    if (mean is None) or (std is None):
        mean, std = normalization_stats(train)
    train = normalize(train, mean, std)
    test = normalize(test, mean, std)

    # select coordinates to be predicted and return them as 'targets'
    train, _ = remove_roots(train, target_sets, in_dim)
    test, targets = remove_roots(test, target_sets, in_dim)

    return train, test, mean, std, targets, offset


def preprocess_3d(train, 
                  test, 
                  roots, 
                  target_sets, 
                  out_dim,
                  mean=None,
                  std=None
):
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
            train:  Zero-mean and unit variance training data
            test: Zero-mean and unit variance test data
            mean: mean parameter for each dimension of train dadta
            std: std parameter for eaach dimension of test data
            targets_3d: TODO
            offset: the root position for corresponding target_sets for each joint
    """
    
    train = train.copy()
    test = test.copy()
    
    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    train, _ = anchor_to_root(train, roots, target_sets, out_dim)
    test, offset = anchor_to_root(test, roots, target_sets, out_dim)

    # Standardize each dimension independently
    if (mean is None) or (std is None):
        mean, std = normalization_stats(train)
    train = normalize(train, mean, std)
    test = normalize(test, mean, std)

    # select coordinates to be predicted and return them as 'targets_3d'
    train, _ = remove_roots(train, target_sets, out_dim)
    test, targets_3d = remove_roots(test, target_sets, out_dim)

    return train, test, mean, std, targets_3d, offset


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
    # TODO 
  
    Args
        data: dictionary of experiments each with array of size n_frames x n_dimensions
        targets: list of list of dimensions to be considered
        n_dim: number of spatial dimensions (e.g., 1,2 or 3)
        
    Returns
        data: dictionary of experiments with roots removed
        dim_to_use: list of dimensions in use for lifting
    """

    dim_to_use = np.squeeze(get_coords_in_dim(targets, n_dim))

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


def init_keypts(train_3d):
    """create a new dictionary with the same (k,v) pairs. v has dtype bool"""
    return {k: np.ones_like(v, dtype=bool) for (k, v) in train_3d.items()}


def flatten_dict(d):
    """reshapes each (N,T,C) value inside the dictionary into (N,T*C)"""
    for (k, v) in d.items():
        d[k] = v.reshape(v.shape[0], v.shape[1] * v.shape[2])
    return d


def get_visible_points(d, good_keypts):
    d = d.copy()
    for (k, v) in d.items():
        d_tmp = []
        for i in range(d[k].shape[0]):
            d_tmp.append(v[i,good_keypts[k][i,:,0],:])
        
        d[k] = np.stack(d_tmp,axis=0)
        
    return d
