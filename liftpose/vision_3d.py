import numpy as np
from typing import List, Dict
from scipy.spatial.transform import Rotation as Rot
import torch


def reprojection_error(
    poses_3d: np.ndarray,
    poses_2d: np.ndarray,
    R: np.ndarray,
    tvec: np.ndarray,
    intr: np.ndarray,
) -> np.ndarray:
    assert poses_3d.ndim == 3
    assert poses_2d.ndim == 3
    assert R.ndim == 2
    assert tvec.ndim == 1 or tvec.ndim == 2
    assert intr.ndim == 2

    poses_3d = world_to_camera(poses_3d, R, tvec)
    proj_2d = project_to_camera(poses_3d, intr)
    assert proj_2d.dim == 3
    return np.linalg.norm(poses_2d - proj_2d, axis=2)


def adjust_tree(pose3d, root, child, offset) -> None:
    """move each subtree by offset amount. the current node is the root. 
    find each node looking at the child[root]"""
    pose3d[:, root] += offset
    for c in child[root]:
        adjust_tree(pose3d, c, child, offset)


def normalize_bone_length(
    pose3d: np.ndarray,
    root: int,
    child: List[int],
    bone_length: Dict[tuple, float],
    thr: float = 0,
) -> np.ndarray:
    """pose3d has shape [N, J, 3]. retargets the bones of each poses without changing the angles.
       bone_length is a dictionary with joint ids """
    assert pose3d.ndim == 3, f"{pose3d.ndim}"
    assert isinstance(bone_length, dict)
    assert isinstance(child, list)

    for c in child[root]:
        vec = pose3d[:, c] - pose3d[:, root]  # find the current bone-length
        curr_length = np.linalg.norm(vec, axis=-1, keepdims=True)

        k = (c, root) if (c, root) in bone_length else (root, c)
        offset = (vec / curr_length) * (bone_length[k] - curr_length)
        # if the curr_length is too small, then normalization will just amplify the noise
        # so just ignore it
        offset[np.squeeze(curr_length < thr)] = 0
        adjust_tree(pose3d, c, child, offset)  # move subtree by offset amount
        normalize_bone_length(
            pose3d, c, child, bone_length
        )  # continue fixing the subtree

    return pose3d


def calculate_bone_length(poses_3d: np.ndarray, edges: List[List[int]]) -> np.ndarray:
    """expects poses_3d in shape [N J 3]. returns a numpy array of shape [N len(edges)]"""
    assert poses_3d.ndim == 3, f"{poses_3d.ndim}"
    assert poses_3d.shape[2] == 3
    edges = np.array(edges)
    return np.linalg.norm(poses_3d[:, edges[:, 0]] - poses_3d[:, edges[:, 1]], axis=2)


def world_to_camera_dict(poses_world: dict, cam_par: dict) -> dict:
    """
    Affine transform 3D coordinates to camera frame

    Args
        poses_world: dictionary with 3d poses in world coordinates
        cams: dictionary with camera parameters
        cam_id: camera_id to consider
    Returns
        poses_cam: dictionary with 3d poses (2d poses if projection is True) in camera coordinates
        vis: boolean array with coordinates visible from the camera
    """

    poses_cam = {}
    for k in poses_world.keys():
        rcams = cam_par[k]
        poses_cam[k] = world_to_camera(poses_world[k], rcams["R"], rcams["tvec"])

    return poses_cam


def world_to_camera(
    poses_world: np.ndarray, R: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """
    Rotate/translate 3d poses from world to camera viewpoint

    Args
        poses_world: array of poses in world coordinates of size n_frames x joints x n_dimensions
        cam_par: dictionary of camera parameters

    Returns
        poses_cam: poses in camera-centred coordinates
    """
    s = poses_world.shape

    poses_world = np.reshape(poses_world, [s[0] * s[1], 3])

    assert poses_world.shape[1] == 3

    if len(R) == poses_world.shape[0] // s[1]:
        poses_cam = np.zeros_like(poses_world)
        for i in range(poses_world.shape[0]):
            poses_cam[i, :] = np.matmul(R[i // s[1]], poses_world[i, :])
    else:
        poses_cam = np.matmul(R, poses_world.T).T
    
    if tvec is not None:
        poses_cam += tvec
    poses_cam = np.reshape(poses_cam, s)

    return poses_cam


def camera_to_world(
    poses_cam: np.ndarray, R: np.ndarray, tvec: np.ndarray
) -> np.ndarray:
    """
    Rotate/translate 3d poses from camera to world

    Args
        poses_cam: poses in camera coordinates, [N, J, 3]
        cam_par: dictionary with camera parameters

    Returns
        poses_world: poses in world coordinates
    """
    s = poses_cam.shape

    poses_world = np.reshape(poses_cam, [-1, 3]).copy()
    poses_world -= tvec
    poses_world = np.matmul(np.linalg.inv(R), poses_world.T).T
    poses_world = np.reshape(poses_world, s)

    return poses_world


def project_to_camera(poses: np.ndarray, intr: np.ndarray=None):
    """
    Project poses to camera frame

    Args
        poses: poses in camera coordinates, [N, J, 3]
        intr: intrinsic camera matrix, [3, 3]

    Returns
        poses_proj: 2D poses projected to camera plane
    """
    s = poses.shape

    poses = np.reshape(poses, [s[0] * s[1], 3])

    if intr is not None:
        poses = np.matmul(intr, poses.T).T
        poses = poses / poses[:, [2]]
    poses = poses[:, :2]
    poses = poses.reshape([s[0], s[1], 2])

    return poses


def process_dict(function, d: dict, n_out: int, *args, **kwargs):
    """
    Apply a function to each array in a dictionary.

    Parameters
    ----------
    function : Callable
        Function to apply to all arrays in d.
    d : dict of 3-dim numpy arrays
        Arrays to operate on.
    n_out : int
        Number of outputs of function
    *args : 
        Arguments of function.
    **kwargs :
        Keyword arguments of function

    Returns
    -------
    d_new
        Output of function.

    """

    if n_out > 1:
        d_new = [{} for i in range(n_out)]
    else:
        d_new = {}
        
    for key in d.keys():

        d_tmp = function(d[key], *args, **kwargs)
        
        for i in range(n_out):
            d_new[i][key] = d_tmp[i]

    return d_new


def project_to_random_eangle(
    poses_world, eangle, axsorder="xyz", project=False, tvec=None, intr=None
):
    """
    Project to a random Euler angle within specified intervals.

    Parameters
    ----------
    poses_world : t x n x 3 numpy array
        Poses in world coordinates.
    eangle_range : dict whose values are list of 3 pairs
        Lower and upper limits of Euler angles.
    axsorder : 'xyz' or a permutation
        Order of Euler angles.
    project : bool, optional
        Do projection. The default is False.
    intr : 3x3 matrix, optional
        Intrinsic camera matrix for the projections. The default is None.

    Returns
    -------
    Pcam : t x n x 2 or t x n x 3 numpy array
        Projected points.

    """
    assert poses_world.ndim == 3

    # generate Euler angles
    n = poses_world.shape[0]
    alpha = uniform(low=eangle[0][0], high=eangle[0][1], size=n)
    beta = uniform(low=eangle[1][0], high=eangle[1][1], size=n)
    gamma = uniform(low=eangle[2][0], high=eangle[2][1], size=n)
    eangle = [[alpha[i], beta[i], gamma[i]] for i in range(n)]

    Pcam = project_to_eangle(poses_world, eangle, axsorder, project=project, tvec=tvec, intr=intr)

    return Pcam, eangle


def uniform(low=0,high=1,size=1):
    '''Draw n uniform random numbers in [a,b]'''

    u = (high-low) * torch.rand(size) + low
    
    return u.numpy()


def project_to_eangle(poses_world, eangle, axsorder="xyz", project=False, tvec=None, intr=None):
    """
    Project to specified Euler angle

    Parameters
    ----------
    poses_world : t x n x 3 numpy array
        Poses in world coordinates.
    eangle : triple
        Euler angle.
    axsorder : 'xyz' or a permutation
        Order of Euler angles.
    project : bool, optional
        Do projection. The default is False.
    intr : 3x3 matrix, optional
        Intrinsic camera matrix for the projections. The default is None.

    Returns
    -------
    Pcam : t x n x 2 or t x n x 3 numpy array
        Projected points.

    """

    # convert to rotation matrices
    R = Rot.from_euler(axsorder, eangle, degrees=True).as_matrix()

    # obtain 3d pose in camera coordinates
    Pcam = world_to_camera(poses_world, R, tvec)

    # project to camera axis
    if project:
        Pcam = project_to_camera(Pcam, intr)

    return Pcam


def XY_coord_dict(poses: dict):
    """
    Project 3d poses to XY plane

    Args
        poses: poses

    Returns
        poses_xy: poses projected to xy plane
    """
    poses_xy = {}

    for key in poses.keys():
        poses_xy[key] = poses[key][:, :, :2]

    return poses_xy


def Z_coord_dict(poses: dict):
    """
    Project 3d poses to XY plane

    Args
        poses: poses

    Returns
        poses_xy: poses projected to xy plane
    """
    poses_xy = {}

    for key in poses.keys():
        poses_xy[key] = poses[key][:, :, [-1]]

    return poses_xy


def intrinsic_matrix(fx, fy, cx, cy):
    """
    Generate camera intrinsic matrix.

    Parameters
    ----------
    fx : float
        Focal length x.
    fy : float
        Focal length y.
    cx : float
        Camera center x.
    cy : float
        Camera center y.

    Returns
    -------
    intr : 3x3 numpy array
        Camera intrinsic matrix.

    """

    intr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1],], dtype=float,)

    return intr


def procrustes(X, Y, scaling=True, reflection="best"):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n, m = X.shape
    ny, my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.0).sum()
    ssY = (Y0 ** 2.0).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m - my)), 0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != "best":

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:, -1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA ** 2

        # transformed coords
        Z = normX * traceTA * np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my, :]
    c = muX - b * np.dot(muY, T)

    # transformation values
    tform = {"rotation": T, "scale": b, "translation": c}

    return d, Z, tform


def find_neighbours(k, pts, target_pts, nn):
    """
    Procrustes align a source pose to all poses in pts and find nearest neighbours

    Parameters
    ----------
    k : int
        Index of pose in target_pts to search nearest neighbours for.
    pts3d : 3-dim numpy array
        P.
    target_pts : 3-dim numpy array
        Poses in the .
    nn : int
        Number of nearest neighbours to return.

    Returns
    -------
    nn_ind: list
        lift of nearest neighbours in ascending order of distances.

    """
    target_pose = target_pts[k, :, :]
    disparity = np.zeros(pts.shape[0])
    for i in range(pts.shape[0]):
        disparity[i], _, _ = procrustes(
            target_pose, pts[i, :, :], scaling=True, reflection="best"
        )

    # find nn
    nn_ind = list(np.argsort(disparity)[:nn])

    return nn_ind


def best_linear_map(source_poses, target_poses, nns, nn):
    """
    Find best linear transformation from poses in target domain to their nearest 
    neighbour poses in source domain
    
    AX = B looking for A
    X^TA^T = B^T is a least squares problem

    Parameters
    ----------
    target_poses : 3-dim numpy array
        Target poses to map to.
    source_poses : 3-dim numpy array
        Source poses to map from.
    nns : list of lists
        Nearest neighbours of each target_pose in the source domain.
    nn : TYPE
        Number of nearest neighbours to use.

    Returns
    -------
    A_est : TYPE
        DESCRIPTION.

    """
    B = []  # target poses
    X = []  # poses to be mapped
    # M = [] #missing points
    for i, n in enumerate(nns):
        B_tmp = source_poses[n[:nn], :, :]
        B_tmp = B_tmp.reshape(B_tmp.shape[0], B_tmp.shape[1] * B_tmp.shape[2]).T
        X_tmp = target_poses[i, :, :]
        X_tmp = X_tmp.reshape(X_tmp.shape[0] * X_tmp.shape[1], 1)
        X_tmp = np.tile(X_tmp, (1, nn))
        # M_tmp = visible_points[[i],:].T
        # M_tmp = np.tile(M_tmp, (1,len(n)))
        B.append(B_tmp)
        X.append(X_tmp)
        # M.append(M_tmp)

    X = np.hstack(X)
    B = np.hstack(B)
    # M = np.hstack(M)
    # M = np.tile(M,(3,1))

    # A_est = censored_lstsq(X.T, B.T, np.ones_like(M).T)
    A_est = np.linalg.pinv(X.T).dot(B.T).T

    return A_est


def apply_linear_map(A, X):
    """
    Apply linear transformation to poses

    Parameters
    ----------
    A : dim*n x dim*n numpy array
        Linear map in dim dimensions between n joints.
    X : 2-dim numpy array of pose or 3-dim numpy array
        Input pose or set of poses.

    Returns
    -------
    B : dim*n x dim*n numpy array
        Mapped poses.

    """

    if len(X.shape) == 2:
        X = X[None, :]

    n_frames, n_joints, n_dim = X.shape
    assert A.shape[0] == n_joints * n_dim

    X_test = X.copy()
    X_test = X_test.reshape(n_frames, n_joints * n_dim)
    B = A.dot(X_test.T).T
    B = B.reshape(n_frames, n_joints, n_dim)

    return B
