import numpy as np


def reprojection_error(poses_3d, poses_2d, R, tvec, intr):
    assert poses_3d.ndim == 3
    assert poses_2d.ndim == 3
    assert R.ndim == 2
    assert tvec.ndim == 1 or tvec.ndim == 2
    assert intr.ndim == 2

    poses_3d = world_to_camera(poses_3d, R, tvec)
    proj_2d = project_to_camera(poses_3d, intr)
    return np.linalg.norm(poses_2d - proj_2d, axis=2)


def normalize_bone_length(pose3d, edges, bone_length, parents, leaves):
    edges = [tuple(e) for e in edges]
    pose3d_normalized = pose3d.copy()
    for leaf in leaves:
        curr = leaf
        print(curr)
        print(",", len(parents))
        parent = parents[curr]
        history = list()
        while parent != -1:
            try:
                idx = edges.index((curr, parent))
            except:
                idx = edges.index((parent, curr))
            vec = pose3d_normalized[curr] - pose3d_normalized[parent]
            curr_length = np.linalg.norm(vec)
            offset = (vec / curr_length) * (bone_length[idx] - curr_length)

            history.append((curr, parent))
            for c, p in history:
                pose3d_normalized[c] += offset

            curr = parent
            parent = parents[curr]

    return pose3d_normalized

def calc_bone_length(poses_3d: np.ndarray, edges: list):
    assert poses_3d.ndim == 3
    bone_length = np.zeros((len(edges)))
    bone_length_std = np.zeros((len(edges)))
    for idx, edge in enumerate(edges):
        bone_length[idx] = np.nanmean(
            np.linalg.norm(poses_3d[:, edge[0]] - poses_3d[:, edge[1]], axis=1)
        )
        bone_length_std[idx] = np.nanstd(
            np.linalg.norm(poses_3d[:, edge[0]] - poses_3d[:, edge[1]], axis=1)
        )

    return bone_length, bone_length_std


def world_to_camera_dict(poses_world: dict, cam_par: dict):
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
        poses_cam[k] = world_to_camera(poses_world[k], rcams['R'], rcams['tvec'])

    # sort dictionary
    #poses_cam = dict(sorted(poses_cam.items()))

    return poses_cam


def world_to_camera(poses_world: np.ndarray, R: np.ndarray, tvec: np.ndarray):
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

    poses_cam = np.matmul(R, poses_world.T).T + tvec
    poses_cam = np.reshape(poses_cam, s)

    return poses_cam


def camera_to_world(poses_cam, R, tvec):
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


def project_to_camera(poses: np.ndarray, intr: np.ndarray):
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
    poses_proj = np.matmul(intr, poses.T).T
    poses_proj = poses_proj / poses_proj[:, [2]]
    poses_proj = poses_proj[:, :2]
    poses_proj = poses_proj.reshape([s[0], s[1], 2])

    return poses_proj


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
