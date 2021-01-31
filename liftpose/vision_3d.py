import numpy as np


def reprojection_error(poses_3d, poses_2d, R, tvec, intr):
    poses_3d = world_to_camera(poses_3d, R, tvec)
    return np.linalg.norm(pose_2d - project_to_camera(poses_3d, intr), axis=2)


def world_to_camera_dict(poses_world:dict, cam_par:dict):
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

    return poses_cam


def world_to_camera(poses_world:np.ndarray, R:np.ndarray, tvec:np.ndarray):
    """
    Rotate/translate 3d poses from world to camera viewpoint
    
    Args
        poses_world: array of poses in world coordinates of size n_frames x n_dimensions
        cam_par: dictionary of camera parameters
        
    Returns
        poses_cam: poses in camera-centred coordinates
    """
    s = poses_world.shape
    poses_world = np.reshape(poses_world, [s[0]*s[1], 3])

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


def project_to_camera(poses:np.ndarray, intr:np.ndarray):
    """
    Project poses to camera frame
    
    Args
        poses: poses in camera coordinates, [N, J, 3]
        intr: intrinsic camera matrix, [3, 3] 
        
    Returns
        poses_proj: 2D poses projected to camera plane
    """
    s = poses.shape
    poses = np.reshape(poses, [s[0]*s[1], 3])

    poses_proj = np.squeeze(np.matmul(intr, poses.T)).T
    poses_proj = poses_proj / poses_proj[:, [2]]
    poses_proj = poses_proj[:, :2]
    poses_proj = poses_proj.reshape([s[0], s[1], 2])

    return poses_proj


def XY_coord_dict(poses:dict):
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


def Z_coord_dict(poses:dict):
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
