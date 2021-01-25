import numpy as np


def transform_frame(poses_world, cam_par, project=False):
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
    for s, a, f in poses_world.keys():
        if cam_par[(s, a, f)] is None:
            return poses_world
        
        for c in list(cam_par[(s, a, f)].keys()):
            rcams = cam_par[(s, a, f)][c]
            poses = poses_world[(s, a, f)]
            
            Pcam = world_to_camera(poses, rcams)
            
            if project:
                Pcam = project_to_camera(Pcam, rcams["intr"])

            poses_cam[ (s, a, f + '.cam_' + str(c)) ] = Pcam

    # sort dictionary
    poses_cam = dict(sorted(poses_cam.items()))

    return poses_cam


def world_to_camera(poses_world, cam_par, rotate=True):
    """
    Rotate/translate 3d poses from world to camera viewpoint
    
    Args
        poses_world: array of poses in world coordinates of size n_frames x joints x n_dimensions
        cam_par: dictionary of camera parameters
        
    Returns
        poses_cam: poses in camera-centred coordinates
    """
    
    assert len(poses_world.shape) == 3
    
    if 'vis' in cam_par.keys():
        # ids = [i for i in cam_par['vis'].astype(bool) for j in range(3)]
        poses_world = poses_world[:,cam_par['vis'].astype(bool),:]
        
    ndim = poses_world.shape[1]
    n_poses = poses_world.shape[0]
    poses_world = np.reshape(poses_world, [-1, 3])
    
    if rotate:
        if len(cam_par['R'])==poses_world.shape[0]//(ndim//3): 
            poses_cam = np.zeros_like(poses_world)
            for i in range(poses_world.shape[0]):
                poses_cam[i,:] = np.matmul(cam_par['R'][i//(ndim//3)], poses_world[i,:])
        else:
            poses_cam = np.matmul(cam_par['R'], poses_world.T).T
                
    poses_cam += cam_par['tvec']
    poses_cam = np.reshape( poses_cam, [n_poses, ndim, 3] )
  
    return poses_cam


def camera_to_world(poses_cam, R, tvec):
    """
    Rotate/translate 3d poses from camera to world
    
    Args
        poses_cam: poses in camera coordinates
        cam_par: dictionary with camera parameters
        
    Returns
        poses_world: poses in world coordinates
    """

    ndim = poses_cam.shape[1]

    poses_world = np.reshape(poses_cam, [-1, 3]).copy()
    poses_world -= tvec
    poses_world = np.matmul(np.linalg.inv(R), poses_world.T).T
    poses_world = np.reshape(poses_world, [-1, ndim])

    return poses_world


def project_to_camera(poses, intr):
    """
    Project poses to camera frame
    
    Args
        poses: poses in camera coordinates
        intr: intrinsic camera matrix
        
    Returns
        poses_proj: 2D poses projected to camera plane
    """

    ndim = poses.shape[1]
    poses = np.reshape(poses, [-1, 3])
    poses_proj = np.squeeze(np.matmul(intr, poses[:, :, np.newaxis]))
    poses_proj = poses_proj / poses_proj[:, [2]]
    poses_proj = poses_proj[:, :2]
    poses_proj = np.reshape(poses_proj, [-1, int(ndim / 3 * 2)])

    return poses_proj


def XY_coord(poses):
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


def Z_coord(poses):
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
