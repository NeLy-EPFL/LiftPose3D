import numpy as np


def world_to_camera(poses_world, cam_par):
    """
    Rotate/translate 3d poses from world to camera viewpoint
    
    Args
        poses_world: array of poses in world coordinates of size n_frames x n_dimensions
        cam_par: dictionary of camera parameters
        
    Returns
        poses_cam: poses in camera-centred coordinates
    """
    
    if 'vis' in cam_par.keys():
        ids = [i for i in cam_par['vis'].astype(bool) for j in range(3)]
        poses_world = poses_world[:,ids]
        
    ndim = poses_world.shape[1]
    poses_world = np.reshape(poses_world, [-1, 3])
  
    assert len(poses_world.shape) == 2
    assert poses_world.shape[1] == 3
  
    poses_cam =  np.matmul(cam_par['R'], poses_world.T).T + cam_par['tvec']
    poses_cam = np.reshape( poses_cam, [-1, ndim] )
  
    return poses_cam


def camera_to_world( poses_cam, cam_par ):
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
    poses_world -= cam_par['tvec']
    poses_world = np.matmul(np.linalg.inv(cam_par['R']), poses_world.T).T
    poses_world = np.reshape( poses_world, [-1, ndim] )
    
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
    poses_proj = np.squeeze(np.matmul(intr, poses[:,:,np.newaxis]))
    poses_proj = poses_proj / poses_proj[:, [2]]
    poses_proj = poses_proj[:, :2]
    poses_proj = np.reshape( poses_proj, [-1, int(ndim/3*2)] )
  
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
        t3d = poses[ key ]

        ndim = t3d.shape[1]
        XY = np.reshape(t3d, [-1, 3])
        XY = XY[:,:2]
        poses_xy[ key ] = np.reshape( XY, [-1, ndim//3*2] )
 
    return poses_xy


def Z_coord( poses):
    """
    Project 3d poses to XY plane
    
    Args
        poses: poses
        
    Returns
        poses_xy: poses projected to xy plane   
    """
    
    poses_z = {}
    for key in poses.keys():
        t3d = poses[ key ]

        ndim = t3d.shape[1]
        Z = np.reshape(t3d, [-1, 3])
        Z = Z[:,2]
        poses_z[ key ] = np.reshape( Z, [-1, ndim//3] )

    return poses_z