import numpy as np

def world_to_camera(poses_world, cam_par, rotate=True):
    """
    Rotate/translate 3d poses from world to camera viewpoint
    
    Args
        poses_world: array of poses in world coordinates of size n_frames x n_dimensions
        cam_par: dictionary of camera parameters
        
    Returns
        poses_cam: poses in camera-centred coordinates
    """

    assert len(poses_world.shape) == 2
    
    if 'vis' in cam_par.keys():
        ids = [i for i in cam_par['vis'].astype(bool) for j in range(3)]
        poses_world = poses_world[:,ids]
        
    ndim = poses_world.shape[1]
    poses_world = np.reshape(poses_world, [-1, 3])
    
    if rotate:
        if len(cam_par['R'])==poses_world.shape[0]//(ndim//3): 
            poses_cam = np.zeros_like(poses_world)
            for i in range(poses_world.shape[0]):
                poses_cam[i,:] = np.matmul(cam_par['R'][i//(ndim//3)], poses_world[i,:])
        else:
            poses_cam = np.matmul(cam_par['R'], poses_world.T).T
                
    poses_cam += cam_par['tvec']
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


def project_to_camera(poses, cam_par=[16041.0, 15971.7, 240, 480]):
    """
    Project poses to camera frame
    
    Args
        poses: poses in camera coordinates
        intr: intrinsic camera matrix
        
    Returns
        poses_proj: 2D poses projected to camera plane
    """
    
    intr = np.array(
                [
                    [cam_par[0], 0, cam_par[2]],
                    [0, cam_par[1], cam_par[3]],
                    [0, 0, 1],
                ],
            dtype=float,
            )
    
    ndim = poses.shape[1]
    poses_proj = np.reshape(poses, [-1, 3])
    poses_proj = np.matmul(intr, poses_proj.T).T
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


def procrustes(X, Y, scaling=True, reflection='best'):
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

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform