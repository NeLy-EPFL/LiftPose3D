import numpy as np
def world_to_camera2(P, R, T):
    """
  Rotate/translate 3d poses to camera viewpoint
  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    transf: Nx2 points on camera
  """

    ndim = P.shape[1]
    P = np.reshape(P, [-1, 3])

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    P_rot = np.matmul(R, P.T).T + T

    return np.reshape(P_rot, [-1, ndim])



def camera_to_world(P, R, T):
    """
  Rotate/translate 3d poses to camera viewpoint
  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    transf: Nx2 points on camera
  """
    s = P.shape
    P = P.reshape(P.shape[0], -1)

    T_inverse =  -1 * np.matmul(R.T,  np.squeeze(T))

    return world_to_camera2(P, R.T, T_inverse).reshape(s)



import scipy.io
def read_data(session_id, cam_id):
    return scipy.io.loadmat(f'/home/user/Dropbox/data_3d_e{str(session_id)}/{str(session_id)}_cam{str(cam_id+1)}_data.mat')

def read_cam(session_id, cam_id):
    return scipy.io.loadmat(f'/home/user/Dropbox/calibration_e{session_id}/hires_cam{cam_id+1}_params_rRDistort.mat')