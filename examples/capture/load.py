import scipy.io
def read_data(session_id, cam_id):
    return scipy.io.loadmat(f'/data/LiftPose3D/capture/data_3d_e{str(session_id)}/{str(session_id)}_cam{str(cam_id+1)}_data.mat')

def read_cam(session_id, cam_id):
    return scipy.io.loadmat(f'/data/LiftPose3D/capture/calibration_e{session_id}/hires_cam{cam_id+1}_params_rRDistort.mat')