import scipy.io
import numpy as np
from liftpose.vision_3d import world_to_camera, project_to_camera

# naming scheme used in the capture dataset for different cameras
cam_list = ['R', 'L', 'E', 'U', 'S', 'U2']

def read_data(path, session_id, cam_id):
    return scipy.io.loadmat(f'{path}/data_3d_e{str(session_id)}/{str(session_id)}_cam{str(cam_id+1)}_data.mat')


def read_cam(path, session_id, cam_id):
    return scipy.io.loadmat(f'{path}/calibration_e{session_id}/hires_cam{cam_id+1}_params_rRDistort.mat')


def load_data(par_train, return_frame_info=False):
    
    print('Loading data...')

    train_2d, train_3d, test_2d, test_3d = [], [], [], []
    train_fid, test_fid = [], []
    train_2d_sh, test_2d_sh = [], []
    train_keypoints, test_keypoints = [], []
    session_id_list = par_train["train_session_id"] + par_train["test_session_id"]
    cams = {'R': [], 'tvec': [], 'intr': []}
    for session_id in session_id_list:
        mat = [read_data(par_train['data_dir'], session_id, cid) for (cid,_) in enumerate(cam_list)]
        for cam_id in range(len(cam_list)):
            c = read_cam(par_train['data_dir'],session_id, cam_id)

            pts2d_sh = mat[cam_id]['data_2d'].reshape(-1, 20, 2)
            pts3d = mat[cam_id]['data_3d'].reshape(-1, 20, 3)
            pts3d = world_to_camera(pts3d, c['r'].T, c['t'])
            pts2d = project_to_camera(pts3d, c['K'].T)
        
            frame_id = mat[cam_id]['data_sampleID']

            if session_id in par_train["train_session_id"]:
                train_2d.append(pts2d)
                train_3d.append(pts3d)
                train_fid.append(frame_id)
                train_2d_sh.append(pts2d_sh)

            if session_id in par_train["test_session_id"] and cam_id in par_train["test_cam_id"]:
                test_2d.append(pts2d)
                test_3d.append(pts3d)                
                test_fid.append(frame_id)
                test_2d_sh.append(pts2d_sh)
                cams['R'].append(c['r'].T)
                cams['tvec'].append(c['t'])
                cams['intr'].append(c['K'].T)
            
    train_2d = np.concatenate(train_2d, axis=0)
    train_3d = np.concatenate(train_3d, axis=0)
    test_2d = np.concatenate(test_2d, axis=0)
    test_3d = np.concatenate(test_3d, axis=0)
    train_2d_sh = np.concatenate(train_2d_sh, axis=0)
    test_2d_sh = np.concatenate(test_2d_sh, axis=0)
    train_fid = np.concatenate(train_fid, axis=0)
    test_fid = np.concatenate(test_fid, axis=0)
    train_keypoints = np.logical_not(np.isnan(train_3d))
    test_keypoints = np.logical_not(np.isnan(test_3d))

    # if we have unknown keypoints, we cannot rescale
    ind_train = np.sum(np.logical_not(train_keypoints), axis=(1,2))<20
    ind_test = np.sum(np.logical_not(test_keypoints), axis=(1,2))<20
    train_2d = train_2d[ind_train]
    train_3d = train_3d[ind_train]
    test_2d = test_2d[ind_test]
    test_3d = test_3d[ind_test]
    train_2d_sh = train_2d_sh[ind_train]
    test_2d_sh = test_2d_sh[ind_test]
    train_fid = train_fid[ind_train]
    test_fid = test_fid[ind_test]
    train_keypoints = train_keypoints[ind_train]
    test_keypoints = test_keypoints[ind_test]
    
    print('OK')
    
    if return_frame_info:
        return train_3d, train_2d, train_keypoints, test_3d, test_2d, test_keypoints, cams, train_fid, test_fid, train_2d_sh, test_2d_sh
    
    else:
        return train_3d, train_2d, train_keypoints, test_3d, test_2d, test_keypoints, cams