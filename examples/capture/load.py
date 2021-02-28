import scipy.io
import numpy as np
from liftpose.vision_3d import normalize_bone_length, world_to_camera, project_to_camera

# naming scheme used in the capture dataset for different cameras
cam_list = ['R', 'L', 'E', 'U', 'S', 'U2']

def read_data(path, session_id, cam_id):
    return scipy.io.loadmat(f'{path}/data_3d_e{str(session_id)}/{str(session_id)}_cam{str(cam_id+1)}_data.mat')

def read_cam(path, session_id, cam_id):
    return scipy.io.loadmat(f'{path}/calibration_e{session_id}/hires_cam{cam_id+1}_params_rRDistort.mat')

def load_data(par_train):
    
    print('Loading data...')

    train_2d, train_3d, test_2d, test_3d = [], [], [], []
    train_keypoints, test_keypoints = [], []
    session_id_list = par_train["train_session_id"] + par_train["test_session_id"]
    cams = {'R': [], 'tvec': [], 'intr': []}
    for session_id in session_id_list:
        mat = [read_data(par_train['data_dir'], session_id, cid) for (cid,_) in enumerate(cam_list)]
        for cam_id in par_train["test_cam_id"]:
            c = read_cam(par_train['data_dir'],session_id, cam_id)

            #pts2d = mat[cam_id]['data_2d'].reshape(-1, 20, 2)
            pts3d = mat[cam_id]['data_3d'].reshape(-1, 20, 3)
            pts3d_tmp = world_to_camera(pts3d.copy(), c['r'].T, c['t'])
            pts2d = project_to_camera(pts3d_tmp, c['K'].T)
            # bone length normalization
            #pts3d = normalize_bone_length(pts3d.copy(), root=par["roots"][0], child=par_data["vis"]["child"], bone_length=bone_length, thr=10)
        
            cams['R'].append(c['r'].T)
            cams['tvec'].append(c['t'])
            cams['intr'].append(c['K'].T)

            if session_id in par_train["train_session_id"]:
                train_2d.append(pts2d)
                train_3d.append(pts3d)

            if session_id in par_train["test_session_id"]:
                test_3d.append(pts3d)
                test_2d.append(pts2d)
            
    train_2d = np.concatenate(train_2d, axis=0)
    train_3d = np.concatenate(train_3d, axis=0)
    test_2d = np.concatenate(test_2d, axis=0)
    test_3d = np.concatenate(test_3d, axis=0)
    train_keypoints = np.logical_not(np.isnan(train_3d))
    test_keypoints = np.logical_not(np.isnan(test_3d))

    # if we have unknown keypoints, we cannot rescale
    ind_train = np.sum(np.logical_not(train_keypoints), axis=(1,2))<20
    ind_test = np.sum(np.logical_not(test_keypoints), axis=(1,2))<20
    train_2d = train_2d[ind_train]
    train_3d = train_3d[ind_train]
    test_2d = test_2d[ind_test]
    test_3d = test_3d[ind_test]
    train_keypoints = train_keypoints[ind_train]
    test_keypoints = test_keypoints[ind_test]
    
    # impute nan's with zeros. zero 3d points will not be counted towards loss.
    #train_2d[np.isnan(train_2d)] = 0
    #train_3d[np.isnan(train_3d)] = 0
    #test_2d[np.isnan(test_2d)] = 0
    #test_3d[np.isnan(test_3d)] = 0
    
    print('OK')
    
    return train_3d, train_2d, train_keypoints, test_3d, test_2d, test_keypoints, cams


