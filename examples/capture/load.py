import scipy.io
import numpy as np
<<<<<<< HEAD
import cv2


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
=======
from liftpose.vision_3d import normalize_bone_length, world_to_camera, project_to_camera
>>>>>>> c7c02553135b4aea36a90fc1c84c06062bbee474

# naming scheme used in the capture dataset for different cameras
cam_list = ['R', 'L', 'E', 'U', 'S', 'U2']

def read_data(path, session_id, cam_id):
    return scipy.io.loadmat(f'{path}/data_3d_e{str(session_id)}/{str(session_id)}_cam{str(cam_id+1)}_data.mat')


def read_cam(path, session_id, cam_id):
    return scipy.io.loadmat(f'{path}/calibration_e{session_id}/hires_cam{cam_id+1}_params_rRDistort.mat')


<<<<<<< HEAD
from liftpose.vision_3d import normalize_bone_length


def get_btch(session_id, cam_list, par, par_data, bone_length):
    train_2d, train_3d = list(), list()
    mat = [read_data(session_id, cid) for (cid, _) in enumerate(cam_list)]
    for cam_id in range(len(cam_list)):
        #mat = [read_data(session_id, cid) for (cid, _) in enumerate(cam_list)]
        c = read_cam(session_id, cam_id)
        pts3d = mat[cam_id]["data_3d"].reshape(-1, 20, 3)
        pts3d = pts3d.reshape(-1, 3)

        pts2d = mat[cam_id]["data_2d"].reshape(-1, 20, 2)
        distort = np.array([c["RDistort"][0][0], c["RDistort"][0][1], 0, 0])
        pts2d = cv2.undistortPoints(
            src=pts2d.reshape(-1, 2)[:, None, :],
            cameraMatrix=c["K"].T,
            distCoeffs=distort,
            P=c["K"].T,
        )
        pts2d = pts2d.reshape(54000, 20, 2)
        # pts2d, _ = cv2.projectPoints(pts3d, rvec=c['r'].T, tvec=c['t'], cameraMatrix=c['K'].T, distCoeffs=None)
        # pts2d = pts2d.reshape(-1, 20, 2)
        train_2d.append(pts2d)

        # bone length normalization
        pts3d = world_to_camera2(pts3d, c["r"].T, c["t"])
        pts3d = pts3d.reshape(-1, 20, 3)
        pts3d = normalize_bone_length(
            pts3d.copy(),
            root=par["roots"][0],
            child=par_data["vis"]["child"],
            bone_length=bone_length,
            thr=10,
        )

        train_3d.append(pts3d)

    return train_2d, train_3d

=======
def load_data(par_train, return_frame_info=False):
    
    print('Loading data...')
>>>>>>> c7c02553135b4aea36a90fc1c84c06062bbee474

    train_2d, train_3d, test_2d, test_3d = [], [], [], []
    train_fid, test_fid = [], []
    train_2d_sh, test_2d_sh = [], []
    train_keypoints, test_keypoints = [], []
    session_id_list = par_train["train_session_id"] + par_train["test_session_id"]
    cams = {'R': [], 'tvec': [], 'intr': []}
    for session_id in session_id_list:
        mat = [read_data(par_train['data_dir'], session_id, cid) for (cid,_) in enumerate(cam_list)]
        for cam_id in par_train["test_cam_id"]:
            c = read_cam(par_train['data_dir'],session_id, cam_id)

<<<<<<< HEAD
    T_inverse = -1 * np.matmul(R.T, np.squeeze(T))
=======
            pts2d_sh = mat[cam_id]['data_2d'].reshape(-1, 20, 2)
            pts3d = mat[cam_id]['data_3d'].reshape(-1, 20, 3)
            pts3d = world_to_camera(pts3d, c['r'].T, c['t'])
            pts2d = project_to_camera(pts3d, c['K'].T)
            # bone length normalization
            #pts3d = normalize_bone_length(pts3d.copy(), root=par["roots"][0], child=par_data["vis"]["child"], bone_length=bone_length, thr=10)
        
            frame_id = mat[cam_id]['data_sampleID']
>>>>>>> c7c02553135b4aea36a90fc1c84c06062bbee474

            if session_id in par_train["train_session_id"]:
                train_2d.append(pts2d)
                train_3d.append(pts3d)
                train_fid.append(frame_id)
                train_2d_sh.append(pts2d_sh)

            if session_id in par_train["test_session_id"]:
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

<<<<<<< HEAD
import scipy.io


def read_data(session_id, cam_id):
    return scipy.io.loadmat(
        f"/home/user/Dropbox/data_3d_e{str(session_id)}/{str(session_id)}_cam{str(cam_id+1)}_data.mat"
    )


def read_cam(session_id, cam_id):
    return scipy.io.loadmat(
        f"/home/user/Dropbox/calibration_e{session_id}/hires_cam{cam_id+1}_params_rRDistort.mat"
    )


import os
import matplotlib.pyplot as plt


def plot_pose_2d(frame_id, cam_id, ax, mat, cam_list):
    idx = mat[0]["data_sampleID"][frame_id][0]
    p = f"/data/rat7M_e0/sample0_{idx}_Camera{cam_list[cam_id]}.png"
    if os.path.isfile(p):
        try:
            ax.set_data(plt.imread(p))
            return ax
        except:
            return ax.imshow(plt.imread(p))
    else:
        return ax
=======
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
    
    # impute nan's with zeros. zero 3d points will not be counted towards loss.
    #train_2d[np.isnan(train_2d)] = 0
    #train_3d[np.isnan(train_3d)] = 0
    #test_2d[np.isnan(test_2d)] = 0
    #test_3d[np.isnan(test_3d)] = 0
    
    print('OK')
    
    if return_frame_info:
        return train_3d, train_2d, train_keypoints, test_3d, test_2d, test_keypoints, cams, train_fid, test_fid, train_2d_sh, test_2d_sh
    
    else:
        return train_3d, train_2d, train_keypoints, test_3d, test_2d, test_keypoints, cams


>>>>>>> c7c02553135b4aea36a90fc1c84c06062bbee474
