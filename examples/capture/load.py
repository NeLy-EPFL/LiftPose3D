import numpy as np
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

    ndim = P.shape[1]
    P = np.reshape(P, [-1, 3])

    assert len(P.shape) == 2
    assert P.shape[1] == 3

    P_rot = np.matmul(R, P.T).T + T

    return np.reshape(P_rot, [-1, ndim])


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

    T_inverse = -1 * np.matmul(R.T, np.squeeze(T))

    return world_to_camera2(P, R.T, T_inverse).reshape(s)


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
