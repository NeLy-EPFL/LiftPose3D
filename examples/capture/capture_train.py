import torch
import yaml
import logging
from imp import reload
import matplotlib.pyplot as plt
import yaml
import numpy as np
import cv2
import os
from tqdm import tqdm
from load import *

tqdm.get_lock().locks = []
reload(logging)
logger = logging.getLogger(__name__).setLevel(logging.INFO)


# decleare data parameters
# out 00 -> without bln, 0.004
par_train = {
    "data_dir": "",  # change the path
    "out_dir": "/home/user/Desktop/LiftPose3D/examples/capture/out_across_undistort_bln_drop/",
    "train_session_id": [1, 2, 3],
    "test_session_id": [0],
    "test_cam_id": [0,1,2,3,4,5],
}

# merge with training parameters
par_data = yaml.full_load(
    open("/home/user/Desktop/LiftPose3D/examples/capture/param.yaml", "rb")
)
par = {**par_data["data"], **par_train}

# meta = mat73.loadmat('nolj_Recording_day8_caff1_nolj_imputed.mat')
# naming scheme used in the capture dataset for different cameras
cam_list = np.array(["R", "L", "E", "U", "S", "U2"])

bone_length = {
    (0, 3): 80.98554459477106,
    (3, 5): 112.90207543313412,
    (5, 8): 28.733636975647393,
    (5, 9): 34.84953102842617,
    (10, 11): 15.956945352305892,
    (3, 12): 29.946991312076776,
    (3, 13): 19.409450872458986,
    (10, 12): 40.857306953944935,
    (13, 14): 37.08634981412786,
    (14, 15): 20.954239695086592,
    (8, 17): 35.39218881579121,
    (9, 16): 25.59466626887338,
    (17, 18): 31.033650232299784,
    (16, 19): 25.244471236727712,
}


import scipy.io
from load import world_to_camera2
from liftpose.vision_3d import normalize_bone_length

train_2d, train_3d, test_2d, test_3d = list(), list(), list(), list()
for s in par_train["train_session_id"]:
    tr2d, tr3d = get_btch(s, cam_list, par, par_data, bone_length)
    train_2d.extend(tr2d)
    train_3d.extend(tr3d)

for s in par_train["test_session_id"]:
    te2d, te3d = get_btch(s, cam_list[par_train["test_cam_id"]], par, par_data, bone_length)
    test_2d.extend(te2d)
    test_3d.extend(te3d)

train_2d = np.concatenate(train_2d, axis=0)
train_3d = np.concatenate(train_3d, axis=0)
test_2d = np.concatenate(test_2d, axis=0)
test_3d = np.concatenate(test_3d, axis=0)
train_keypoints = np.logical_not(np.isnan(train_3d))
test_keypoints = np.logical_not(np.isnan(test_3d))

# if more than one third is missing remove it
train_keypoints[np.sum(np.logical_not(train_keypoints), axis=(1,2)) > 20] = False
test_keypoints[np.sum(np.logical_not(test_keypoints), axis=(1,2)) > 20] = False

# if the root is none, then ignore that point, otherwise we cannot anchor
train_keypoints[np.any(np.isnan(train_2d[:,3]),axis=-1)] = False
test_keypoints[np.any(np.isnan(test_2d[:,3]),axis=-1)] = False

from liftpose.main import train_np

train_np(
    train_2d=train_2d,
    test_2d=test_2d,
    train_3d=train_3d,
    test_3d=test_3d,
    out_dir=par["out_dir"],
    root=par["roots"][0],
    train_keypts=train_keypoints,
    test_keypts=test_keypoints,
    training_kwargs={
        "epochs": 100,
        "lr_decay": 50000,
        "lr_gamma": 0.95,
        "drop_input": 0.05,
    },
)

