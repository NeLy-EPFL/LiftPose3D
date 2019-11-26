
"""Utility functions for dealing with Drosophila melanogaster data."""

from __future__ import division

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pickle import load
import random
import pandas as pd

import glob
import copy
import procrustes

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

####### Preparing training, testing and OptoBot files #######
train_dir = "../flydata/flydata_train/"
TRAIN_FILES = [os.path.join(train_dir, f) \
     for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
TRAIN_FILES.sort()
test_dir = "../flydata/flydata_test/"
TEST_FILES = [os.path.join(test_dir, f) \
     for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
TEST_FILES.sort()
OPTOBOT_DIR = "../octopamineExperiments"
OPTOBOT_FILES = []
for d in os.walk(OPTOBOT_DIR):
    if len(d[2]) > 0:
        for f in d[2]:
            OPTOBOT_FILES.append(d[0]+"/"+f)

FILES = TRAIN_FILES + TEST_FILES
FILE_NUM = len(FILES)

random.shuffle(TRAIN_FILES)
random.shuffle(TEST_FILES)
####### --------------------------------------------- #######

####### raw DeepFly data information #######
LEGS = [list(range(5)), list(range(5,10)), list(range(10,15)),
    list(range(19,24)), list(range(24,29)), list(range(29,34))]
LEGS_JOINTS = []
for l in LEGS:
    for j in l:
        LEGS_JOINTS.append(j)

XY_COORD = [0,2]
Z_COORD = 1
####### ---------------------------- #######

####### DeepFly3D information #######
DF_LEGS = np.array(list(range(30)))

DF_NUM_JOINTS = len(DF_LEGS)

DF_BODY_COXA = np.array([0, 5, 10, 15, 20, 25])
DF_COXA_FEMUR = np.array([1, 6, 11, 16, 21, 26])
DF_NON_COXA_FEMUR = np.array([x for x in DF_LEGS if x not in DF_COXA_FEMUR])

DF_JOINTS_PER_LEG = 5

DF_COXA_FEMUR_3D = np.sort(np.hstack((DF_COXA_FEMUR*3, DF_COXA_FEMUR*3+1, DF_COXA_FEMUR*3+2)))
DF_NON_COXA_FEMUR_3D =\
        np.sort(np.hstack((DF_NON_COXA_FEMUR*3, DF_NON_COXA_FEMUR*3+1, DF_NON_COXA_FEMUR*3+2)))

DF_LEGS_2D = np.sort(np.hstack((DF_LEGS*2, DF_LEGS*2+1)))
DF_LEGS_3D = np.sort(np.hstack((DF_LEGS*3, DF_LEGS*3+1, DF_LEGS*3+2)))
DF_LEGS_Z = []
idx = Z_COORD
while idx <= max(DF_LEGS_3D):
    DF_LEGS_Z.append(idx)
    idx += 3
DF_LEGS_XY = [x for x in DF_LEGS_3D if x not in DF_LEGS_Z]

####### --------------------- #######

####### Used data information #######
OB_LEGS = np.array(list(range(24)))

OB_NUM_JOINTS = len(OB_LEGS)
OB_BODY_COXA = np.array([0, 4, 8, 12, 16, 20])

OB_JOINTS_PER_LEG = 4

OB_LEGS_2D = np.sort(np.hstack((OB_LEGS*2, OB_LEGS*2+1)))
OB_LEGS_3D = np.sort(np.hstack((OB_LEGS*3, OB_LEGS*3+1, OB_LEGS*3+2)))
OB_LEGS_Z = []
idx = Z_COORD
while idx <= max(OB_LEGS_3D):
    OB_LEGS_Z.append(idx)
    idx += 3
OB_LEGS_XY = [x for x in OB_LEGS_3D if x not in OB_LEGS_Z]
####### --------------------- #######

####### Dimensions to ignore #######
JOINTS_IGNORE = np.array([0])
JOINTS_IGNORE_2D = np.sort(np.hstack((JOINTS_IGNORE*2, JOINTS_IGNORE*2+1)))
JOINTS_IGNORE_3D = np.sort(np.hstack((JOINTS_IGNORE*3, JOINTS_IGNORE*3+1, JOINTS_IGNORE*3+2)))
####### -------------------- #######

def read_data(dir):
    # helper function to read Pickle files
    with (open(dir, "rb")) as file:
        try:
            return load(file)
        except EOFError:
            return None

def rotate_to_register(data3d):
    num_joints = 38
    data3d -= np.median(data3d)
    d3d_median = np.median(data3d, axis=0)

    origin = d3d_median[5, :]
    d3d_median -= origin
    x_new = +-(
        d3d_median[num_joints // 2 + 5]
        - d3d_median[num_joints // 2 + 15]
        )  # average the lines from both sides
    x_new /= np.linalg.norm(x_new)
    x_new[1] = 0  # set the y-axis to zero
    y_new = [0, 1, 0]
    z_new = np.cross(x_new, y_new)
    R = np.array([x_new, y_new, z_new])
    for idx in range(data3d.shape[0]):
        for j_idx in range(data3d[idx].shape[0]):
            j = data3d[idx][j_idx]
            data3d[idx, j_idx] = np.matmul(R, j)

    return data3d

def load_deepfly_reference():
    # load DeepFly3D reference data for procrustes
    dics = []
    for idx, f in enumerate(FILES):
        if "MDN_CsCh_Fly9" in f:
            d = read_data(f)['points3d']
            d = rotate_to_register(d)

            dics.append(d)
    
    d_data = np.vstack(dics)
    # keep only the legs
    d_data = d_data[:, LEGS_JOINTS, :]
    print("[+] done reading DeepFly3D reference, shape: ", d_data.shape)

    # keep only the walking frames
    return d_data[150:500]

def load_deepfly_data():
    # load all DeepFly3D data
    dics = []
    dims = np.zeros((FILE_NUM+1), dtype=int)
    train_test = np.zeros((FILE_NUM), dtype=int) # 1 is train
    for idx, f in enumerate(FILES):

        d = read_data(f)['points3d']
        
        dics.append(d)
        dims[idx+1] = dims[idx] + d.shape[0]
        if f in TRAIN_FILES:
          train_test[idx] = 1

    d_data = np.vstack(dics)
    # keep only the legs
    d_data = d_data[:, LEGS_JOINTS, :]
    print("[+] done reading DeepFly3D data, shape: ", d_data.shape)

    return d_data, dims, train_test

def load_optobot_data():
    # load all OptoBot data
    resnet = "DeepCut_resnet50_dlcTrackingAug6shuffle1_1030000"
    limbs = ["RF", "RM", "RH", "LF", "LM", "LH"]
    joints = ["bodyCoxa", "femurTibia", "tibiaTarsus", "claw"]

    dics = []
    dims = [0]
    for idx, f in enumerate(OPTOBOT_FILES):
        data = pd.read_hdf(f)
        len_ = len(data.values)
        data_np = np.empty((len(limbs)*len(joints), 2, len_), dtype=float)

        lj = 0
        for l in limbs:
            for j in joints:
                key = l + j
                for c, coord in enumerate(["x", "y"]):
                    data_np[lj, c] = data[(resnet, key, coord)].to_numpy()
                lj +=1

        new_data_np = np.empty((len_, len(limbs)*len(joints), 2), dtype=float)
        for f in range(len_):
            new_data_np[f] = data_np[:, :, f]

        dics.append(new_data_np)
        dims.append(dims[idx] + len_)

    d_data = np.vstack(dics)
    print("[+] done reading OptoBot data, shape: ", d_data.shape)
    return d_data, dims

def origin_first_joint(data):
    # fix the first joint to be in the origin
    for i in range(data.shape[0]):
        for coord in range(data.shape[2]):
            data[i, :, coord] -= data[i, 0, coord]
    return data

def adapt_to_deepfly(data2d):
    # adapt OptoBot data to DeepFly3D
    # reorder x, y, z coordinates
    data3d = np.zeros((data2d.shape[0], data2d.shape[1], data2d.shape[2]+1))
    data3d[:,:,:-1] = data2d
    x = -np.copy(data3d[:,:,1])
    y = np.copy(data3d[:,:,2])
    z = np.copy(data3d[:,:,0])
    data3d[:,:,0] = x
    data3d[:,:,1] = y
    data3d[:,:,2] = z

    return data3d

def normalization_stats(df_data_3d, ob_data_3d):
    '''
    Calculate normalization statistics for DeepFly3D data and OptoBot data.
    For X, Y coordinate statistics, both datasets are considered,
    for the depth Z, only DeepFly3D data contains information.
    Args
        df_data_3d: DeepFly3D data (n_samples, n_joints_yes_coxa_femur, 3)
        ob_data_3d: OptoBot data (n_samples_ob, n_joints_no_coxa_femur, 3). Depth is set to zero
    Return
        data_mean: the average value per joint per coordinate
        data_std: the standard deviation per joint per coordinate
        ob_dtu: OptoBot dimensions to use. subset of [0, 23]
        ob_dtu_flat_2d: OptoBot dimensions to use in 2d. subset of [0, 47]
        ob_dtu_flat_xy: OptoBot dimensions to use, only X, Y coordinates. subset of [0, 71]
        df_dtu: DeepFly3D dimensions to use. subset of [0, 29]
        df_dtu_flat_3d: DeepFly3D dimensions to use in 3d. subset of [0, 89]
    '''
    df_data_3d_nocoxa = np.delete(np.copy(df_data_3d), DF_COXA_FEMUR, axis=1)

    ob_data_2d_flat = np.reshape(ob_data_3d[:,:,XY_COORD], (-1, 2*OB_NUM_JOINTS))
    df_data_2d_flat = np.reshape(df_data_3d_nocoxa[:,:,XY_COORD], (-1, 2*OB_NUM_JOINTS))
    all_data_2d_flat = np.vstack([df_data_2d_flat, ob_data_2d_flat])
    
    df_data_3d_flat = np.reshape(df_data_3d, (-1, 3*DF_NUM_JOINTS))
    df_data_3d_nocoxa_flat = np.reshape(df_data_3d_nocoxa, (-1, 3*OB_NUM_JOINTS))
    df_data_3d_onlycoxa_flat = df_data_3d_flat[:, DF_COXA_FEMUR_3D]
    
    data_mean = np.zeros((df_data_3d_flat.shape[1],))
    data_mean_nocoxa = np.zeros((df_data_3d_nocoxa_flat.shape[1],))
    data_mean_nocoxa[OB_LEGS_XY] = np.mean(all_data_2d_flat, axis=0)
    data_mean_nocoxa[OB_LEGS_Z] = np.mean(df_data_3d_nocoxa_flat[:, OB_LEGS_Z], axis=0)
    data_mean[DF_NON_COXA_FEMUR_3D] = data_mean_nocoxa

    data_std = np.zeros((df_data_3d_flat.shape[1],))
    data_std_nocoxa = np.zeros((df_data_3d_nocoxa_flat.shape[1],))
    data_std_nocoxa[OB_LEGS_XY] = np.std(all_data_2d_flat, axis=0)
    data_std_nocoxa[OB_LEGS_Z] = np.std(df_data_3d_nocoxa_flat[:, OB_LEGS_Z], axis=0)
    data_std[DF_NON_COXA_FEMUR_3D] = data_std_nocoxa

    data_mean[DF_COXA_FEMUR_3D] = np.mean(df_data_3d_onlycoxa_flat, axis=0)
    data_std[DF_COXA_FEMUR_3D] = np.std(df_data_3d_onlycoxa_flat, axis=0)
  
    ob_dtu = [x for x in OB_LEGS if x not in JOINTS_IGNORE]
    ob_dtu_flat_3d = [x for x in OB_LEGS_3D if x not in JOINTS_IGNORE_3D]
    ob_dtu_flat_2d = [x for x in OB_LEGS_2D if x not in JOINTS_IGNORE_2D]
    ob_dtu_flat_xy = [x for x in OB_LEGS_XY if x not in JOINTS_IGNORE_3D]

    df_dtu = [x for x in DF_LEGS if x not in JOINTS_IGNORE]
    df_dtu_flat_3d = [x for x in DF_LEGS_3D if x not in JOINTS_IGNORE_3D]
    df_dtu_flat_2d = [x for x in DF_LEGS_2D if x not in JOINTS_IGNORE_2D]
    df_dtu_flat_xy = [x for x in DF_LEGS_XY if x not in JOINTS_IGNORE_3D]
    
    return data_mean, data_std, ob_dtu, ob_dtu_flat_2d,\
        ob_dtu_flat_xy, df_dtu, df_dtu_flat_3d

def normalize_data(data3d, data_mean, data_std, dtu_flat):
    '''
    normalize 3d (DeepFly3D) data.
    Args
        data3d: DeepFly3D data or 3d shapes data
        data_mean
        data_std
        dtu_flat: dimensions to use in 3d.
    Return
        normalized data
    '''
    data_out = np.reshape(np.copy(data3d), (-1, 3*DF_NUM_JOINTS))
    data_out[:, dtu_flat] = \
        np.divide((data_out[:, dtu_flat] - data_mean[dtu_flat]), data_std[dtu_flat])
  
    return np.reshape(data_out, (-1, DF_NUM_JOINTS, 3))

def normalize_ob_data(ob_data, data_mean, data_std, dtu_flat, dtu_flat_xy):
    '''
    normalize 2d (OptoBot) data.
    Args
        ob_data: OptoBot data or 2d shaped data
        data_mean
        data_std
        dtu_flat: dimensions to use in 2d.
        dtu_flat_xy: dimensions to use, only X, Y coordinates.
    Return
        normalized data
    '''
    data_out = np.reshape(np.copy(ob_data), (-1, 2*OB_NUM_JOINTS))
    data_out[:, dtu_flat] = \
        np.divide((data_out[:, dtu_flat] - data_mean[dtu_flat_xy]), data_std[dtu_flat_xy])

    return np.reshape(data_out, (-1, OB_NUM_JOINTS, 2))

def unNormalize_data(data3d, data_mean, data_std, dtu_flat):
    data_out = np.reshape(np.copy(data3d), (-1, 3*DF_NUM_JOINTS))
    data_out[:, dtu_flat] = \
        np.multiply(data_out[:, dtu_flat], data_std[dtu_flat]) + data_mean[dtu_flat]

    return np.reshape(data_out, (-1, DF_NUM_JOINTS, 3))

def unNormalize_ob_data(ob_data, data_mean, data_std, dtu_flat, dtu_flat_xy):
    data_out = np.reshape(np.copy(ob_data), (-1, 2*OB_NUM_JOINTS))
    data_out[:, dtu_flat] = \
        np.multiply(data_out[:, dtu_flat], data_std[dtu_flat_xy]) + data_mean[dtu_flat_xy]

    return np.reshape(data_out, (-1, OB_NUM_JOINTS, 2))

def unNormalize_batch(normalized_data, data_mean, data_std, dtu):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation. Some dimensions might also be missing
    Args
        normalized_data: nxd matrix to unnormalize
        data_mean: np vector with the mean of the data
        data_std: np vector with the standard deviation of the data
        dim_to_use: list of dimensions that are used in the original data
    Return
        orig_data: the input normalized_data, but unnormalized
    """
    T = normalized_data.shape[0] # Batch size
    D = data_mean.shape[0] # Dimensionality
    
    orig_data = np.zeros((T, D), dtype=np.float32)
    orig_data[:, dtu] = normalized_data

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat

    return orig_data

def read_3d_data():
    # Load DeepFly3D data
    df_data, dims, train_test = load_deepfly_data()
    df_reference = load_deepfly_reference()
    ob_data, ob_dims = load_optobot_data()

    # remove depth
    df_data_depth = np.copy(df_data[:,:,1])
    df_data[:,:,1] = 0
    df_reference[:,:,1] = 0

    df_data_nocoxa = np.delete(np.copy(df_data), DF_COXA_FEMUR, axis=1)
    df_reference_nocoxa = np.delete(np.copy(df_reference), DF_COXA_FEMUR, axis=1)

    # adapt OptoBot data to DeepFly
    ob_data_3d = adapt_to_deepfly(ob_data)

    # apply procrustes analysis to DeepFly and OptoBot, with reference to df_reference
    m_left, tform_left, m_right, tform_right =\
            procrustes.procrustes_get_tform(df_data_nocoxa, df_reference_nocoxa)
    # apply transform to all DeepFly3D data
    R_b, s_b, t_b = tform_left["rotation"], tform_left["scale"], tform_left["translation"]
    df_data[:, m_left] = procrustes.apply_transformation(np.copy(df_data[:, m_left]), R_b, t_b, s_b)
    R_b, s_b, t_b = tform_right["rotation"], tform_right["scale"], tform_right["translation"]
    df_data[:, m_right] = procrustes.apply_transformation(np.copy(df_data[:, m_right]), R_b, t_b, s_b)
    
    ob_data_3d = procrustes.procrustes_separate(ob_data_3d, df_reference_nocoxa)

    # give depth back to DeepFly data
    df_data[:,:,1] = df_data_depth
    
    # set first joint as origin
    df_data = origin_first_joint(df_data)
    ob_data_3d = origin_first_joint(ob_data_3d)

    np.save("saved_structures/df_data.npy", df_data)
    np.save("saved_structures/ob_data_3d.npy", ob_data_3d)

    data_mean, data_std, ob_dtu, ob_dtu_flat_2d, ob_dtu_flat_xy, df_dtu, df_dtu_flat_3d =\
        normalization_stats(df_data, ob_data_3d)
    
    # Divide every dimension independently
    df_data = normalize_data(df_data, data_mean, data_std, df_dtu_flat_3d)
    ob_data_3d[:,:,XY_COORD] =\
        normalize_ob_data(ob_data_3d[:,:,XY_COORD], data_mean[DF_NON_COXA_FEMUR_3D],
                          data_std[DF_NON_COXA_FEMUR_3D], ob_dtu_flat_2d, ob_dtu_flat_xy)
    
    return df_data, ob_data_3d, dims, ob_dims, train_test, data_mean, data_std,\
        ob_dtu, ob_dtu_flat_2d, ob_dtu_flat_xy, df_dtu, df_dtu_flat_3d
