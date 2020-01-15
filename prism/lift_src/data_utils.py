
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
train_dir = "../../flydata/flydata_train/"
TRAIN_FILES = [os.path.join(train_dir, f) \
     for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
TRAIN_FILES.sort()
test_dir = "../../flydata/flydata_test/"
TEST_FILES = [os.path.join(test_dir, f) \
     for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
TEST_FILES.sort()

prism_dir = "../prism_data/"

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
DEPTH_COORD = 1
####### ---------------------------- #######

####### DeepFly3D information #######
LEGS = np.array(list(range(30)))
LEGS_2D = np.sort(np.hstack((LEGS*2, LEGS*2+1)))
LEGS_3D = np.sort(np.hstack((LEGS*3, LEGS*3+1, LEGS*3+2)))

NUM_JOINTS = len(DF_LEGS)

BODY_COXA = np.array([0, 5, 10, 15, 20, 25])
COXA_FEMUR = np.array([1, 6, 11, 16, 21, 26])

JOINTS_PER_LEG = 5
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

def load_prism_data():
    # load all prism data
    dics = []
    dims = np.zeros((PRISM_FILE_NUM+1), dtype=int)
    for idx, f in enumerate(PRISM_FILES):
        # TODO: save data in d

        dics.append(d)
        dims[idx+1] = dims[idx] + d.shape[0]

    d_data = np.vstack(dics)
    print("[+] done reading OptoBot data, shape: ", d_data.shape)
    return d_data, dims

def origin_first_joint(data):
    # fix the first joint to be in the origin
    for i in range(data.shape[0]):
        for coord in range(data.shape[2]):
            data[i, :, coord] -= data[i, 0, coord]
    return data

def normalization_stats(df_data, pr_data):
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
    df_data_flat = np.reshape(df_data, (-1, 3*NUM_JOINTS))
    pr_data_flat = np.reshape(pr_data, (-1, 3*NUM_JOINTS))
    all_data_flat = np.vstack([df_data_flat, pr_data_flat])
    
    data_mean = np.mean(all_data_flat, axis=0)
    data_std = np.std(all_data_flat, axis=0)

    dtu = [x for x in LEGS if x not in JOINTS_IGNORE]
    dtu_flat = [x for x in LEGS_3D if x not in JOINTS_IGNORE_3D]
    
    return data_mean, data_std, dtu, dtu_flat

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
    data_out = np.reshape(np.copy(data3d), (-1, 3*NUM_JOINTS))
    data_out[:, dtu_flat] = \
        np.divide((data_out[:, dtu_flat] - data_mean[dtu_flat]), data_std[dtu_flat])
  
    return np.reshape(data_out, (-1, NUM_JOINTS, 3))

def unNormalize_data(data3d, data_mean, data_std, dtu_flat):
    data_out = np.reshape(np.copy(data3d), (-1, 3*DF_NUM_JOINTS))
    data_out[:, dtu_flat] = \
        np.multiply(data_out[:, dtu_flat], data_std[dtu_flat]) + data_mean[dtu_flat]

    return np.reshape(data_out, (-1, DF_NUM_JOINTS, 3))

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
    df_data, df_dims, train_test = load_deepfly_data()
    df_reference = load_deepfly_reference()
    pr_data, pr_dims = load_prism_data()

    # remove depth
    df_data_depth = np.copy(df_data[:,:,1])
    df_data[:,:,DEPTH_COORD] = 0
    df_reference[:,:,DEPTH_COORD] = 0
    # pr_data[:,:,DEPTH_COORD] = 0 ???

    # TODO: if we have PRISM data in 3d too
    # maybe we should apply procrustes before
    # removing the depth

    # apply procrustes analysis to DeepFly and prism, with reference to df_reference
    df_data = procrustes.procrustes_separate(df_data, df_reference)

    pr_data = procrustes.procrustes_separate(pr_data, df_reference)

    # give depth back to DeepFly data
    df_data[:,:,1] = df_data_depth
    
    # set first joint as origin
    df_data = origin_first_joint(df_data)
    pr_data = origin_first_joint(pr_data)

    np.save("saved_structures/df_data.npy", df_data)
    np.save("saved_structures/pr_data.npy", pr_data)

    data_mean, data_std, dtu, dtu_flat = normalization_stats(df_data, pr_data)
    
    # Divide every dimension independently
    df_data = normalize_data(df_data, data_mean, data_std, dtu_flat)
    pr_data = normalize_data(pr_data, data_mean, data_std, dtu_flat)
    
    return df_data, pr_data, df_dims, pr_dims, train_test, data_mean, data_std,\
        dtu, dtu_flat
