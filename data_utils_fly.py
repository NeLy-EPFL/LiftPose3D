"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import glob
import copy
import torch
from src.normalize import normalize_data, anchor
import pickle


TRAIN_SUBJECTS = [0,1,2,3,4,5,6,7]
TEST_SUBJECTS  = [8,9]

anchors = [0, 5, 10, 19, 24, 29]
target_sets = [[ 1,  2,  3,  4], [ 6,  7,  8,  9], [11, 12, 13, 14],
               [20, 21, 22, 23], [25, 26, 27, 28], [30, 31, 32, 33]]

MARKER_NAMES = ['']*38
MARKER_NAMES[0] = 'BODY_COXA'
MARKER_NAMES[1] = 'COXA_FEMUR'
MARKER_NAMES[2] = 'FEMUR_TIBIA'
MARKER_NAMES[3] = 'TIBIA_TARSUS'
MARKER_NAMES[4] = 'TARSUS_TIP'
MARKER_NAMES[5] = 'BODY_COXA'
MARKER_NAMES[6] = 'COXA_FEMUR'
MARKER_NAMES[7] = 'FEMUR_TIBIA'
MARKER_NAMES[8] = 'TIBIA_TARSUS'
MARKER_NAMES[9] = 'TARSUS_TIP'
MARKER_NAMES[10] = 'BODY_COXA'
MARKER_NAMES[11] = 'COXA_FEMUR'
MARKER_NAMES[12] = 'FEMUR_TIBIA'
MARKER_NAMES[13] = 'TIBIA_TARSUS'
MARKER_NAMES[14] = 'TARSUS_TIP'

MARKER_NAMES[19] = 'BODY_COXA'
MARKER_NAMES[20] = 'COXA_FEMUR'
MARKER_NAMES[21] = 'FEMUR_TIBIA'
MARKER_NAMES[22] = 'TIBIA_TARSUS'
MARKER_NAMES[23] = 'TARSUS_TIP'
MARKER_NAMES[24] = 'BODY_COXA'
MARKER_NAMES[25] = 'COXA_FEMUR'
MARKER_NAMES[26] = 'FEMUR_TIBIA'
MARKER_NAMES[27] = 'TIBIA_TARSUS'
MARKER_NAMES[28] = 'TARSUS_TIP'
MARKER_NAMES[29] = 'BODY_COXA'
MARKER_NAMES[30] = 'COXA_FEMUR'
MARKER_NAMES[31] = 'FEMUR_TIBIA'
MARKER_NAMES[32] = 'TIBIA_TARSUS'
MARKER_NAMES[33] = 'TARSUS_TIP'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*38
SH_NAMES[0] = 'BODY_COXA'
SH_NAMES[1] = 'COXA_FEMUR'
SH_NAMES[2] = 'FEMUR_TIBIA'
SH_NAMES[3] = 'TIBIA_TARSUS'
SH_NAMES[4] = 'TARSUS_TIP'
SH_NAMES[5] = 'BODY_COXA'
SH_NAMES[6] = 'COXA_FEMUR'
SH_NAMES[7] = 'FEMUR_TIBIA'
SH_NAMES[8] = 'TIBIA_TARSUS'
SH_NAMES[9] = 'TARSUS_TIP'
SH_NAMES[10] = 'BODY_COXA'
SH_NAMES[11] = 'COXA_FEMUR'
SH_NAMES[12] = 'FEMUR_TIBIA'
SH_NAMES[13] = 'TIBIA_TARSUS'
SH_NAMES[14] = 'TARSUS_TIP'

data_dir = '/data/DF3D/'#'/Users/adamgosztolai/Dropbox/'#
actions = ['MDN_CsCh']
rcams = pickle.load(open('cameras.pkl', "rb"))
camera_frame = True #boolean. Whether to convert the data to camera coordinates


def main():
    
    # HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std = \
    read_2d_predictions( actions, data_dir )
    
    torch.save(train_set, '/data/DF3D/train_2d_ft.pth.tar')
    torch.save(test_set, '/data/DF3D/test_2d_ft.pth.tar')
    
    #xy data
    train_set, test_set, data_mean, data_std = \
    create_xy_data( actions, data_dir, rcams, target_sets, anchors )
        
    torch.save(train_set, data_dir + 'train_2d.pth.tar')
    torch.save(test_set, data_dir + 'test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'anchors': anchors},
                data_dir + 'stat_2d.pth.tar')
    
    #z data
    train_set, test_set, data_mean, data_std = \
    create_z_data( actions, data_dir, rcams, target_sets, anchors )
        
    torch.save(train_set, data_dir + 'train_z.pth.tar')
    torch.save(test_set, data_dir + 'test_z.pth.tar')   
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'anchors': anchors},
                data_dir + 'stat_z.pth.tar')
    
    #3D ground truth
#    train_set, test_set, data_mean, data_std = \
#    read_3d_data( actions, data_dir, camera_frame, rcams, target_sets, anchors)
#    
#    torch.save(train_set, data_dir +'train_3d.pth.tar')
#    torch.save(test_set, data_dir + 'test_3d.pth.tar')
#    torch.save({'mean': data_mean, 'std': data_std, 
#                'target_sets': target_sets, 'anchors': anchors},
#                data_dir + 'stat_3d.pth.tar')
    
#    return train_set_sh, train_set_2d, train_set


# =============================================================================
# Preprocess pipelines
# =============================================================================

#def read_2d_predictions( actions, data_dir ):
#  """
#  Loads 2d data from precomputed Stacked Hourglass detections
#
#  Args
#    actions: list of strings. Actions to load
#    data_dir: string. Directory where the data can be loaded from
#  Returns
#    train_set: dictionary with loaded 2d stacked hourglass detections for training
#    test_set: dictionary with loaded 2d stacked hourglass detections for testing
#    data_mean: vector with the mean of the 2d training data
#    data_std: vector with the standard deviation of the 2d training data
#    dim_to_ignore: list with the dimensions to not predict
#    dim_to_use: list with the dimensions to predict
#  """
#
#  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
#  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions)
#  
#  # anchor points
#  train_set = anchor( train_set, anchors, target_sets, dim = 2)
#  test_set = anchor( test_set, anchors, target_sets, dim = 2)
#
#  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
#  data_mean, data_std = normalization_stats( complete_train, anchors, dim=2 )
#
#  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
#  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )
#
#  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def create_xy_data( actions, data_dir, rcams, target_sets, anchors ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters.
  """

  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )
  
  #rotate to align with 2D
  train_set = project_to_cameras( train_set, rcams, cam_ids=[1] )
  test_set  = project_to_cameras( test_set, rcams, cam_ids=[1] )
  
  # anchor points
  train_set = anchor( train_set, anchors, target_sets, dim=2)
  test_set = anchor( test_set, anchors, target_sets, dim=2)

  # Compute normalization statistics.
  data_mean, data_std = normalization_stats( train_set, anchors, dim=2 )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=2 )

  return train_set, test_set, data_mean, data_std


def create_z_data( actions, data_dir, rcams, target_sets, anchors ):

  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )
  
  #rotate to align with 2D
  train_set = Z_coord( train_set)
  test_set  = Z_coord( test_set )
  
  # anchor points
  train_set = anchor( train_set, anchors, target_sets, dim = 1)
  test_set = anchor( test_set, anchors, target_sets, dim = 1)

  # Compute normalization statistics.
  data_mean, data_std = normalization_stats( train_set, anchors, dim=1 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=1 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=1 )

  return train_set, test_set, data_mean, data_std


def read_3d_data( actions, data_dir, camera_frame, rcams, target_sets, anchors):
  """
  Loads 3d poses, zero-centres and normalizes them

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    camera_frame: boolean. Whether to convert the data to camera coordinates
    rcams: dictionary with camera parameters

  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )

#  if camera_frame:
#  train_set = transform_world_to_camera( train_set, rcams )
#  test_set  = transform_world_to_camera( test_set, rcams )

  # anchor points
  train_set = anchor( train_set, anchors, target_sets, dim = 3)
  test_set = anchor( test_set, anchors, target_sets, dim = 3)

  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, anchors, dim=3 )

#   Divide every dimension independently
#  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=3 )
#  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=3 )

  return train_set, test_set, data_mean, data_std


# =============================================================================
# Load functions
# =============================================================================

def load_data( path, flies, actions ):
  """
  Loads 3d ground truth, and puts it in an easy-to-acess dictionary

  Args
    path: String. Path where to load the data from
    flies: List of integers. Flies whose data will be loaded
    actions: List of strings. The actions to load
  Returns:
    data: Dictionary with keys k=(subject, action)
  """

  path = os.path.join(path, '*')
  fnames = glob.glob( path )
  
  data = {}
  for fly in flies:
    for action in actions:
        
      fname = [file for file in fnames if ("Fly" + str(fly) + '_' in file) and (action in file)]    
      
      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        
        poses = pickle.load(open(fname_, "rb"))
        poses = poses['points3d']
        poses = np.reshape(poses, 
                          (poses.shape[0], poses.shape[1]*poses.shape[2]))
        data[ (fly, action, seqname) ] = poses

  return data


#def load_stacked_hourglass(data_dir, flies, actions, cam_ids = [1]):
#  """
#  Load 2d detections from disk, and put it in an easy-to-acess dictionary.
#
#  Args
#    data_dir: string. Directory where to load the data from,
#    subjects: list of integers. Subjects whose data will be loaded.
#    actions: list of strings. The actions to load.
#  Returns
#    data: dictionary with keys k=(subject, action, seqname)
#  """
#
#  path = os.path.join(data_dir, '*')
#  fnames = glob.glob( path )
#
#  data = {}
#  for fly in flies:
#    for action in actions:
#        
#      fname = [file for file in fnames if ("Fly"+ str(fly) + '_' in file) and (action in file)]   
#
#      for fname_ in fname:
#          
#        seqname = os.path.basename( fname_ )  
#        
#        poses = pickle.load(open(fname_, "rb"))
#        poses = poses['points2d']
#        
#        for cam in cam_ids:
#            seqname_cam = seqname[:-4] + '.cam_' + str(cam) + '-sh'
#        
#            poses_cam = poses[cam,:,:,:]
#            poses_cam = np.reshape(poses_cam, 
#                         (poses.shape[1], poses.shape[2]*poses.shape[3]))    
#        
#            data[ (fly, action, seqname_cam) ] = poses_cam
#
#  return data


# =============================================================================
# Define actions
# =============================================================================

def define_actions( action ):
  """
  List of actions.
  """
  actions = ["MDN_CsCh"]

  if action == "All" or action == "all":
    return actions

  return [action]


# =============================================================================
# Anchor data and normalise
# =============================================================================


def normalization_stats(train_set, anchors, dim ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={1,2,3} dimensionality of the data
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
  """

  complete_data = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)

  return data_mean, data_std

# =============================================================================
# Projection functions
# =============================================================================
#def transform_world_to_camera(poses_set, cams, cam_ids=[1] ):
#    """
#    Project 3d poses from world coordinate to camera coordinate system
#    Args
#      poses_set: dictionary with 3d poses
#      cams: dictionary with cameras
#      cam_ids: camera ids to consider
#    Return:
#      t3d_camera: dictionary with 3d poses in camera coordinate
#    """
#    t3d_camera = {}
#    for t3dk in sorted( poses_set.keys() ):
#
#      subj, action, seqname = t3dk
#      t3d_world = poses_set[ t3dk ]
#
#      for c in cam_ids:
#        R, T, intr, distort = cams[c]
#        camera_coord = np.reshape(t3d_world, [-1, 3]) #expand
#        camera_coord = world_to_camera_frame(camera_coord, R, T[:,None]) #transform
#        camera_coord = np.reshape( camera_coord, [-1, t3d_world.shape[1]] ) #compress again
#
#        sname = seqname[:-4] + ".cam_" + str(c)
#        t3d_camera[ (subj, action, sname) ] = camera_coord
#
#    return t3d_camera


def project_to_cameras( poses_set, cams, cam_ids=[1] ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    cam_ids: camera_ids to consider
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for c in cam_ids:
      R, T, intr, distort = cams[c]
      pts2d = np.reshape(t3d, [-1, 3])
      pts2d, _ = transform_and_project( pts2d, R, T[:,None])#, f, c, k, p )
      pts2d = np.reshape( pts2d, [-1, len(MARKER_NAMES)*2] )
      
      sname = seqname[:-4] + ".cam_" + str(c)
      t2d[ (subj, a, sname) ] = pts2d

  return t2d


def Z_coord( poses_set):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    cam_ids: camera_ids to consider
  Returns
    t2d: dictionary with 2d poses
  """
  t1d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    Z = np.reshape(t3d, [-1, 3])
    Z = Z[:,2]
    Z = np.reshape( Z, [-1, len(MARKER_NAMES)] )
      
    sname = seqname[:-4] + ".cam_" + str(1)
    t1d[ (subj, a, sname) ] = Z

  return t1d


def transform_and_project( P, R, T):
  """
  Project points from 3d to 2d using camera parameters
  including radial and tangential distortion

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
  Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
  """

  # P is a matrix of 3-dimensional points
  assert len(P.shape) == 2
  assert P.shape[1] == 3

#  N = P.shape[0]
#  print(P.shape)
#  print(R.shape)
  X = P.T#R.dot( P.T - T ) # rotate and translate
  
#  XX = X[:2,:] / X[2,:]
#  r2 = XX[0,:]**2 + XX[1,:]**2
#
#  radial = 1 + np.einsum( 'ij,ij->j', np.tile(k,(1, N)), np.array([r2, r2**2, r2**3]) );
#  tan = p[0]*XX[1,:] + p[1]*XX[0,:]
#
#  XXX = XX * np.tile(radial+tan,(2,1)) + np.outer(np.array([p[1], p[0]]).reshape(-1), r2 )
#
#  Proj = (f * XXX) + c
#  Proj = Proj.T
#  print(X.shape)
  Proj = X[:2,:].T
  D = X[2,]

  return Proj, D#, radial, tan, r2


#def world_to_camera_frame(P, R, T):
#  """
#  Convert points from world to camera coordinates
#
#  Args
#    P: Nx3 3d points in world coordinates
#    R: 3x3 Camera rotation matrix
#    T: 3x1 Camera translation parameters
#  Returns
#    X_cam: Nx3 3d points in camera coordinates
#  """
#
#  assert len(P.shape) == 2
#  assert P.shape[1] == 3
#
#  X_cam = R.dot( P.T - T ) # rotate and translate
#
#  return X_cam.T


#def camera_to_world_frame(P, R, T):
#  """Inverse of world_to_camera_frame
#
#  Args
#    P: Nx3 points in camera coordinates
#    R: 3x3 Camera rotation matrix
#    T: 3x1 Camera translation parameters
#  Returns
#    X_cam: Nx3 points in world coordinates
#  """
#
#  assert len(P.shape) == 2
#  assert P.shape[1] == 3
#
#  X_cam = R.T.dot( P.T ) + T # rotate and translate
#
#  return X_cam.T


if __name__ == "__main__":
    main()