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

anchors = [0, 5, 10]
target_sets = [[ 1,  2,  3,  4], [ 6,  7,  8,  9], [11, 12, 13, 14]]

#anchors = [0, 5, 10, 19, 24, 29]
#target_sets = [[ 1,  2,  3,  4], [ 6,  7,  8,  9], [11, 12, 13, 14],
#               [20, 21, 22, 23], [25, 26, 27, 28], [30, 31, 32, 33]]

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

#MARKER_NAMES[19] = 'BODY_COXA'
#MARKER_NAMES[20] = 'COXA_FEMUR'
#MARKER_NAMES[21] = 'FEMUR_TIBIA'
#MARKER_NAMES[22] = 'TIBIA_TARSUS'
#MARKER_NAMES[23] = 'TARSUS_TIP'
#MARKER_NAMES[24] = 'BODY_COXA'
#MARKER_NAMES[25] = 'COXA_FEMUR'
#MARKER_NAMES[26] = 'FEMUR_TIBIA'
#MARKER_NAMES[27] = 'TIBIA_TARSUS'
#MARKER_NAMES[28] = 'TARSUS_TIP'
#MARKER_NAMES[29] = 'BODY_COXA'
#MARKER_NAMES[30] = 'COXA_FEMUR'
#MARKER_NAMES[31] = 'FEMUR_TIBIA'
#MARKER_NAMES[32] = 'TIBIA_TARSUS'
#MARKER_NAMES[33] = 'TARSUS_TIP'

data_dir = '/data/DF3D/'#'/Users/adamgosztolai/Dropbox/'#
actions = ['MDN_CsCh']
rcams = pickle.load(open('cameras.pkl', "rb"))


def main():   
    
# =============================================================================
# This part is for predicting xyz of groundtruth from SH predictions
# =============================================================================
    
    # HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std = \
        read_2d_predictions( actions, data_dir, rcams, target_sets, anchors )
    
    torch.save(train_set, '/data/DF3D/train_2d_ft.pth.tar')
    torch.save(test_set, '/data/DF3D/test_2d_ft.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'anchors': anchors},
                data_dir + 'stat_2d.pth.tar')
    
    #3D ground truth
    train_set, test_set, data_mean, data_std = \
        read_3d_data( actions, data_dir, target_sets, anchors, rcams)
    
    torch.save(train_set, data_dir + 'train_3d.pth.tar')
    torch.save(test_set, data_dir + 'test_3d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'anchors': anchors},
                data_dir + 'stat_3d.pth.tar')


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
# Preprocess pipelines
# =============================================================================
    
def read_3d_data( actions, data_dir, target_sets, anchors, rcams=None):
  """
  Loads 3d poses, zero-centres and normalizes them
  """
  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )

  if rcams is not None:
      train_set = transform_world_to_camera( train_set, rcams, cam_ids=[1] )
      test_set  = transform_world_to_camera( test_set, rcams, cam_ids=[1] )

  # anchor points
  train_set = anchor( train_set, anchors, target_sets, dim=3)
  test_set = anchor( test_set, anchors, target_sets, dim=3)

  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, anchors, dim=3 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=3 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=3 )

  return train_set, test_set, data_mean, data_std


def read_2d_predictions( actions, data_dir, rcams, target_sets, anchors ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions, cam_ids = [1])
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions, cam_ids = [1])

  # anchor points
  train_set = anchor( train_set, anchors, target_sets, dim=2)
  test_set = anchor( test_set, anchors, target_sets, dim=2)
  
  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, anchors, dim=2 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=2 )
  
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
        data[ (fly, action, seqname[:-4]) ] = poses #[:-4] is to get rid of .pkl extension

  return data


def load_stacked_hourglass(data_dir, flies, actions, cam_ids = [1]):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Directory where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
    actions: list of strings. The actions to load.
  Returns
    data: dictionary with keys k=(subject, action, seqname)
  """

  path = os.path.join(data_dir, '*')
  fnames = glob.glob( path )

  data = {}
  for fly in flies:
    for action in actions:
        
      fname = [file for file in fnames if ("Fly"+ str(fly) + '_' in file) and (action in file)]   

      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        
        poses = pickle.load(open(fname_, "rb"))
        poses = poses['points2d']
        
        for cam in cam_ids:

            poses_cam = poses[cam,:,:,:]
            poses_cam = np.reshape(poses_cam, 
                         (poses.shape[1], poses.shape[2]*poses.shape[3]))    
        
            data[ (fly, action, seqname[:-4] + '.cam_' + str(cam) + '-sh') ] = poses_cam

  return data
    

# =============================================================================
# Projection functions
# =============================================================================

def transform_world_to_camera( poses_set, cams, cam_ids=[1], project=False ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    cam_ids: camera_ids to consider
  Returns
    transf: dictionary with 3d poses or 2d poses if projection is True
  """
  Ptransf = {}

  for subj, a, seqname in sorted( poses_set.keys() ):

    P = poses_set[ (subj, a, seqname) ]

    for c in cam_ids:
      R, T, intr, distort = cams[c]
      P = transform( P, R, T)
      
      if project:
         P = project(P, intr)
      
      Ptransf[ (subj, a, seqname + ".cam_" + str(c)) ] = P

  return Ptransf


def transform( P, R, T):
  """
  Transform 3d poses to camera viewpoint

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    transf: Nx2 points on camera
  """

  P = np.reshape(P, [-1, 3])
  
  assert len(P.shape) == 2
  assert P.shape[1] == 3
  
  points3d_new_cs =  np.squeeze(np.matmul(R, P[:,:,np.newaxis])) + T
  
  return np.reshape( points3d_new_cs, [-1, len(MARKER_NAMES)*3] )


def project(P, intr):
    
  P = np.reshape(P, [-1, 3])  
  proj = np.squeeze(np.matmul(intr, P[:,:,np.newaxis]))
  proj = proj / proj[:, [2]]
  proj = proj[:, :2]
  
  return np.reshape( proj, [-1, len(MARKER_NAMES)*2] )


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


if __name__ == "__main__":
    main()