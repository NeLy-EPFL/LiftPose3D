"""Utility functions for dealing with DeepFly3D data."""

from __future__ import division

import os
import numpy as np
import glob
import torch
import pickle
import copy


TRAIN_SUBJECTS = [1,2,3]
TEST_SUBJECTS  = [4]

data_dir = '/data/LiftFly3D/prism/'
actions = ['PR']
rcams = []

#select cameras and joints visible from cameras
#cam_ids = [0] #L side
#target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14]]
#ref_points = [0, 5, 10]

cam_ids = [1] #R side
target_sets = [[16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]]
ref_points = [15, 20, 25]


MARKER_NAMES = ['']*30
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

MARKER_NAMES[15] = 'BODY_COXA'
MARKER_NAMES[16] = 'COXA_FEMUR'
MARKER_NAMES[17] = 'FEMUR_TIBIA'
MARKER_NAMES[18] = 'TIBIA_TARSUS'
MARKER_NAMES[19] = 'TARSUS_TIP'
MARKER_NAMES[20] = 'BODY_COXA'
MARKER_NAMES[21] = 'COXA_FEMUR'
MARKER_NAMES[22] = 'FEMUR_TIBIA'
MARKER_NAMES[23] = 'TIBIA_TARSUS'
MARKER_NAMES[24] = 'TARSUS_TIP'
MARKER_NAMES[25] = 'BODY_COXA'
MARKER_NAMES[26] = 'COXA_FEMUR'
MARKER_NAMES[27] = 'FEMUR_TIBIA'
MARKER_NAMES[28] = 'TIBIA_TARSUS'
MARKER_NAMES[29] = 'TARSUS_TIP'


def main():   
    
# =============================================================================
# This part is for predicting xyz of groundtruth from SH predictions
# =============================================================================
    
    print('processing for camera' + str(cam_ids[0]))
    # HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std, offset = \
        read_2d_predictions( actions, data_dir, target_sets, ref_points )
    
    torch.save(train_set, data_dir + 'train_2d_ft.pth.tar')
    torch.save(test_set, data_dir + 'test_2d_ft.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'offset': offset},
                data_dir + 'stat_2d.pth.tar')
    
    #3D ground truth
    train_set, test_set, data_mean, data_std, offset = \
        read_3d_data( actions, data_dir, target_sets, ref_points, rcams)
    
    torch.save(train_set, data_dir + 'train_3d.pth.tar')
    torch.save(test_set, data_dir + 'test_3d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'offset': offset},
                data_dir + 'stat_3d.pth.tar')

# =============================================================================
# Preprocess pipelines
# =============================================================================
    
def read_3d_data( actions, data_dir, target_sets, ref_points, rcams=None):
  """
  Loads 3d poses, zero-centres and normalizes them
  """
  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )

  if rcams is not None:
      train_set = transform_world_to_camera( train_set, rcams, cam_ids )
      test_set  = transform_world_to_camera( test_set, rcams, cam_ids )

  # anchor points
  train_set, offset = anchor( train_set, ref_points, target_sets, dim=3)
  test_set, offset = anchor( test_set, ref_points, target_sets, dim=3)

  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, ref_points, dim=3 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=3 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=3 )

  return train_set, test_set, data_mean, data_std, offset


def read_2d_predictions( actions, data_dir, target_sets, ref_points ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions, cam_ids)
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions, cam_ids)
  
  if cam_ids==1:
      test_set = reflect(test_set)

  # anchor points
  train_set, offset = anchor( train_set, ref_points, target_sets, dim=2)
  test_set, offset = anchor( test_set, ref_points, target_sets, dim=2)
  
  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, ref_points, dim=2 )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=2 )
  
  return train_set, test_set, data_mean, data_std, offset


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

  path = os.path.join(path, '*.pkl')
  fnames = glob.glob( path )
  
  data = {}
  for fly in flies:
    for action in actions:
        
      fname = [file for file in fnames if ("00" + str(fly) + '_' in file) and (action in file)]   
      
      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        
        poses = pickle.load(open(fname_, "rb"))
        poses = poses['points3d']
        poses = np.reshape(poses, 
                          (poses.shape[0], poses.shape[1]*poses.shape[2]))
        data[ (fly, action, seqname[:-4]) ] = poses #[:-4] is to get rid of .pkl extension

  return data


def load_stacked_hourglass(data_dir, flies, actions, cam_ids):
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
        
      fname = [file for file in fnames if ("00"+ str(fly) + '_' in file) and (action in file)]   

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
# Normalize
# =============================================================================
def get_coords_in_dim(targets, dim):
    
    if len(targets)>1:
      dim_to_use = []
      for i in targets:
          dim_to_use += i
    else:
      dim_to_use = targets
  
    dim_to_use = np.array(dim_to_use)
    if dim == 2:    
      dim_to_use = np.sort( np.hstack( (dim_to_use*2, 
                                        dim_to_use*2+1)))
  
    elif dim == 3:
      dim_to_use = np.sort( np.hstack( (dim_to_use*3,
                                        dim_to_use*3+1,
                                        dim_to_use*3+2)))
    return dim_to_use

  
def normalize_data(data, data_mean, data_std, targets, dim ):
  """
  Normalizes a dictionary of poses
  """

  dim_to_use = get_coords_in_dim(targets, dim)
 
  for key in data.keys():
    data[ key ] -= data_mean
    data[ key ] /= data_std
    data[ key ] = data[ key ][ :, dim_to_use ]  

  return data


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


def anchor(poses, anchors, target_sets, dim=3):
  """
  Center points in targset sets around anchors
  """
  
  offset = {}
  for k in poses.keys():
      offset[k] = np.zeros_like(poses[k])
      for i, anch in enumerate(anchors):
          for j in [anch]+target_sets[i]:
              offset[k][:, dim*j:dim*(j+1)] += poses[k][:, dim*anch:dim*(anch+1)]

  for k in poses.keys():
      poses[k] -= offset[k]
      
  return poses, offset


# =============================================================================
# Projection functions
# =============================================================================

def transform_world_to_camera( poses_set, cams, cam_ids, project=False ):
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
#      R, T, intr, distort = cams[c]
#      P = transform(P, R, T)
#      
#      if project:
#         P = project(P, intr)
      
      Ptransf[ (subj, a, seqname + ".cam_" + str(c)) ] = P

  return Ptransf


def reflect(poses):
    
    for k in poses.keys():
        tmp = poses[k]
        tmp = np.reshape(tmp, [-1, 2])
        tmp[:,1] = -1*tmp[:,1]
        tmp = np.reshape( tmp, [-1, 60] )
        poses[k] = tmp
    
    return poses
#
#
#def transform_camera_to_world( data, cams, cam_ids ):
#  """
#  Project 3d poses using camera parameters
#
#  Args
#    poses_set: dictionary with 3d poses
#    cams: dictionary with camera parameters
#    cam_ids: camera_ids to consider
#  Returns
#    transf: dictionary with 3d poses or 2d poses if projection is True
#  """
#
#  for c in cam_ids:
#    R, T, _, _ = cams[c]
#    
#    P = np.reshape(data, [-1, 3])
#    P -= T
#    P = np.squeeze(np.matmul(np.inv(R),P[:,:,np.newaxis]))
#  
#  return np.reshape( P, [-1, len(MARKER_NAMES)*3] )
#
#
#
#def transform(P, R, T):
#  """
#  Rotate/translate 3d poses to camera viewpoint
#
#  Args
#    P: Nx3 points in world coordinates
#    R: 3x3 Camera rotation matrix
#    T: 3x1 Camera translation parameters
#  Returns
#    transf: Nx2 points on camera
#  """
#
#  P = np.reshape(P, [-1, 3])
#  
#  assert len(P.shape) == 2
#  assert P.shape[1] == 3
#  
#  points3d_new_cs =  np.matmul(R, P.T).T + T
#  
#  return np.reshape( points3d_new_cs, [-1, len(MARKER_NAMES)*3] )
#
#
#def project(P, intr):
#    
#  P = np.reshape(P, [-1, 3])  
#  proj = np.squeeze(np.matmul(intr, P[:,:,np.newaxis]))
#  proj = proj / proj[:, [2]]
#  proj = proj[:, :2]
#  
#  return np.reshape( proj, [-1, len(MARKER_NAMES)*2] )


if __name__ == "__main__":
    main()