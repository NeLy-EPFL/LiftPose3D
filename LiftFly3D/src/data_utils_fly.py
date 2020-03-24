"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
import src.cameras as cameras
import glob
import copy
import torch

import pickle

TRAIN_SUBJECTS = [0,1,2,3,4,5,6,7]
TEST_SUBJECTS  = [8,9]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
MARKER_NAMES = ['']*38
MARKER_NAMES[0] = 'BODY_COXA',
MARKER_NAMES[1] = 'COXA_FEMUR',
MARKER_NAMES[2] = 'FEMUR_TIBIA',
MARKER_NAMES[3] = 'TIBIA_TARSUS',
MARKER_NAMES[4] = 'TARSUS_TIP',
MARKER_NAMES[5] = 'BODY_COXA',
MARKER_NAMES[6] = 'COXA_FEMUR',
MARKER_NAMES[7] = 'FEMUR_TIBIA',
MARKER_NAMES[8] = 'TIBIA_TARSUS',
MARKER_NAMES[9] = 'TARSUS_TIP',
MARKER_NAMES[10] = 'BODY_COXA',
MARKER_NAMES[11] = 'COXA_FEMUR',
MARKER_NAMES[12] = 'FEMUR_TIBIA',
MARKER_NAMES[13] = 'TIBIA_TARSUS',
MARKER_NAMES[14] = 'TARSUS_TIP',
MARKER_NAMES[15] = 'ANTENNA',
MARKER_NAMES[16] = 'STRIPE',
MARKER_NAMES[17] = 'STRIPE',
MARKER_NAMES[18] = 'STRIPE',
MARKER_NAMES[19] = 'BODY_COXA',
MARKER_NAMES[20] = 'COXA_FEMUR',
MARKER_NAMES[21] = 'FEMUR_TIBIA',
MARKER_NAMES[22] = 'TIBIA_TARSUS',
MARKER_NAMES[23] = 'TARSUS_TIP',
MARKER_NAMES[24] = 'BODY_COXA',
MARKER_NAMES[25] = 'COXA_FEMUR',
MARKER_NAMES[26] = 'FEMUR_TIBIA',
MARKER_NAMES[27] = 'TIBIA_TARSUS',
MARKER_NAMES[28] = 'TARSUS_TIP',
MARKER_NAMES[29] = 'BODY_COXA',
MARKER_NAMES[30] = 'COXA_FEMUR',
MARKER_NAMES[31] = 'FEMUR_TIBIA',
MARKER_NAMES[32] = 'TIBIA_TARSUS',
MARKER_NAMES[33] = 'TARSUS_TIP',
MARKER_NAMES[34] = 'ANTENNA',
MARKER_NAMES[35] = 'STRIPE',
MARKER_NAMES[36] = 'STRIPE',
MARKER_NAMES[37] = 'STRIPE',

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*38
SH_NAMES[0] = 'BODY_COXA',
SH_NAMES[1] = 'COXA_FEMUR',
SH_NAMES[2] = 'FEMUR_TIBIA',
SH_NAMES[3] = 'TIBIA_TARSUS',
SH_NAMES[4] = 'TARSUS_TIP',
SH_NAMES[5] = 'BODY_COXA',
SH_NAMES[6] = 'COXA_FEMUR',
SH_NAMES[7] = 'FEMUR_TIBIA',
SH_NAMES[8] = 'TIBIA_TARSUS',
SH_NAMES[9] = 'TARSUS_TIP',
SH_NAMES[10] = 'BODY_COXA',
SH_NAMES[11] = 'COXA_FEMUR',
SH_NAMES[12] = 'FEMUR_TIBIA',
SH_NAMES[13] = 'TIBIA_TARSUS',
SH_NAMES[14] = 'TARSUS_TIP',
SH_NAMES[15] = 'ANTENNA',
SH_NAMES[16] = 'STRIPE',
SH_NAMES[17] = 'STRIPE',
SH_NAMES[18] = 'STRIPE',
SH_NAMES[19] = 'BODY_COXA',
SH_NAMES[20] = 'COXA_FEMUR',
SH_NAMES[21] = 'FEMUR_TIBIA',
SH_NAMES[22] = 'TIBIA_TARSUS',
SH_NAMES[23] = 'TARSUS_TIP',
SH_NAMES[24] = 'BODY_COXA',
SH_NAMES[25] = 'COXA_FEMUR',
SH_NAMES[26] = 'FEMUR_TIBIA',
SH_NAMES[27] = 'TIBIA_TARSUS',
SH_NAMES[28] = 'TARSUS_TIP',
SH_NAMES[29] = 'BODY_COXA',
SH_NAMES[30] = 'COXA_FEMUR',
SH_NAMES[31] = 'FEMUR_TIBIA',
SH_NAMES[32] = 'TIBIA_TARSUS',
SH_NAMES[33] = 'TARSUS_TIP',
SH_NAMES[34] = 'ANTENNA',
SH_NAMES[35] = 'STRIPE',
SH_NAMES[36] = 'STRIPE',
SH_NAMES[37] = 'STRIPE',


def main():
    
    data_dir = '/data/DF3D/'
    actions = define_actions('all')
    rcams = pickle.load(open('cameras.pkl', "rb"))
    camera_frame = True #boolean. Whether to convert the data to camera coordinates
    
    # HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = \
    read_2d_predictions( actions, data_dir )
    
    torch.save(train_set, '/data/DF3D/train_2d_ft.pth.tar')
    torch.save(test_set, '/data/DF3D/test_2d_ft.pth.tar')
    
    #3D ground truth
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = \
    create_2d_data( actions, data_dir, rcams )
    
    torch.save(train_set, '/data/DF3D/train_2d.pth.tar')
    torch.save(test_set, '/data/DF3D/test_2d.pth.tar')
    
    #3D ground truth
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, \
    train_root_positions, test_root_positions = \
    read_3d_data( actions, data_dir, camera_frame, rcams)
    
    stat_3d = {'mean': data_mean, 'std': data_std, 'dim_use': dim_to_use, 'dim_ignore': dim_to_ignore}
    torch.save(stat_3d, '/data/DF3D/stat_3d.pth.tar')
    torch.save(train_set, '/data/DF3D/train_3d.pth.tar')
    torch.save(test_set, '/data/DF3D/test_3d.pth.tar')


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


def load_stacked_hourglass(data_dir, flies, actions):
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
        
        for cam in range(poses.shape[0]):
            seqname_cam = seqname[:-4] + '.cam_' + str(cam) + '-sh'
        
            poses_cam = poses[cam,:,:,:]
            poses_cam = np.reshape(poses_cam, 
                         (poses.shape[1], poses.shape[2]*poses.shape[3]))    
        
            data[ (fly, action, seqname_cam) ] = poses_cam

  return data


def transform_world_to_camera(poses_set, cams, ncams=1 ):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted( poses_set.keys() ):

      subj, action, seqname = t3dk
      t3d_world = poses_set[ t3dk ]

      for c in range( ncams ):
        R, T, intr, distort = cams[c]
        camera_coord = np.reshape(t3d_world, [-1, 3]) #expand
        camera_coord = cameras.world_to_camera_frame(camera_coord, R, T[:,None]) #transform
        camera_coord = np.reshape( camera_coord, [-1, t3d_world.shape[1]] ) #compress again

        sname = seqname[:-4] + ".cam_" + str(c)
        t3d_camera[ (subj, action, sname) ] = camera_coord

    return t3d_camera


def project_to_cameras( poses_set, cams, ncams=1 ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of cameras per subject
  Returns
    t2d: dictionary with 2d poses
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    for c in range( ncams ):
      R, T, intr, distort = cams[c]
      pts2d = np.reshape(t3d, [-1, 3])
      pts2d, _ = cameras.project_point_radial( pts2d, R, T[:,None])#, f, c, k, p )

      pts2d = np.reshape( pts2d, [-1, len(MARKER_NAMES)*2] )
      sname = seqname[:-4] + ".cam_" + str(c)
      t2d[ (subj, a, sname) ] = pts2d

  return t2d


def postprocess_3d( poses_set ):
  """
  Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
  root_positions = {}
  for k in poses_set.keys():
    # Keep track of the global position
    root_positions[k] = copy.deepcopy(poses_set[k][:,:3])

    # Remove the root from the 3d position
    poses_set[k] = poses_set[k] - np.tile( poses_set[k][:,:3], [1, len(MARKER_NAMES)] )

  return poses_set, root_positions


def normalization_stats(complete_data, dim ):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
  if not dim in [2,3]:
    raise(ValueError, 'dim must be 2 or 3')

  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)

  dimensions_to_ignore = []
  if dim == 2:
    dimensions_to_use    = np.where(np.array([x != '' for x in MARKER_NAMES]))[0]
    dimensions_to_use    = np.sort( np.hstack( (dimensions_to_use*2, 
                                                dimensions_to_use*2+1)))
    dimensions_to_ignore = np.delete( np.arange(len(MARKER_NAMES)*2), dimensions_to_use )
  
  else: # dim == 3
    dimensions_to_use = np.where(np.array([x != '' for x in MARKER_NAMES]))[0]
    dimensions_to_use = np.delete( dimensions_to_use, 0 ) #remove anchor point
    dimensions_to_use = np.sort( np.hstack( (dimensions_to_use*3,
                                             dimensions_to_use*3+1,
                                             dimensions_to_use*3+2)))
    dimensions_to_ignore = np.delete( np.arange(len(MARKER_NAMES)*3), dimensions_to_use )

  return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def normalize_data(data, data_mean, data_std, dim_to_use ):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
  data_out = {}

  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]
    mu = data_mean[dim_to_use]
    stddev = data_std[dim_to_use]
    data_out[ key ] = np.divide( (data[key] - mu), stddev )

  return data_out


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
  T = normalized_data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality

  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:, dimensions_to_use] = normalized_data

  # Multiply times stdev and add the mean
  stdMat = data_std.reshape((1, D))
  stdMat = np.repeat(stdMat, T, axis=0)
  meanMat = data_mean.reshape((1, D))
  meanMat = np.repeat(meanMat, T, axis=0)
  orig_data = np.multiply(orig_data, stdMat) + meanMat
  return orig_data


def define_actions( action ):
  """
  Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the fly behaviours
  Returns
    actions: List of strings. Actions to use.
  """
  actions = ["MDN_CsCh"]

  if action == "All" or action == "all":
    return actions

  return [action]


# =============================================================================
# Below are the main load functions
# =============================================================================

def read_2d_predictions( actions, data_dir ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
  Returns
    train_set: dictionary with loaded 2d stacked hourglass detections for training
    test_set: dictionary with loaded 2d stacked hourglass detections for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions)
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions)

  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std,  dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def create_2d_data( actions, data_dir, rcams ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters. Also normalizes the 2d poses

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with projected 2d poses for training
    test_set: dictionary with projected 2d poses for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

  # Load 3d data
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )

  train_set = project_to_cameras( train_set, rcams, ncams=1 )
  test_set  = project_to_cameras( test_set, rcams, ncams=1 )

  # Compute normalization statistics.
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=2 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def read_3d_data( actions, data_dir, camera_frame, rcams):
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

  if camera_frame:
    train_set = transform_world_to_camera( train_set, rcams )
    test_set  = transform_world_to_camera( test_set, rcams )

  # Apply 3d post-processing (centering around root)
  train_set, train_root_positions = postprocess_3d( train_set )
  test_set,  test_root_positions  = postprocess_3d( test_set )

  # Compute normalization statistics
  complete_train = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats( complete_train, dim=3 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, dim_to_use )
  test_set  = normalize_data( test_set,  data_mean, data_std, dim_to_use )

  return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


if __name__ == "__main__":
    main()