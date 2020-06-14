import os
import numpy as np
import glob
import copy
import torch
import pickle

TRAIN_SUBJECTS = [1,2,3]
#TRAIN_SUBJECTS = [1,2,3,4] #for optobot
TEST_SUBJECTS  = [4]

#data_dir = '/data/LiftFly3D/prism/data_oriented/'
data_dir = '/data/LiftFly3D/optobot/network/'
actions = ['PR']
rcams = []

#select cameras and joints visible from cameras
cam_ids = [0]
target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14],
               [16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]]
#target_sets = [[ 2,  3,  4],  [7,  8,  9], [12, 13, 14], #for optobot
#               [17, 18, 19], [22, 23, 24], [27, 28, 29]]
ref_points = [0, 5, 10,15, 20, 25]


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
# This part is for predicting z-coordinate of groundtruth from xy coords of groungtruth    
# =============================================================================
    
    #xy data
    train_set, test_set, data_mean, data_std, offset = \
    create_xy_data( actions, data_dir, target_sets, ref_points )

    torch.save(train_set, data_dir + '/train_2d.pth.tar')
    torch.save(test_set, data_dir + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'ref_points': ref_points, 'offset': offset},
                data_dir + '/stat_2d.pth.tar')
    
    #z data
    train_set, test_set, data_mean, data_std, offset, LR_train, LR_test = \
        create_z_data( actions, data_dir, target_sets, ref_points, rcams )
        
    torch.save([train_set, LR_train], data_dir + '/train_3d.pth.tar')
    torch.save([test_set, LR_test], data_dir + '/test_3d.pth.tar')   
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'ref_points': ref_points, 'offset': offset,
                'LR_train': LR_train, 'LR_test': LR_test},
                data_dir + '/stat_3d.pth.tar')
    
    
# =============================================================================
# Define actions
# =============================================================================    
def create_xy_data( actions, data_dir, target_sets, ref_points ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters.
  """

  # Load 3d data
  train_set, _ = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set, _  = load_data( data_dir, TEST_SUBJECTS,  actions )
  
  #rotate to align with 2D
  train_set = XY_coord( train_set )
  test_set  = XY_coord( test_set )
  
  # anchor points
  train_set, _ = anchor( train_set, ref_points, target_sets, dim=2)
  test_set, offset = anchor( test_set, ref_points, target_sets, dim=2)

  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, ref_points, dim=2 )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=2 )

  return train_set, test_set, data_mean, data_std, offset


def create_z_data( actions, data_dir, target_sets, ref_points, rcams ):

  # Load 3d data
  train_set, LR_train = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set, LR_test  = load_data( data_dir, TEST_SUBJECTS,  actions )
  
  #rotate to align with 2D
  train_set = Z_coord( train_set)
  test_set  = Z_coord( test_set )
  
  # anchor points
  train_set, _ = anchor( train_set, ref_points, target_sets, dim = 1)
  test_set, offset = anchor( test_set, ref_points, target_sets, dim = 1)

  # Compute normalization statistics.
  data_mean, data_std = normalization_stats( train_set, ref_points, dim=1 )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=1 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=1 )
  
  return train_set, test_set, data_mean, data_std, offset, LR_train, LR_test


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
  LR = {}
  for fly in flies:
    for action in actions:
        
      fname = [file for file in fnames if ("00" + str(fly) + '_' in file) and (action in file)]    
      
      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        
        poses = pickle.load(open(fname_, "rb"))
        poses3d = poses['points3d']
        poses3d = np.reshape(poses3d, 
                          (poses3d.shape[0], poses3d.shape[1]*poses3d.shape[2]))
        
        data[ (fly, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension
        LR[ (fly, action, seqname[:-4]) ] = poses['flip_idx']

  return data, LR


# =============================================================================
# Projection functions
# =============================================================================
def XY_coord( poses_set):
  """
  Project 3d poses to XY coord
  """
  t2d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    XY = np.reshape(t3d, [-1, 3])
    XY = XY[:,:2]
    t2d[ (subj, a, seqname) ] = np.reshape( XY, [-1, len(MARKER_NAMES)*2] )
 
  return t2d


def Z_coord( poses_set):
  """
  Project 3d poses to Z coord
  """
  t1d = {}

  for t3dk in sorted( poses_set.keys() ):
    subj, a, seqname = t3dk
    t3d = poses_set[ t3dk ]

    Z = np.reshape(t3d, [-1, 3])
    Z = Z[:,2]
    t1d[ (subj, a, seqname) ] = np.reshape( Z, [-1, len(MARKER_NAMES)] )

  return t1d


# =============================================================================
# Collect statistics for later use
# =============================================================================
def normalization_stats(train_set, ref_points, dim ):
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
  
    return dim_to_use


def anchor(poses, anchors, target_sets, dim):
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


if __name__ == "__main__":
    main()