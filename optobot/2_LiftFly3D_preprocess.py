import os
import numpy as np
import glob
import copy
import torch
import pickle

TEST_SUBJECTS  = [1]

data_dir = '/data/LiftFly3D/optobot/'
actions = ['pre']

#select cameras and joints visible from cameras
target_sets = [[ 1,  2,  3],  [5,  6,  7], [9, 10, 11],
               [13, 14, 15], [17, 18, 19], [21, 22, 23]]
ref_points = [0, 4, 8, 12, 16, 20]


def main():   
    
    test_set, data_mean, data_std, offset = \
    create_xy_data( actions, data_dir, target_sets, ref_points )

    torch.save(test_set, data_dir + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'ref_points': ref_points, 'offset': offset},
                data_dir + '/stat_2d.pth.tar')

    
# =============================================================================
# Define actions
# =============================================================================    
def create_xy_data( actions, data_dir, target_sets, ref_points ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters.
  """

  # Load 3d data
  test_set = load_data( data_dir, TEST_SUBJECTS,  actions )
  
  # anchor points
  test_set, offset = anchor( test_set, ref_points, target_sets, dim=2)

  # Compute normalization statistics
  data_mean, data_std = normalization_stats( test_set, ref_points, dim=2 )
  
  # Divide every dimension independently
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=2  )

  return test_set, data_mean, data_std, offset


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
        
      fname = [file for file in fnames if ("fly" + str(fly) + '_' in file and '.pkl' in file) and (action in file)]    
      
      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        print(fname)
        poses = pickle.load(open(fname_, "rb"))
        poses2d = poses['points2d']
        poses2d = np.reshape(poses2d, 
                          (poses2d.shape[0], poses2d.shape[1]*poses2d.shape[2]))
        data[ (fly, action, seqname[:-4]) ] = poses2d #[:-4] is to get rid of .pkl extension

  return data


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