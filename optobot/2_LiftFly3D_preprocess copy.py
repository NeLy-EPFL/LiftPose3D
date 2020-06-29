import os
import numpy as np
import glob
import torch
import pickle
import src.utils as utils
import src.procrustes as procrustes

TEST_SUBJECTS  = [0]

data_dir = '/data/LiftFly3D/optobot/102906_s1a5_p6-0/'
template_dir = '/data/LiftFly3D/optobot/network/'
actions = ['off2']

#select cameras and joints visible from cameras
target_sets = [[ 1,  2,  3],  [5,  6,  7], [9, 10, 11],
               [13, 14, 15], [17, 18, 19], [21, 22, 23]]
ref_points = [0, 4, 8, 12, 16, 20]
scale = 28


def main():   
    
    try:
        os.remove(data_dir + '/test_2d.pth.tar')
        os.remove(data_dir + '/stat_2d.pth.tar')
    except:
        print("Did not delete the file as it didn't exists")
    
    test_set, data_mean, data_std, targets_1d, targets_2d = \
    create_xy_data( actions, data_dir, target_sets, ref_points )
    
    torch.save(test_set, data_dir + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_1d': targets_1d, 'targets_2d': targets_2d},
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
  test_set = load_data( data_dir, TEST_SUBJECTS,  actions, scale )
  
  #procrustes wrt prism data
#  template = torch.load(template_dir + 'template.pth.tar')
#  target = np.median(np.vstack(test_set.values()),axis=0)
#  template = template.reshape([-1, 2])
#  target = target.reshape([-1, 2])
#  template = template[:,:]
#  target = target[:,:]#

#  _, _, R, s, t = procrustes.compute_similarity_transform(template, target)
#  for key in test_set.keys():
#      tmp = test_set[key].reshape([-1, 2])
#      tmp  = (R@tmp.T).T
#      tmp += t
#      tmp *= s
#      test_set[key] = tmp.reshape([-1, 48])
      
  # anchor points
  test_set, _ = utils.anchor( test_set, ref_points, target_sets, dim=2)    

  # Compute normalization statistics
  data_mean, data_std = utils.normalization_stats( test_set)
  
  # Divide every dimension independently
  test_set = utils.normalize_data( test_set, data_mean, data_std )
  
  #select coordinates to be predicted and return them as 'targets_3d'
  test_set, targets_2d = utils.collapse(test_set, None, target_sets, 2)
  _, targets_1d = utils.collapse(test_set.copy(), None, target_sets, 1)

  return test_set, data_mean, data_std, targets_1d, targets_2d


# =============================================================================
# Load functions
# =============================================================================
def load_data( path, flies, actions, scale ):
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
        
        if scale is not None:
            poses2d /= scale 
        data[ (fly, action, seqname[:-4]) ] = poses2d #[:-4] is to get rid of .pkl extension

  return data


if __name__ == "__main__":
    main()