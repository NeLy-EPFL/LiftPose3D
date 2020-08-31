import os
import sys
import numpy as np
import glob
import torch
import pickle
import src.utils as utils
import yaml

usr_input = sys.argv[-1]

#load global parameters
par = yaml.full_load(open(usr_input, "rb"))

def main():   
    
    test_set, data_mean, data_std, targets_1d, targets_2d = \
    create_xy_data(par['actions'], par['data_dir'], par['target_sets'], par['roots'] )
    
    torch.save(test_set, par['data_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_1d': targets_1d, 'targets_2d': targets_2d},
                par['data_dir'] + '/stat_2d.pth.tar')

      
def create_xy_data( actions, data_dir, target_sets, roots ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters.
  """

  # Load 3d data
  test_set = load_data( data_dir, par['test_subjects'], actions )
      
  # anchor points
  test_set, _ = utils.anchor( test_set, roots, target_sets, dim=2)    

  # Compute normalization statistics
  data_mean = torch.load(par['template_dir'] + 'stat_2d.pth.tar')['mean']
  data_std = torch.load(par['template_dir'] + 'stat_2d.pth.tar')['std']
  
  if 'template_coords' in par.keys():
      refs = np.array(par['template_coords'])
      refs = np.sort( np.hstack( (refs*2, refs*2+1)))
  else: 
      refs = None
  
  # Divide every dimension independently
  test_set = utils.normalize_data( test_set, data_mean[refs], data_std[refs] )
  
  #select coordinates to be predicted and return them as 'targets_3d'
  test_set, targets_2d = utils.collapse(test_set, None, target_sets, 2)
  _, targets_1d = utils.collapse(test_set.copy(), None, target_sets, 1)
  
  return test_set, data_mean, data_std, targets_1d, targets_2d


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
        
      fname = fnames.copy()
        
      if fly!='all':
          fname = [file for file in fname if "Fly"+ str(fly) in file]   
                
      if action!='all':
          fname = [file for file in fname if action in file] 
                
      assert len(fname)!=0, 'No files found. Check path!'    
         
      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        print(fname)
        poses = pickle.load(open(fname_, "rb"))
        poses2d = poses['points2d']
        poses2d = np.reshape(poses2d, 
                          (poses2d.shape[0], poses2d.shape[1]*poses2d.shape[2]))
        
        data[ (fly, action, seqname[:-4]) ] = poses2d #[:-4] is to get rid of .pkl extension

  return data


if __name__ == "__main__":
    main()