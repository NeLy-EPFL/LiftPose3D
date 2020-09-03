import sys
import numpy as np
import torch
import src.utils as utils
import src.load as load
import src.stat as stat
import yaml

usr_input = sys.argv[-1]

#load global parameters
par = yaml.full_load(open(usr_input, "rb"))

def main():   
    
    test_set, mean, std, targets_1d, targets_2d = \
    create_xy_data(par)
    
    torch.save(test_set, par['data_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': mean, 'std': std, 
                'targets_1d': targets_1d, 'targets_2d': targets_2d,
                'input_size': len(targets_2d)},
                par['data_dir'] + '/stat_2d.pth.tar')

      
def create_xy_data( par ):
  """
  Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters.
  """

  # Load
  test_set, _, _ = load.load_3D( par['data_dir'], par, subjects=par['test_subjects'], actions=par['actions'] )
      
  # anchor points
  test_set, _ = utils.anchor( test_set, par['roots'], par['target_sets'], dim=2)    

  # Compute normalization statistics
  mean = torch.load(par['template_dir'] + 'stat_2d.pth.tar')['mean']
  std = torch.load(par['template_dir'] + 'stat_2d.pth.tar')['std']
  
  if 'template_coords' in par.keys():
      refs = np.array(par['template_coords'])
      refs = np.sort( np.hstack( (refs*2, refs*2+1)))
  else: 
      refs = None
  
  # Divide every dimension independently
  test_set = stat.normalize( test_set, mean[refs], std[refs] )
  
  #select coordinates to be predicted and return them as 'targets_3d'
  test_set, targets_2d = utils.collapse(test_set, par['target_sets'], 2)
  _, targets_1d = utils.collapse(test_set.copy(), par['target_sets'], 1)
  
  return test_set, mean, std, targets_1d, targets_2d


if __name__ == "__main__":
    main()