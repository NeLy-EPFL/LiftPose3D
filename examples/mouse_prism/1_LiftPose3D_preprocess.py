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

    print('behaviors' + str(par['actions']))
        
    #xy data
    train_set, test_set, data_mean, data_std, targets_2d, offset_2d = \
    create_xy_data( par['actions'], par['data_dir'], par['target_sets'], par['roots'] )
    
    torch.save(train_set, par['data_dir'] + '/train_2d.pth.tar')
    torch.save(test_set, par['data_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_2d': targets_2d, 'offset': offset_2d},
                par['data_dir'] + '/stat_2d.pth.tar')
    
    #z data
    train_set, test_set, data_mean, data_std, train_keypts, test_keypts, targets_1d, offset_1d = \
        create_z_data( par['actions'], par['data_dir'], par['target_sets'], par['roots'] )
        
    torch.save([train_set, train_keypts], par['data_dir'] + '/train_3d.pth.tar')
    torch.save([test_set, test_keypts], par['data_dir'] + '/test_3d.pth.tar')   
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_1d': targets_1d, 'offset': offset_1d,
                'LR_train': train_keypts, 'LR_test': test_keypts},
                par['data_dir'] + '/stat_3d.pth.tar')
    
      
def create_xy_data( actions, data_dir, target_sets, roots ):
    """
    Creates 2d poses by projecting 3d poses with the corresponding camera
    parameters.
    """

    # Load 3d data
    train_set, _ = load_data( data_dir, par['train_subjects'], actions )
    test_set, _  = load_data( data_dir, par['test_subjects'],  actions )
  
    #project data to ventral view
    train_set = utils.XY_coord( train_set )
    test_set  = utils.XY_coord( test_set )

    # anchor points
    train_set, _ = utils.anchor( train_set, roots, target_sets, par['in_dim'])
    test_set, offset = utils.anchor( test_set, roots, target_sets, par['in_dim'])
    
    # Divide every dimension independently
    data_mean, data_std = utils.normalization_stats( train_set)
    train_set = utils.normalize_data( train_set, data_mean, data_std )
    test_set  = utils.normalize_data( test_set,  data_mean, data_std )
    
    #select coordinates to be predicted and return them as 'targets_3d'
    train_set, _ = utils.collapse(train_set, None, target_sets, par['in_dim'])
    test_set, targets_2d = utils.collapse(test_set, None, target_sets, par['in_dim'])
    
    return train_set, test_set, data_mean, data_std, targets_2d, offset


def create_z_data( actions, data_dir, target_sets, roots ):

    # Load 3d data
    train_set, train_keypts = load_data( data_dir, par['train_subjects'], actions )
    test_set, test_keypts  = load_data( data_dir, par['test_subjects'],  actions )
  
    #rotate to align with 2D
    train_set = utils.Z_coord( train_set)
    test_set  = utils.Z_coord( test_set )
  
    # anchor points
    train_set, _ = utils.anchor( train_set, roots, target_sets, par['out_dim'])
    test_set, offset = utils.anchor( test_set, roots, target_sets, par['out_dim'])

    # Divide every dimension independently
    data_mean, data_std = utils.normalization_stats( train_set)
    train_set = utils.normalize_data( train_set, data_mean, data_std )
    test_set  = utils.normalize_data( test_set,  data_mean, data_std )
  
    #select coordinates to be predicted and return them as 'targets_1d'
    train_set, _ = utils.collapse(train_set, None, target_sets, par['out_dim'])
    test_set, targets_1d = utils.collapse(test_set, None, target_sets, par['out_dim'])
    
    for key in train_keypts.keys():
        train_keypts[key] = train_keypts[key][:,targets_1d]
    for key in test_keypts.keys():
        test_keypts[key] = test_keypts[key][:,targets_1d]
      
    return train_set, test_set, data_mean, data_std, train_keypts, test_keypts, targets_1d, offset


def load_data( path, subjects, actions):
    """
    Load 3d ground truth, put it in a dictionary

    Args
        path: String. Path where to load the data from
        flies: List of integers. Flies whose data will be loaded
        actions: List of strings. The actions to load
    Returns:
        data: Dictionary with keys k=(subject, action)
    """

    path = os.path.join(path, '*.pkl')
    fnames = glob.glob( path )
    
    assert fnames!=[], 'No files found'
  
    data = {}
    good_keypts = {}
    for subject in subjects:
        for action in actions:
            
            fname = fnames.copy()
            
            if subject in par.keys():
                fname = [file for file in fname if str(subject) in file]   
                
            if action in par.keys():
                fname = [file for file in fname if action in file]
                
            assert fnames!=[], 'No files found with desired criteria'
              
            for fname_ in fname:
          
                seqname = os.path.basename( fname_ )  
        
                poses = pickle.load(open(fname_, "rb"))
                poses3d = poses['points3d']
                
                if 'interval' in par.keys():
                    frames = np.arange(par['interval'][0], par['interval'][1])
                    poses3d = poses3d[frames, :,:] #only load the stimulation interval
                
                if 'dims_to_exclude' in par.keys():
                    dimensions = [i for i in range(par['ndims']) if i not in par['dims_to_exclude']]   
                    poses3d = poses3d[:, dimensions,:]
                
                poses3d = np.reshape(poses3d, 
                          (poses3d.shape[0], poses3d.shape[1]*poses3d.shape[2]))
                        
                data[ (subject, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension
                good_keypts[ (subject, action, seqname[:-4]) ] = poses['good_keypts']
                
    data = dict(sorted(data.items()))
    good_keypts = dict(sorted(good_keypts.items()))

    return data, good_keypts


if __name__ == "__main__":
    main()