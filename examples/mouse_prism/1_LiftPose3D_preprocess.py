import sys
import torch
import src.utils as utils
import src.load as load 
import src.stat as stat
import src.transform as transform
import yaml

usr_input = sys.argv[-1]

#load global parameters
par = yaml.full_load(open(usr_input, "rb"))

def main():   

    print('behaviors' + str(par['actions']))
        
    #xy data
    train_set, test_set, mean, std, targets_2d, offset_2d = \
    create_xy_data( par )
    
    torch.save(train_set, par['data_dir'] + '/train_2d.pth.tar')
    torch.save(test_set, par['data_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': mean, 'std': std, 
                'targets_2d': targets_2d, 'offset': offset_2d},
                par['data_dir'] + '/stat_2d.pth.tar')
    
    #z data
    train_set, test_set, mean, std, train_keypts, test_keypts, targets_1d, offset_1d = \
        create_z_data( par )
        
    torch.save([train_set, train_keypts], par['data_dir'] + '/train_3d.pth.tar')
    torch.save([test_set, test_keypts], par['data_dir'] + '/test_3d.pth.tar')   
    torch.save({'mean': mean, 'std': std, 
                'targets_1d': targets_1d, 'offset': offset_1d,
                'LR_train': train_keypts, 'LR_test': test_keypts,
                'output_size': len(targets_1d),
                'input_size': len(targets_2d)},
                par['data_dir'] + '/stat_3d.pth.tar')
    
      
def create_xy_data( par ):
    """
    Creates 2d poses by projecting 3d poses with the corresponding camera
    parameters.
    """

    # Load data
    train_set, _, _ = load.load_3D( par['data_dir'], subjects=par['train_subjects'], actions=par['actions'] )
    test_set,  _, _ = load.load_3D( par['data_dir'], subjects=par['test_subjects'],  actions=par['actions'] )
  
    #project data to ventral view
    train_set = transform.XY_coord( train_set )
    test_set  = transform.XY_coord( test_set )

    # anchor to root points
    train_set, _ = utils.anchor_to_root( train_set, par['roots'], par['target_sets'], par['in_dim'])
    test_set, offset = utils.anchor_to_root( test_set, par['roots'], par['target_sets'], par['in_dim'])
    
    # Divide every dimension independently
    mean, std = stat.normalization_stats( train_set )
    train_set = stat.normalize( train_set, mean, std )
    test_set  = stat.normalize( test_set,  mean, std )
    
    #select coordinates to be predicted and return them as 'targets_3d'
    train_set, _ = utils.remove_roots( train_set, par['target_sets'], par['in_dim'] )
    test_set, targets_2d = utils.remove_roots( test_set, par['target_sets'], par['in_dim'] )
    
    return train_set, test_set, mean, std, targets_2d, offset


def create_z_data( par ):

    # Load data
    train_set, train_keypts, _ = load.load_3D( par['data_dir'], subjects=par['train_subjects'], actions=par['actions'] )
    test_set, test_keypts, _  = load.load_3D( par['data_dir'], subjects=par['test_subjects'],  actions=par['actions'] )
  
    #rotate to align with 2D
    train_set = transform.Z_coord( train_set)
    test_set  = transform.Z_coord( test_set )
  
    # anchor to root points
    train_set, _ = utils.anchor_to_root( train_set, par['roots'], par['target_sets'], par['out_dim'])
    test_set, offset = utils.anchor_to_root( test_set, par['roots'], par['target_sets'], par['out_dim'])

    # Divide every dimension independently
    mean, std = stat.normalization_stats( train_set)
    train_set = stat.normalize( train_set, mean, std )
    test_set  = stat.normalize( test_set,  mean, std )
  
    #select coordinates to be predicted and return them as 'targets_1d'
    train_set, _ = utils.remove_roots(train_set, par['target_sets'], par['out_dim'])
    test_set, targets_1d = utils.remove_roots(test_set, par['target_sets'], par['out_dim'])
    
    for key in train_keypts.keys():
        train_keypts[key] = train_keypts[key][:,targets_1d]
    for key in test_keypts.keys():
        test_keypts[key] = test_keypts[key][:,targets_1d]
      
    return train_set, test_set, mean, std, train_keypts, test_keypts, targets_1d, offset


if __name__ == "__main__":
    main()