import os
import numpy as np
import glob
import torch
import pickle
import src.utils as utils

TRAIN_SUBJECTS = [1,2,4]
#TRAIN_SUBJECTS = [1,2,3,4] #for optobot (using all data here)
TEST_SUBJECTS  = [3]

data_dir = '/data/LiftFly3D/prism/data_oriented/test_data'
#data_dir = '/data/LiftFly3D/prism/data_oriented/training_data'
#data_dir = '/data/LiftFly3D/optobot/network'
actions = ['PR']
rcams = []
scale = None

#select cameras and joints visible from cameras
target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14],
               [16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]]
#target_sets = [[ 2,  3,  4],  [7,  8,  9], [12, 13, 14], #for optobot
#               [17, 18, 19], [22, 23, 24], [27, 28, 29]]
ref_points = [0, 5, 10, 15, 20, 25]


def main():   

    #xy data
    train_set, test_set, data_mean, data_std, targets_2d, offset = \
    create_xy_data( actions, data_dir, target_sets, ref_points )
    
    torch.save(train_set, data_dir + '/train_2d.pth.tar')
    torch.save(test_set, data_dir + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_2d': targets_2d, 'offset': offset},
                data_dir + '/stat_2d.pth.tar')
    
    #z data
    train_set, test_set, data_mean, data_std, train_keypts, test_keypts, targets_1d, offset = \
        create_z_data( actions, data_dir, target_sets, ref_points, rcams )
        
    torch.save([train_set, train_keypts], data_dir + '/train_3d.pth.tar')
    torch.save([test_set, test_keypts], data_dir + '/test_3d.pth.tar')   
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_1d': targets_1d, 'offset': offset,
                'LR_train': train_keypts, 'LR_test': test_keypts},
                data_dir + '/stat_3d.pth.tar')
    
      
def create_xy_data( actions, data_dir, target_sets, ref_points ):
    """
    Creates 2d poses by projecting 3d poses with the corresponding camera
    parameters.
    """

    dim=2
    # Load 3d data
    train_set, _ = load_data( data_dir, TRAIN_SUBJECTS, actions, scale )
    test_set, _  = load_data( data_dir, TEST_SUBJECTS,  actions, scale )
  
    #filter data
#  train_set = utils.filter_data(train_set)
#    test_set = utils.filter_data(test_set, window=5, order=2)
  
    #project data to ventral view
    train_set = XY_coord( train_set )
    test_set  = XY_coord( test_set )
    
    #save a template before normalizing
#    template_set = np.mean(np.vstack(train_set.values()),axis=0)
#    refs = [ 0, 1,  2,  3,  4, 5,  6,  7, 4, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
#    refs = np.array(refs)
#    refs = np.sort( np.hstack( (refs*2, refs*2+1)))
#    torch.save(template_set[refs], data_dir + '/template.pth.tar')

    # anchor points
    train_set, _ = utils.anchor( train_set, ref_points, target_sets, dim)
    test_set, offset = utils.anchor( test_set, ref_points, target_sets, dim)
    
    # Compute normalization statistics
    data_mean, data_std = utils.normalization_stats( train_set)
  
    # Divide every dimension independently
    train_set = utils.normalize_data( train_set, data_mean, data_std )
    test_set  = utils.normalize_data( test_set,  data_mean, data_std )
  
    #select coordinates to be predicted and return them as 'targets_3d'
    train_set, _ = utils.collapse(train_set, None, target_sets, dim)
    test_set, targets_2d = utils.collapse(test_set, None, target_sets, dim)

    return train_set, test_set, data_mean, data_std, targets_2d, offset


def create_z_data( actions, data_dir, target_sets, ref_points, rcams ):

    dim = 1
    # Load 3d data
    train_set, train_keypts = load_data( data_dir, TRAIN_SUBJECTS, actions, scale )
    test_set, test_keypts  = load_data( data_dir, TEST_SUBJECTS,  actions, scale )

    #filter data
#  train_set = utils.filter_data(train_set)
#    test_set = utils.filter_data(test_set, window=5, order=2)
  
    #rotate to align with 2D
    train_set = Z_coord( train_set)
    test_set  = Z_coord( test_set )
  
    # anchor points
    train_set, _ = utils.anchor( train_set, ref_points, target_sets, dim)
    test_set, offset = utils.anchor( test_set, ref_points, target_sets, dim)

    # Compute normalization statistics.
    data_mean, data_std = utils.normalization_stats( train_set)

    # Divide every dimension independently
    train_set = utils.normalize_data( train_set, data_mean, data_std )
    test_set  = utils.normalize_data( test_set,  data_mean, data_std )
  
    #select coordinates to be predicted and return them as 'targets_1d'
    train_set, _ = utils.collapse(train_set, None, target_sets, dim)
    test_set, targets_1d = utils.collapse(test_set, None, target_sets, dim)
    
    for key in train_keypts.keys():
        train_keypts[key] = train_keypts[key][:,targets_1d]
    for key in test_keypts.keys():
        test_keypts[key] = test_keypts[key][:,targets_1d]
  
    return train_set, test_set, data_mean, data_std, train_keypts, test_keypts, targets_1d, offset


def load_data( path, flies, actions, scale=None):
    """
    Load 3d ground truth, put it in a dictionary

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
    good_keypts = {}
    for fly in flies:
        for action in actions:
        
            fname = [file for file in fnames if ("00" + str(fly) + '_' in file) and (action in file)]    
      
            for fname_ in fname:
          
                seqname = os.path.basename( fname_ )  
        
                poses = pickle.load(open(fname_, "rb"))
                poses3d = poses['points3d']
                poses3d = np.reshape(poses3d, 
                          (poses3d.shape[0], poses3d.shape[1]*poses3d.shape[2]))
                
                if scale is not None:
                    poses3d /= scale
        
                data[ (fly, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension
                good_keypts[ (fly, action, seqname[:-4]) ] = poses['good_keypts'].to_numpy()

    return data, good_keypts


def XY_coord( poses_set):
    """
    Project 3d poses to XY coord
    """
    t2d = {}

    for key in sorted( poses_set.keys() ):
        t3d = poses_set[ key ]

        ndim = t3d.shape[1]
        XY = np.reshape(t3d, [-1, 3])
        XY = XY[:,:2]
        t2d[ key ] = np.reshape( XY, [-1, ndim//3*2] )
 
    return t2d


def Z_coord( poses_set):
    """
    Project 3d poses to Z coord
    """
    t1d = {}

    for key in sorted( poses_set.keys() ):
        t3d = poses_set[ key ]

        ndim = t3d.shape[1]
        Z = np.reshape(t3d, [-1, 3])
        Z = Z[:,2]
        t1d[ key ] = np.reshape( Z, [-1, ndim//3] )

    return t1d


if __name__ == "__main__":
    main()