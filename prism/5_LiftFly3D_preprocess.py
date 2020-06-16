import os
import numpy as np
import glob
import torch
import pickle
import src.utils as utils

#TRAIN_SUBJECTS = [1,2,3]
TRAIN_SUBJECTS = [1,2,3,4] #for optobot
TEST_SUBJECTS  = [4]

data_dir = '/data/LiftFly3D/prism/data_oriented_plus_noise/'
#data_dir = '/data/LiftFly3D/optobot/network/'
actions = ['PR']
rcams = []

#select cameras and joints visible from cameras
#target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14],
#               [16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]]
target_sets = [[ 2,  3,  4],  [7,  8,  9], [12, 13, 14], #for optobot
               [17, 18, 19], [22, 23, 24], [27, 28, 29]]
ref_points = [0, 5, 10,15, 20, 25]


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
    train_set, test_set, data_mean, data_std, LR_train, LR_test, targets_1d, offset = \
        create_z_data( actions, data_dir, target_sets, ref_points, rcams )
        
    torch.save([train_set, LR_train], data_dir + '/train_3d.pth.tar')
    torch.save([test_set, LR_test], data_dir + '/test_3d.pth.tar')   
    torch.save({'mean': data_mean, 'std': data_std, 
                'targets_1d': targets_1d, 'offset': offset,
                'LR_train': LR_train, 'LR_test': LR_test},
                data_dir + '/stat_3d.pth.tar')
    
      
def create_xy_data( actions, data_dir, target_sets, ref_points ):
    """
    Creates 2d poses by projecting 3d poses with the corresponding camera
    parameters.
    """

    dim=2
    # Load 3d data
    train_set, _ = load_data( data_dir, TRAIN_SUBJECTS, actions )
    test_set, _  = load_data( data_dir, TEST_SUBJECTS,  actions )
  
    #filter data
#  train_set = utils.filter_data(train_set)
#    test_set = utils.filter_data(test_set, window=5, order=2)
  
    #rotate to align with 2D
    train_set = XY_coord( train_set )
    test_set  = XY_coord( test_set )
  
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
    train_set, LR_train = load_data( data_dir, TRAIN_SUBJECTS, actions )
    test_set, LR_test  = load_data( data_dir, TEST_SUBJECTS,  actions )

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
  
    return train_set, test_set, data_mean, data_std, LR_train, LR_test, targets_1d, offset


def load_data( path, flies, actions ):
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