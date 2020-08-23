import os
import numpy as np
import glob
import torch
import src.utils as utils
import pickle
import yaml

#load global parameters
par = yaml.full_load(open("params.yaml", "rb"))

def main():   
    
    print('behaviors' + str(par['actions']))
    print('processing for camera' + str(par['cam_id']))
        
    #3D ground truth
    train_set, test_set, data_mean, data_std, targets_3d, rcams_test, offset = \
        read_3d_data( par['actions'], par['data_dir'], par['target_sets'], par['roots'])
    
    torch.save(train_set, par['out_dir'] + '/train_3d.pth.tar')
    torch.save(test_set, par['out_dir'] + '/test_3d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 'targets_3d': targets_3d, 'rcams': rcams_test, 'offset': offset},
                par['out_dir'] + '/stat_3d.pth.tar')
    
    #HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std, targets_2d = \
        read_2d_predictions( par['actions'], par['data_dir'], par['target_sets'], par['roots'])
    
    torch.save(train_set, par['out_dir'] + '/train_2d.pth.tar')
    torch.save(test_set, par['out_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 'targets_2d': targets_2d},
                par['out_dir'] + '/stat_2d.pth.tar')
       

def read_3d_data( actions, data_dir, target_sets, roots):
    """
    Pipeline for processing 3D ground-truth data
    """
  
    # Load 3d data
    train_set, rcams_train = load_data( data_dir, par['cam_id'], par['train_subjects'], actions )
    test_set, rcams_test  = load_data( data_dir, par['cam_id'], par['test_subjects'],  actions )
  
    #transform to camera coordinates
    train_set = transform_frame( train_set, rcams_train )
    test_set = transform_frame( test_set, rcams_test )
    
    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    train_set, _ = utils.anchor( train_set, roots, target_sets, par['out_dim'])
    test_set, offset = utils.anchor( test_set, roots, target_sets, par['out_dim'])

    # Standardize each dimension independently
    data_mean, data_std = utils.normalization_stats( train_set )
    train_set = utils.normalize_data( train_set, data_mean, data_std )
    test_set  = utils.normalize_data( test_set,  data_mean, data_std )
      
    #select coordinates to be predicted and return them as 'targets_3d'
    train_set, _ = utils.collapse(train_set, par['target_sets'], par['out_dim'])
    test_set, targets_3d = utils.collapse(test_set, par['target_sets'], par['out_dim'])

    return train_set, test_set, data_mean, data_std, targets_3d, rcams_test, offset


def read_2d_predictions( actions, data_dir, target_sets, roots):
    """
    Pipeline for processing 2D data (stacked hourglass predictions)
    """

    # Load 2d data
    train_set = load_stacked_hourglass( data_dir, par['cam_id'], par['train_subjects'], actions)
    test_set  = load_stacked_hourglass( data_dir, par['cam_id'], par['test_subjects'],  actions)

    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    train_set, _ = utils.anchor( train_set, roots, target_sets, par['in_dim'])
    test_set, offset = utils.anchor( test_set, roots, target_sets, par['in_dim'])
  
    # Standardize each dimension independently
    data_mean, data_std = utils.normalization_stats( train_set )
    train_set = utils.normalize_data( train_set, data_mean, data_std)
    test_set  = utils.normalize_data( test_set,  data_mean, data_std)
  
    #select coordinates to be predicted and return them as 'targets_2d'
    train_set, _ = utils.collapse(train_set, par['target_sets'], par['in_dim'])
    test_set, targets_2d = utils.collapse(test_set, par['target_sets'], par['in_dim'])
  
    return train_set, test_set, data_mean, data_std, targets_2d


def load_data( path, cam_id, flies='all', actions='all' ):
    """
    Load 3d ground truth, put it in an easy-to-access dictionary

    Args
        path: String. Path where to load the data from
        flies: List of integers. Flies whose data will be loaded
        actions: List of strings. The actions to load
    Returns
        data: Dictionary with keys (fly, action, filename)
    """
    
    path = os.path.join(path, '*.pkl')
    fnames = glob.glob( path )
    data, cam_par = {}, {}
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
        
                data[ (fly, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension
                cam_par[(fly, action, seqname[:-4])] = [poses[cam_id] for i in range(poses3d.shape[0])]
                
    #sort
    data = dict(sorted(data.items()))
    cam_par = dict(sorted(cam_par.items()))

    return data, cam_par


def load_stacked_hourglass(path, cam_id, flies='all', actions='all'):
    """
    Load 2d data, put it in a dictionary.
    
    Args
        path: string. Directory where to load the data from,
        flies: list of integers. Subjects whose data will be loaded.
        actions: list of strings. The actions to load.
    Returns
        data: dictionary with keys k=(fly, action, filename)
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
        
                poses = pickle.load(open(fname_, "rb"))
                poses = poses['points2d']
                if par['interval'] != []:
                    frames = np.arange(par['interval'][0], par['interval'][1])
                    poses = poses[:,frames,:,:]
                    
                dimensions = [i for i in range(par['ndims']) if i not in par['dims_to_exclude']]      
                poses = poses[:,:,dimensions,:]
                poses_cam = poses[cam_id,:,:,:]
                poses_cam = np.reshape(poses_cam, 
                        (poses.shape[1], poses.shape[2]*poses.shape[3]))    
        
                data[ (fly, action, seqname[:-4]) ] = poses_cam
            
    #sort
    data = dict(sorted(data.items()))

    return data


def transform_frame( poses, cam_par, project=False ):
    """
    Affine transform 3D cooridinates to camera frame

    Args
        poses: dictionary with 3d poses
        cams: dictionary with camera parameters
        cam_id: camera_id to consider
    Returns
        Ptransf: dictionary with 3d poses or 2d poses if projection is True
        vis: boolean array with coordinates visible from the camera
    """
    Ptransf = {}
 
    for fly, a, seqname in poses.keys():

        Pworld = poses[ (fly, a, seqname) ]
        rcams = cam_par[ (fly, a, seqname) ]
        
        Pcam = utils.world_to_camera(Pworld, rcams)
      
        if project:
            Pcam = utils.project_to_camera(Pcam, rcams['intr'])
      
        Ptransf[ (fly, a, seqname) ] = Pcam

    return Ptransf
   
            
if __name__ == "__main__":
    main()   