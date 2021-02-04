import sys
import torch
import src.utils as utils
import src.load as load 
import src.stat as stat
import src.transform as transform
import yaml
from scipy.spatial.transform import Rotation as Rot
from numpy import linalg
import numpy as np
import pickle 

usr_input = sys.argv[-1]

#load global parameters
par = yaml.full_load(open(usr_input, "rb"))

def main():   
    
    print('processing' )
    
    #obtain bootstrapped mean, std
    mean_2d, std_2d, mean_3d, std_3d = obtain_projected_stats(par['eangle'])
    # mean_2d, std_2d, mean_3d, std_3d = 0,0,0,0
    
    #HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, targets_2d = \
        read_2d_predictions( par, mean_2d, std_2d )
    
    torch.save(train_set, par['out_dir'] + '/train_2d.pth.tar')
    torch.save(test_set, par['out_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': mean_2d, 
                'std': std_2d, 
                'targets_2d': targets_2d,
                'eangle': par['eangle'],
                'roots': par['roots'], 
                'target_sets': par['target_sets'],
                'vis': par['vis'],
                'focal_length': np.array(par['focal_length'])*par['pxtomm'],
                'frame_size': par['frame_size'],
                'axsorder': par['axsorder']},
                par['out_dir'] + '/stat_2d.pth.tar')
    
    #3D ground truth
    train_set, test_set, targets_3d, rcams_test, offset = \
        read_3d_data( par, mean_3d, std_3d )
    
    torch.save(train_set, par['out_dir'] + '/train_3d.pth.tar')
    torch.save(test_set, par['out_dir'] + '/test_3d.pth.tar')
    torch.save({'mean': mean_3d, 'std': std_3d, 
                'targets_3d': targets_3d, 
                'rcams': rcams_test, 
                'offset': offset,
                'output_size': len(targets_3d),
                'input_size': len(targets_2d)},
                par['out_dir'] + '/stat_3d.pth.tar')
       
    
def obtain_projected_stats(eangle, th=0.05):
    
    error = 1
    count = 0
    error_log = []
    #run until convergence
    while(error>th):
                
        #obtain randomly projected points
        train, _, _ = load.load_3D( par['data_dir'], par, subjects=par['train_subjects'], actions=par['actions'] )
        train_2d = project_to_eangle( train, eangle, project=True )
        train_3d = project_to_eangle( train, eangle, project=False )
        train_2d, _  = utils.anchor_to_root( train_2d, par['roots'], par['target_sets'], par['in_dim'])
        train_3d, _  = utils.anchor_to_root( train_3d, par['roots'], par['target_sets'], par['out_dim'])     
        train_2d = np.concatenate([v for k,v in train_2d.items()], 0)
        train_3d = np.concatenate([v for k,v in train_3d.items()], 0)
        
        #bootstral mean, std
        if count == 0:
            train_samples_2d = train_2d
            mean_old_2d = np.zeros(train_2d.shape[1])
            std_old_2d  = np.zeros(train_2d.shape[1])
            train_samples_3d = train_3d
            mean_old_3d = np.zeros(train_3d.shape[1])
            std_old_3d  = np.zeros(train_3d.shape[1])
        else:
            train_samples_2d = np.vstack((train_samples_2d,train_2d))
            train_samples_3d = np.vstack((train_samples_3d,train_3d))
            
        mean_2d = np.nanmean(train_samples_2d, axis=0)
        std_2d  = np.nanstd(train_samples_2d, axis=0)
        mean_3d = np.nanmean(train_samples_3d, axis=0)
        std_3d  = np.nanstd(train_samples_3d, axis=0)
        
        error = linalg.norm(mean_2d - mean_old_2d) + linalg.norm(std_2d - std_old_2d) + \
                linalg.norm(mean_3d - mean_old_3d) + linalg.norm(std_3d - std_old_3d)
        error_log.append(error)
        
        print(error)
        mean_old_2d = mean_2d
        std_old_2d = std_2d
        mean_old_3d = mean_3d
        std_old_3d = std_3d
        count += 1
        
        pickle.dump(error_log, open(par['out_dir'] + '/error_log.pkl','wb'))
    
    return mean_2d, std_2d, mean_3d, std_3d


def read_3d_data( par, mean, std):
    """
    Pipeline for processing 3D ground-truth data
    """

    # Load data
    train, _, rcams_train = load.load_3D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['train_subjects'], actions=par['actions'] )
    test,  _, rcams_test  = load.load_3D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['test_subjects'],  actions=par['actions'] )
    
    #transform to camera coordinates
    # train = transform_frame( train, rcams_train )
    test = transform_frame( test, rcams_test )
    
    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    # train, _     = utils.anchor_to_root( train, par['roots'], par['target_sets'], par['out_dim'])
    test, offset = utils.anchor_to_root(  test, par['roots'], par['target_sets'], par['out_dim'])

    # Standardize each dimension independently
    # train = stat.normalize( train, mean, std )
    test  = stat.normalize( test,  mean, std )
      
    #select coordinates to be predicted and return them as 'targets_3d'
    # train, _ = utils.remove_roots(train, par['target_sets'], par['out_dim'])
    test, targets_3d = utils.remove_roots(test, par['target_sets'], par['out_dim'])

    return train, test, targets_3d, rcams_test, offset


def read_2d_predictions( par, mean, std ):
    """
    Pipeline for processing 2D data (stacked hourglass predictions)
    """

    # Load data
    # train = load.load_2D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['train_subjects'], actions=par['actions'])
    test  = load.load_2D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['test_subjects'],  actions=par['actions'])

    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    # train, _ = utils.anchor_to_root( train, par['roots'], par['target_sets'], par['in_dim'])
    test, offset = utils.anchor_to_root( test, par['roots'], par['target_sets'], par['in_dim'])
  
    # Standardize each dimension independently
    # train = stat.normalize( train, mean, std)
    test  = stat.normalize( test,  mean, std)
  
    #select coordinates to be predicted and return them as 'targets'
    # train, _ = utils.remove_roots(train, par['target_sets'], par['in_dim'])
    test, targets = utils.remove_roots(test, par['target_sets'], par['in_dim'])
    
    return 0, test, targets


def transform_frame( poses_world, cam_par=None, project=False ):
    """
    Affine transform 3D cooridinates to camera frame

    Args
        poses_world: dictionary with 3d poses in world coordinates
        cams: dictionary with camera parameters
        cam_id: camera_id to consider
    Returns
        poses_cam: dictionary with 3d poses (2d poses if projection is True) in camera coordinates 
        vis: boolean array with coordinates visible from the camera
    """
    
    poses_cam = {}
    for s, a, f in poses_world.keys():
        if cam_par[(s, a, f)]is None:
            return poses_world
        
        for c in list(cam_par[(s, a, f)].keys()):

            rcams = cam_par[ (s, a, f) ][c]
            poses = poses_world[(s, a, f)]
                
            Pcam = transform.world_to_camera(poses, rcams)
      
            if project:
                Pcam = transform.project_to_camera(Pcam, rcams['intr'])
      
            poses_cam[ (s, a, f + '.cam_' + str(c)) ] = Pcam
            
    #sort dictionary
    poses_cam = dict(sorted(poses_cam.items()))

    return poses_cam


def project_to_eangle( poses_world, eangles, project=False ):
    """
    Affine transform 3D cooridinates to camera frame

    Args
        poses_world: dictionary with 3d poses in world coordinates
        cams: dictionary with camera parameters
        cam_id: camera_id to consider
    Returns
        poses_cam: dictionary with 3d poses (2d poses if projection is True) in camera coordinates 
        vis: boolean array with coordinates visible from the camera
    """
    
    poses_cam = {}
    for s, a, f in poses_world.keys():
        
        
        if len(par['vis'])==2:
            lr = np.random.binomial(1,0.5)
            if lr:
                eangle = eangles[0]
                vis = np.array(par['vis'][0])
            else:
                eangle = eangles[1]
                vis = np.array(par['vis'][1])
        else:
            vis = np.array(par['vis'][0])
            eangle = eangles[0]
        
        #generate Euler angles
        n = poses_world[(s, a, f)].shape[0]
        alpha = np.random.uniform(low=eangle[0][0], high=eangle[0][1], size=n)
        beta = np.random.uniform(low=eangle[1][0], high=eangle[1][1], size=n)
        gamma = np.random.uniform(low=eangle[2][0], high=eangle[2][1], size=n)
        
        #convert to rotation matrices
        eangle = [[alpha[i], beta[i], gamma[i]] for i in range(n)]      
        R = Rot.from_euler(par['axsorder'], eangle, degrees=True).as_matrix()
            
        rcams = {'R': R, 
                 'tvec': np.array([0, 0, 117]),
                 'vis': vis}
        
        #obtain 3d pose in camera coordinates
        Pcam = transform.world_to_camera(poses_world[(s, a, f)], rcams)
      
        #project to camera axis
        if project:
            fl = np.array(par['focal_length'])*par['pxtomm']
            ic = par['frame_size']
            Pcam = transform.project_to_camera(Pcam, cam_par=(fl[0], fl[1], ic[0], ic[1]))
      
        poses_cam[ (s, a, f) ] = Pcam
            
    #sort dictionary
    poses_cam = dict(sorted(poses_cam.items()))

    return poses_cam

    
if __name__ == "__main__":
    main()   