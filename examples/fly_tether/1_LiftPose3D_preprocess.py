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
    print('processing for camera' + str(par['cam_id']))
        
    #3D ground truth
    train_set, test_set, mean, std, targets_3d, rcams_test, offset = \
        read_3d_data( par )
    
    torch.save(train_set, par['out_dir'] + '/train_3d.pth.tar')
    torch.save(test_set, par['out_dir'] + '/test_3d.pth.tar')
    torch.save({'mean': mean, 'std': std, 'targets_3d': targets_3d, 'rcams': rcams_test, 'offset': offset},
                par['out_dir'] + '/stat_3d.pth.tar')
    
    #HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, mean, std, targets_2d = \
        read_2d_predictions( par )
    
    torch.save(train_set, par['out_dir'] + '/train_2d.pth.tar')
    torch.save(test_set, par['out_dir'] + '/test_2d.pth.tar')
    torch.save({'mean': mean, 'std': std, 'targets_2d': targets_2d},
                par['out_dir'] + '/stat_2d.pth.tar')
       

def read_3d_data( par ):
    """
    Pipeline for processing 3D ground-truth data
    """
    
    # Load 3d data
    train, _, rcams_train = load.load_3D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['train_subjects'], actions=par['actions'] )
    test,  _, rcams_test  = load.load_3D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['test_subjects'],  actions=par['actions'] )
    
    #transform to camera coordinates
    train = transform_frame( train, rcams_train )
    test = transform_frame( test, rcams_test )
    
    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    train, _     = utils.anchor_to_root( train, par['roots'], par['target_sets'], par['out_dim'])
    test, offset = utils.anchor_to_root(  test, par['roots'], par['target_sets'], par['out_dim'])

    # Standardize each dimension independently
    mean, std = stat.normalization_stats( train )
    train = stat.normalize( train, mean, std )
    test  = stat.normalize( test,  mean, std )
      
    #select coordinates to be predicted and return them as 'targets_3d'
    train, _ = utils.remove_roots(train, par['target_sets'], par['out_dim'])
    test, targets_3d = utils.remove_roots(test, par['target_sets'], par['out_dim'])

    return train, test, mean, std, targets_3d, rcams_test, offset


def read_2d_predictions( par ):
    """
    Pipeline for processing 2D data (stacked hourglass predictions)
    """

    # Load 2d data
    train = load.load_2D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['train_subjects'], actions=par['actions'])
    test  = load.load_2D( par['data_dir'], par, cam_id=par['cam_id'], subjects=par['test_subjects'],  actions=par['actions'])

    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    train, _ = utils.anchor_to_root( train, par['roots'], par['target_sets'], par['in_dim'])
    test, offset = utils.anchor_to_root( test, par['roots'], par['target_sets'], par['in_dim'])
  
    # Standardize each dimension independently
    mean, std = stat.normalization_stats( train )
    train = stat.normalize( train, mean, std)
    test  = stat.normalize( test,  mean, std)
  
    #select coordinates to be predicted and return them as 'targets'
    train, _ = utils.remove_roots(train, par['target_sets'], par['in_dim'])
    test, targets = utils.remove_roots(test, par['target_sets'], par['in_dim'])
  
    return train, test, mean, std, targets


def transform_frame( poses_world, cam_par, project=False ):
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
        for c in list(cam_par[(s, a, f)].keys()):

            rcams = cam_par[ (s, a, f) ][c]
        
            Pcam = transform.world_to_camera(poses_world[(s, a, f)], rcams)
      
            if project:
                Pcam = transform.project_to_camera(Pcam, rcams['intr'])
      
            poses_cam[ (s, a, f + '.cam_' + str(c)) ] = Pcam
            
    #sort dictionary
    poses_cam = dict(sorted(poses_cam.items()))

    return poses_cam
   
    
if __name__ == "__main__":
    main()   