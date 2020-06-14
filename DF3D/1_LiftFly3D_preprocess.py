import os
import numpy as np
import glob
import torch
import src.normalize as norm
import pickle
import scipy.signal as scs


TRAIN_SUBJECTS = [0,1,2,3,4,5]
TEST_SUBJECTS  = [6,7]

cam_id = 5

data_dir = '/data/LiftFly3D/DF3D/data_DF3D/'
out_dir = '/data/LiftFly3D/DF3D/cam_angles/cam15/cam' + str(cam_id)
actions = ['MDN_CsCh', 'aDN_CsCh', 'PR']
rcams = pickle.load(open('cameras.pkl', "rb"))

interval = None #all data
#interval = np.arange(200,700) #data during stimulation only
dims_to_consider = [i for i in range(38) if i not in [15,16,17,18,34,35,36,37]] 
target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14],
               [16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]]
ref_points = [0, 5, 10, 15, 20, 25]


def main():   
    
    print('behaviors' + str(actions))
    print('processing for camera' + str(cam_id))
    
    #3D ground truth
    train_set, test_set, data_mean, data_std, targets_3d, vis = \
        read_3d_data( actions, data_dir, target_sets, ref_points, rcams)
    
    torch.save(train_set, out_dir + '/train_3d.pth.tar')
    torch.save(test_set, out_dir + '/test_3d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 'targets_3d': targets_3d},
                out_dir + '/stat_3d.pth.tar')
    
    # HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std, targets_2d = \
        read_2d_predictions( actions, data_dir, rcams, target_sets, ref_points, vis)
    
    torch.save(train_set, out_dir + '/train_2d.pth.tar')
    torch.save(test_set, out_dir + '/test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 'targets_2d': targets_2d},
                out_dir + '/stat_2d.pth.tar')
    
def read_3d_data( actions, data_dir, target_sets, ref_points, rcams=None):
    """
    Pipeline for processing 3D ground-truth data
    """
  
    dim = 3
    # Load 3d data
    train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
    test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )
  
    #filter data
#  train_set = filter_data(train_set)
    test_set = filter_data(test_set)
  
    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    train_set, _ = norm.anchor( train_set, ref_points, target_sets, dim)
    test_set, _ = norm.anchor( test_set, ref_points, target_sets, dim)

    # Compute mean, std
    data_mean, data_std = norm.normalization_stats( train_set )

    # Standardize each dimension independently
    train_set = norm.normalize_data( train_set, data_mean, data_std )
    test_set  = norm.normalize_data( test_set,  data_mean, data_std )
  
    #transform to camera coordinates
    if rcams is not None:
        train_set, _ = transform_frame( train_set, rcams, cam_id )
        test_set, vis  = transform_frame( test_set, rcams, cam_id )
      
    #select coordinates to be predicted and return them as 'targets_3d'
    train_set, _ = norm.collapse(train_set, vis, target_sets, dim)
    test_set, targets_3d = norm.collapse(test_set, vis, target_sets, dim)

    return train_set, test_set, data_mean, data_std, targets_3d, vis


def read_2d_predictions( actions, data_dir, rcams, target_sets, ref_points, vis):
    """
    Pipeline for processing 2D data (stacked hourglass predictions)
    """

    dim = 2
    # Load 2d data
    train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions, cam_id)
    test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions, cam_id)
  
    #filter data
#  train_set = filter_data(train_set)
    test_set = filter_data(test_set)

    # anchor points to body-coxa (to predict legjoints wrt body-boxas)
    train_set, _ = norm.anchor( train_set, ref_points, target_sets, dim)
    test_set, offset = norm.anchor( test_set, ref_points, target_sets, dim)
  
    # Compute mean, std
    data_mean, data_std = norm.normalization_stats( train_set )
  
    # Standardize each dimension independently
    train_set = norm.normalize_data( train_set, data_mean, data_std)
    test_set  = norm.normalize_data( test_set,  data_mean, data_std)
  
    #select coordinates to be predicted and return them as 'targets_2d'
    train_set, _ = norm.collapse(train_set, vis, target_sets, dim)
    test_set, targets_2d = norm.collapse(test_set, vis, target_sets, dim)
  
    return train_set, test_set, data_mean, data_std, targets_2d


def load_data( path, flies, actions ):
    """
    Load 3d ground truth, put it in an easy-to-access dictionary

    Args:
        path: String. Path where to load the data from
        flies: List of integers. Flies whose data will be loaded
        actions: List of strings. The actions to load
    Returns:
        data: Dictionary with keys k=(fly, action, filename)
    """

    path = os.path.join(path, '*')
    fnames = glob.glob( path )
    data = {}
    for fly in flies:
        for action in actions:
        
            fname = [file for file in fnames if ("Fly" + str(fly) + '_' in file) and (action in file)]    
      
            for fname_ in fname:

                seqname = os.path.basename( fname_ )  
        
                poses = pickle.load(open(fname_, "rb"))
                poses3d = poses['points3d']
                if interval is not None:
                    poses3d = poses3d[interval, :,:] #only load the stimulation interval
                poses3d = poses3d[:, dims_to_consider,:]
                poses3d = np.reshape(poses3d, 
                          (poses3d.shape[0], poses3d.shape[1]*poses3d.shape[2]))
        
                data[ (fly, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension

    return data


def load_stacked_hourglass(path, flies, actions, cam_id):
    """
    Load 2d data, put it in an easy-to-acess dictionary.
    
    Args
        path: string. Directory where to load the data from,
        flies: list of integers. Subjects whose data will be loaded.
        actions: list of strings. The actions to load.
    Returns
        data: dictionary with keys k=(fly, action, filename)
    """

    path = os.path.join(path, '*')
    fnames = glob.glob( path )

    data = {}
    for fly in flies:
        for action in actions:
        
            fname = [file for file in fnames if ("Fly"+ str(fly) + '_' in file) and (action in file)]   

            for fname_ in fname:
          
                seqname = os.path.basename( fname_ )  
        
                poses = pickle.load(open(fname_, "rb"))
                poses = poses['points2d']
                if interval is not None:
                    poses = poses[:,interval,:,:]
                poses = poses[:,:,dims_to_consider,:]
        
                poses_cam = poses[cam_id,:,:,:]
                poses_cam = np.reshape(poses_cam, 
                        (poses.shape[1], poses.shape[2]*poses.shape[3]))    
        
                data[ (fly, action, seqname[:-4] + '.cam_' + str(cam_id)) ] = poses_cam

    return data
    

def transform_frame( poses, cams, cam_id, project=False ):
    """
    Affine transform 3D cooridinates to camera frame

    Args
        poses: dictionary with 3d poses
        cams: dictionary with camera parameters
        cam_ids: camera_ids to consider
    Returns
        Ptransf: dictionary with 3d poses or 2d poses if projection is True
        vis: boolean array with coordinates visible from the camera
    """
    Ptransf = {}
    vis = {}
    for fly, a, seqname in sorted( poses.keys() ):

        Pworld = poses[ (fly, a, seqname) ]

        R, T, intr, distort, vis_pts = cams[cam_id]
        Pcam = norm.world_to_camera(Pworld, R, T)
      
        if project:
            Pcam = norm.project_to_camera(Pcam, intr)
      
        Ptransf[ (fly, a, seqname + ".cam_" + str(cam_id)) ] = Pcam
      
        vis_pts = vis_pts[dims_to_consider]
        vis = np.array(vis_pts, dtype=bool)

    return Ptransf, vis


def filter_data(poses, window=19, order=4):
    '''
    Filter time series using Savitzky-Golay filter
    
    Args
        poses: dictionary with 3d poses
        window: int window of filter function (odd)
        order: int order of filter
    '''
        
    for k in poses.keys():
        poses_smooth = np.zeros_like(poses[k])
        for j in range(poses_smooth.shape[1]):
                poses_smooth[:,j] = scs.savgol_filter(poses[k][:,j], window, order)
                
        poses[k] = poses_smooth
        
    return poses    
            
if __name__ == "__main__":
    main()