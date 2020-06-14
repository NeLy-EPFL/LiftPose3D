"""Utility functions for dealing with DeepFly3D data."""

from __future__ import division

import os
import numpy as np
import glob
import torch
from src.normalize import normalize_data, anchor, normalization_stats
import pickle


TRAIN_SUBJECTS = [0,1,2,3,4,5]
TEST_SUBJECTS  = [6,7]

data_dir = '/data/LiftFly3D/DF3D/data_DF3D/'
out_dir = '/data/LiftFly3D/DF3D/cam_angles_2/cams15/'
actions = ['MDN_CsCh']
rcams = pickle.load(open('cameras.pkl', "rb"))

#cam_ids = [2, 4]
cam_ids = [1, 5]
#cam_ids = [0, 6]

#coordinate of limbs (see skeleton.py for description)
interval = np.arange(200,700)
dims_to_consider = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]
target_sets = [[ 1,  2,  3,  4],  [6,  7,  8,  9], [11, 12, 13, 14],
               [16, 17, 18, 19], [21, 22, 23, 24], [26, 27, 28, 29]]
ref_points = [0, 5, 10, 15, 20, 25]


def main():   
    
# =============================================================================
# This part is for predicting xyz of groundtruth from SH predictions
# =============================================================================
    
    print('behaviors' + str(actions))
    print('processing for camera' + str(cam_ids))
    
    # HG prediction (i.e. deeplabcut or similar) 
    train_set, test_set, data_mean, data_std, offset = \
        read_2d_predictions( actions, data_dir, rcams, target_sets, ref_points )
    
    torch.save(train_set, out_dir + 'train_2d.pth.tar')
    torch.save(test_set, out_dir + 'test_2d.pth.tar')
    torch.save({'mean': data_mean, 'std': data_std, 'offset': offset},
                out_dir + 'stat_2d.pth.tar')
    
    #3D ground truth
    train_set, test_set, data_mean, data_std, offset, vis_train, vis_test = \
        read_3d_data( actions, data_dir, target_sets, ref_points, rcams )    
    
    torch.save([train_set, vis_train], out_dir + 'train_3d.pth.tar')
    torch.save([test_set, vis_test], out_dir + 'test_3d.pth.tar')   
    torch.save({'mean': data_mean, 'std': data_std, 
                'target_sets': target_sets, 'ref_points': ref_points, 'offset': offset},
                out_dir + '/stat_3d.pth.tar')

# =============================================================================
# Preprocess pipelines
# =============================================================================
    
def read_3d_data( actions, data_dir, target_sets, ref_points, rcams=None):
  """
  Loads 3d poses, zero-centres and normalizes them
  """
  dim = 3
  
  # Load 3d data  
  train_set = load_data( data_dir, TRAIN_SUBJECTS, actions )
  test_set  = load_data( data_dir, TEST_SUBJECTS,  actions )

  #rotate to align with 2D
  train_set, vis_train = transform_world_to_camera( train_set, rcams, cam_ids )
  test_set, vis_test  = transform_world_to_camera( test_set, rcams, cam_ids )

  # anchor points
  train_set, _ = anchor( train_set, ref_points, target_sets, dim)
  test_set, offset = anchor( test_set, ref_points, target_sets, dim)

  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, ref_points, dim )

  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim )

  return train_set, test_set, data_mean, data_std, offset, vis_train, vis_test


def read_2d_predictions( actions, data_dir, rcams, target_sets, ref_points ):
  """
  Loads 2d data from precomputed Stacked Hourglass detections
  """

  train_set = load_stacked_hourglass( data_dir, TRAIN_SUBJECTS, actions, cam_ids)
  test_set  = load_stacked_hourglass( data_dir, TEST_SUBJECTS,  actions, cam_ids)

  # anchor points
  train_set, offset = anchor( train_set, ref_points, target_sets, dim=2)
  test_set, offset = anchor( test_set, ref_points, target_sets, dim=2)
  
  # Compute normalization statistics
  data_mean, data_std = normalization_stats( train_set, ref_points, dim=2 )
  
  # Divide every dimension independently
  train_set = normalize_data( train_set, data_mean, data_std, target_sets, dim=2 )
  test_set  = normalize_data( test_set,  data_mean, data_std, target_sets, dim=2 )
  
  return train_set, test_set, data_mean, data_std, offset


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

  path = os.path.join(path, '*')
  fnames = glob.glob( path )
  
  data = {}
  for fly in flies:
    for action in actions:
        
      fname = [file for file in fnames if ("Fly" + str(fly) + '_' in file) and (action in file)]    
      
      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        
        poses = pickle.load(open(fname_, "rb"))
        poses3d = poses['points3d'][interval, :,:] #only load the stimulation interval
        poses3d = poses3d[:, dims_to_consider,:]
        poses3d = np.reshape(poses3d, 
                          (poses3d.shape[0], poses3d.shape[1]*poses3d.shape[2]))
        
        data[ (fly, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension

  return data


def load_stacked_hourglass(data_dir, flies, actions, cam_ids):
  """
  Load 2d detections from disk, and put it in an easy-to-acess dictionary.

  Args
    data_dir: string. Directory where to load the data from,
    subjects: list of integers. Subjects whose data will be loaded.
    actions: list of strings. The actions to load.
  Returns
    data: dictionary with keys k=(subject, action, seqname)
  """

  path = os.path.join(data_dir, '*')
  fnames = glob.glob( path )

  data = {}
  for fly in flies:
    for action in actions:
        
      fname = [file for file in fnames if ("Fly"+ str(fly) + '_' in file) and (action in file)]   

      for fname_ in fname:
          
        seqname = os.path.basename( fname_ )  
        
        poses = pickle.load(open(fname_, "rb"))
        poses = poses['points2d'][:,interval,:,:]
        poses = poses[:,:,dims_to_consider,:]
        
        for cam in cam_ids:

            poses_cam = poses[cam,:,:,:]
            poses_cam = np.reshape(poses_cam, 
                         (poses.shape[1], poses.shape[2]*poses.shape[3]))    
        
            data[ (fly, action, seqname[:-4] + '.cam_' + str(cam)) ] = poses_cam

  return data
    

# =============================================================================
# Projection functions
# =============================================================================

def transform_world_to_camera( poses_set, cams, cam_ids, project=False ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    cam_ids: camera_ids to consider
  Returns
    transf: dictionary with 3d poses or 2d poses if projection is True
  """
  Ptransf = {}
  vis = {}
  for fly, a, seqname in sorted( poses_set.keys() ):

    P = poses_set[ (fly, a, seqname) ]

    for c in cam_ids:
      R, T, intr, distort, vis_pts = cams[c]
      P = transform(P, R, T)
      
      if project:
         P = project(P, intr)
      
      Ptransf[ (fly, a, seqname + ".cam_" + str(c)) ] = P
     
      vis_pts = vis_pts[dims_to_consider]
      vis_pts = vis_pts[get_coords_in_dim(target_sets, 1)]
      vis_pts = [item for item in list(vis_pts) for i in range(3)]
      vis[ (fly, a, seqname + ".cam_" + str(c)) ] = np.array(vis_pts, dtype=bool)
      
  return Ptransf, vis


def transform(P, R, T):
  """
  Rotate/translate 3d poses to camera viewpoint

  Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
  Returns
    transf: Nx2 points on camera
  """


  ndim = P.shape[1]
  P = np.reshape(P, [-1, 3])
  
  assert len(P.shape) == 2
  assert P.shape[1] == 3
  
  points3d_new_cs =  np.matmul(R, P.T).T + T
  
  return np.reshape( points3d_new_cs, [-1, ndim] )


def project(P, intr):
    
  ndim = P.shape[1]

  P = np.reshape(P, [-1, 3])  
  proj = np.squeeze(np.matmul(intr, P[:,:,np.newaxis]))
  proj = proj / proj[:, [2]]
  proj = proj[:, :2]
  
  return np.reshape( proj, [-1, int(ndim/3*2)] )


def get_coords_in_dim(targets, dim):
    
    if any(isinstance(el, list) for el in targets): #check is lists of lists
      dim_to_use = []
      for i in targets:
          dim_to_use += i
    else:
      dim_to_use = targets
  
    dim_to_use = np.array(dim_to_use)
    if dim == 2:    
      dim_to_use = np.sort( np.hstack( (dim_to_use*2, 
                                        dim_to_use*2+1)))
  
    elif dim == 3:
      dim_to_use = np.sort( np.hstack( (dim_to_use*3,
                                        dim_to_use*3+1,
                                        dim_to_use*3+2)))
    return dim_to_use


if __name__ == "__main__":
    main()