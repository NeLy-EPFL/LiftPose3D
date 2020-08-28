import numpy as np
import copy
import scipy.signal as scs
from operator import itemgetter
from itertools import groupby
import os
import matplotlib.pyplot as plt

def normalization_stats(train_set):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={1,2,3} dimensionality of the data
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
  """

  complete_data = copy.deepcopy( np.vstack( train_set.values() ))
  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)
  
  return data_mean, data_std


def normalize_data(data, data_mean, data_std ):
  """
  Normalizes a dictionary of poses
  """
 
  for key in data.keys():
    data[ key ] -= data_mean
    data[ key ] /= data_std

  return data


def unNormalizeData(data, data_mean, data_std):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing
  """
  data *= data_std
  data += data_mean
    
  return data


def get_coords_in_dim(targets, dim):
    
    if len(targets)>1:
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


def anchor(poses, anchors, target_sets, dim):
  """
  Center points in targset sets around anchors
  """
  
  offset = {}
  for k in poses.keys():
      offset[k] = np.zeros_like(poses[k])
      for i, anch in enumerate(anchors):
          for j in [anch]+target_sets[i]:
              offset[k][:, dim*j:dim*(j+1)] += poses[k][:, dim*anch:dim*(anch+1)]

  for k in poses.keys():
      poses[k] -= offset[k]
      
  return poses, offset


def collapse(data, vis, targets, dim ):
  """
  Normalizes a dictionary of poses
  """
  dim_to_use = get_coords_in_dim(targets, dim)
  
  if vis is not None:
      vis = np.array([item for item in list(vis) for i in range(dim)])
      vis = vis[dim_to_use]
      dim_to_use = dim_to_use[vis]
 
  for key in data.keys():
    data[ key ] = data[ key ][ :, dim_to_use ]  

  return data, dim_to_use


def expand(data,dim_to_use,dim):
    
    T = data.shape[0]
    D = dim
    orig_data = np.zeros((T, D), dtype=np.float32)
    orig_data[:,dim_to_use] = data
    
    return orig_data


def filter_data(poses, window=5, order=2):
    '''
    Filter time series using Savitzky-Golay filter
    
    Args
        poses: dictionary with poses
        window: int window of filter function (odd)
        order: int order of filter
        
    Output
        poses: dictionary with filtered poses
    '''
        
    poses_smooth = np.zeros_like(poses)
    for j in range(poses_smooth.shape[1]):
        poses_smooth[:,j] = scs.savgol_filter(poses[:,j], window, order) 
        
    return poses_smooth


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def lr_decay(optimizer, step, lr, decay_step, gamma):
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def world_to_camera(P, R, T):
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
  
    P_rot =  np.matmul(R, P.T).T + T
  
    return np.reshape( P_rot, [-1, ndim] )


def camera_to_world( data, cam_par, cam ):
    """
    Project 3d poses using camera parameters
    
    Args
        poses_set: dictionary with 3d poses
        cams: dictionary with camera parameters
        cam_ids: camera_ids to consider
    Returns
        transf: dictionary with 3d poses or 2d poses if projection is True
    """

    ndim = data.shape[1]
    R, T, _, _, _ = cam_par[cam]
        
    Pcam = np.reshape(data, [-1, 3]).copy()
    Pcam -= T
    Pworld = np.matmul(np.linalg.inv(R), Pcam.T).T
    
    return np.reshape( Pworld, [-1, ndim] )


def project_to_camera(P, intr):
    
    ndim = P.shape[1]
    P = np.reshape(P, [-1, 3])  
    proj = np.squeeze(np.matmul(intr, P[:,:,np.newaxis]))
    proj = proj / proj[:, [2]]
    proj = proj[:, :2]
  
    return np.reshape( proj, [-1, int(ndim/3*2)] )


def XY_coord( poses_set):
    """
    Project 3d poses to XY coord
    """
    t2d = {}

    for key in poses_set.keys():
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

    for key in poses_set.keys():
        t3d = poses_set[ key ]

        ndim = t3d.shape[1]
        Z = np.reshape(t3d, [-1, 3])
        Z = Z[:,2]
        t1d[ key ] = np.reshape( Z, [-1, ndim//3] )

    return t1d


def flip_LR(data):
    cols = list(data.columns)
    half = int(len(cols)/2)
    tmp = data.loc[:,cols[:half]].values
    data.loc[:,cols[:half]] = data.loc[:,cols[half:]].values
    data.loc[:,cols[half:]] = tmp
    
    return data
    
        
def get_epochs(data):
    data_idx = list(data.index)
    epochs = []
    for k, g in groupby(enumerate(data_idx), lambda ix : ix[0] - ix[1]):
        epochs.append(list(map(itemgetter(1), g)))
        
    return epochs


def select_best_data(bottom, side, th1, th2, leg_tips):
    
    #select those frames with high confidence ventral view (for lifting)
    bottom_lk = bottom.loc[:,(leg_tips,'likelihood')]
    mask = (bottom_lk>th1).sum(1)==6
    bottom = bottom[mask].dropna()
    side = side[mask].dropna()
        
    #find high confidence and low discrepancy keypoints in each frame
    likelihood = side.loc[:,(slice(None),'likelihood')]
    discrepancy = np.abs(bottom.loc[:,(slice(None),'x')].values - side.loc[:,(slice(None),'x')].values)
    good_keypts = (likelihood>th1) & (discrepancy<th2)
    good_keypts = good_keypts.droplevel(1,axis=1) 
    
    assert side.shape[0]==bottom.shape[0], 'Number of rows must match in filtered data!'
    
    return bottom, side, good_keypts


def read_crop_pos(file):
    
    assert os.path.exists(file), 'File does not exist: %s' % file
    f=open(file, "r")
    contents =f.readlines()
    im_file = []
    x_pos = []
    for i in range(4,len(contents)):
        line = contents[i][:-1].split(' ')
        im_file.append(line[0])
        x_pos.append(line[1])
        
    return im_file, x_pos


def read_convergence_info(file):
    f=open(file, "r")
    contents=f.readlines()
    epoch, lr, loss_train, loss_test, err_test = [], [], [], [], []
    for i in range(1,len(contents)):
        line = contents[i][:-1].split('\t')
        epoch.append(float(line[0]))
        lr.append(float(line[1]))
        loss_train.append(float(line[2]))
        loss_test.append(float(line[3]))
        err_test.append(float(line[4]))
        
    return epoch, lr, loss_train, loss_test, err_test


def plot_3d_graph(G, pos, ax, color_edge=None, style=None, good_keypts=None):
    
    for i, j in enumerate(G.edges()):
            
        if good_keypts is not None:
            if (good_keypts[j[0]]==0) | (good_keypts[j[1]]==0):
                continue
            
        #coordinates
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        #edge color
        if color_edge is not None:
            c = color_edge[j[0]]
        else:
            c = 'k'
            
        #edge style
        if style is None:
            style = '-'

        #plot           
        ax.plot(x, y, z, style, c=c, alpha=1.0, linewidth=2) 
        
        
def plot_skeleton(G, x, y, color_edge,  ax=None, good_keypts=None):
           
    for i, j in enumerate(G.edges()): 
        if good_keypts is not None:
            if (good_keypts[j[0]]==0) | (good_keypts[j[1]]==0):
                continue   
       
        u = np.array((x[j[0]], x[j[1]]))
        v = np.array((y[j[0]], y[j[1]]))
        if ax is not None:
            ax.plot(u, v, c=color_edge[j[0]], alpha=1.0, linewidth = 2)
        else:
            plt.plot(u, v, c=color_edge[j[0]], alpha=1.0, linewidth = 2)  
        
        
def plot_trailing_points(pos,thist,ax):
    alphas = np.linspace(0.1, 1, thist)
    rgba_colors = np.zeros((thist,4))
    rgba_colors[:,[0,1,2]] = 0.8
    rgba_colors[:, 3] = alphas
    for j in range(pos.shape[0]):
        ax.scatter(pos[j,0,:], pos[j,1,:], pos[j,2,:], '-o', color=rgba_colors)
        for i in range(thist-1):
            if i<thist:
                ax.plot(pos[j,0,i:i+2], pos[j,1,i:i+2], pos[j,2,i:i+2], '-o', c=rgba_colors[i,:])