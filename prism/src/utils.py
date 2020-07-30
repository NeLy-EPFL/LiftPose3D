import numpy as np
import copy
import scipy.signal as scs
from operator import itemgetter
from itertools import groupby
import os

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
        
    for k in poses.keys():
        poses_smooth = np.zeros_like(poses[k])
        for j in range(poses_smooth.shape[1]):
                poses_smooth[:,j] = scs.savgol_filter(poses[k][:,j], window, order)
                
        poses[k] = poses_smooth
        
    return poses


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


def flip_LR(data):
    cols = list(data.columns)
    half = int(len(cols)/2)
    tmp = data.loc[:,cols[:half]].values
    data.loc[:,cols[:half]] = data.loc[:,cols[half:]].values
    data.loc[:,cols[half:]] = tmp
    
    return data


def orient_left(bottom, th1, flip_idx):
    #rotate flies pointing right    
    theta = np.radians(180)
    cos, sin = np.cos(theta), np.sin(theta)
    R = np.array(((cos, -sin), (sin, cos)))     
    
    if np.sum(flip_idx) != 0:
        tmp = bottom.loc[flip_idx,(slice(None),['x','y'])].to_numpy()
        tmp = np.reshape(tmp, [-1, 2])
        mu = tmp.mean(axis=0)
        tmp = np.matmul(tmp-mu,R)# + mu
        tmp = np.reshape( tmp, [-1, 60] )
        bottom.loc[flip_idx,(slice(None),['x','y'])] = tmp

    return bottom
    
        
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
    
    #check which way the flies are pointing
    side_R_lk = side.loc[:,(leg_tips[:3],'likelihood')] #high confidence on R joints means fly points right
    flip_idx = (side_R_lk>th1).sum(1)==3
    
    #find high confidence and low discrepancy keypoints in each frame
    likelihood = side.loc[:,(slice(None),'likelihood')]
    discrepancy = np.abs(bottom.loc[:,(slice(None),'x')].values - side.loc[:,(slice(None),'x')].values)
    good_keypts = (likelihood>th1) & (discrepancy<th2)
    good_keypts = good_keypts.droplevel(1,axis=1) 
    
    assert side.shape[0]==bottom.shape[0], 'Number of rows must match in filtered data!'
    
    return bottom, side, np.array(flip_idx), good_keypts


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


def plot_3d_graph(G, pos, ax, l, color_edge=None, style=None, good_keypts=None):
    
    for i, j in enumerate(G.edges()):
        try:
            l[i].remove()
        except:
            pass
            
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

        #udpate lines
#        if i not in l.keys():               
        l[i], = ax.plot(x, y, z, style, c=c, alpha=1.0, linewidth=2) 
#        else:
#            l[i].set_xdata(x)
#            l[i].set_ydata(y)
#            l[i].set_3d_properties(z, zdir='z')
    
    return l
