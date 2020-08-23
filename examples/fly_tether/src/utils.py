import numpy as np
import scipy.signal as scs

def normalization_stats(data):
  """
  Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    data: dictionary or np arrays or size frames x dimensions
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
  """

  complete_data = np.concatenate([v for k,v in data.items()], 0)
  print(complete_data.shape)
  data_mean = np.mean(complete_data, axis=0)
  data_std  =  np.std(complete_data, axis=0)
  
  return data_mean, data_std


def normalize_data(data, data_mean, data_std ):
  """
  Normalizes a dictionary of poses
  
  Args
    data: 
  Returns
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


def collapse(data, target_sets, dim ):
  """
  Normalizes a dictionary of poses
  """
  dim_to_use = get_coords_in_dim(target_sets, dim)
 
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


def ms_error(tar,out,dim):
    
    abserr = np.abs(out - tar)

    n_pts = abserr.shape[1]//dim
    e = np.zeros((abserr.shape[0], n_pts))
    for k in range(n_pts):
        e[:, k] = np.mean(abserr[:, dim*k:dim*(k + 1)], axis=1)
    
    return e


def world_to_camera(Pworld, par):
  """
  Rotate/translate 3d poses to camera viewpoint

  Args
    P: Nx3 points in world coordinates
    par: dictionary of camera parameters
  Returns
    transf: Nx2 points on camera
  """
      
  ndim = Pworld.shape[1]
  
  assert len(Pworld.shape) == 2
      
  if type(par) is list:
      assert len(par) == Pworld.shape[0]
      
      Pcam = np.zeros_like(Pworld)
      for i in range(len(par)):   
          tmp = np.reshape(Pworld[i,:], [-1, 3]).copy()
          tmp =  np.matmul(par[i]['R'], tmp.T).T 
          tmp += par[i]['tvec']
          Pcam[i,:] = np.reshape( tmp, [-1, ndim] )
  else:
      Pcam = np.reshape(Pworld, [-1, 3]).copy()
      Pcam = np.matmul(par['R'], Pcam.T).T
      Pcam += par['tvec']
      Pcam = np.reshape( Pcam, [-1, ndim] )
  
  return Pcam


def camera_to_world(Pcam, par):
  """
  Project 3d poses using camera parameters

  Args
    Pcam: dictionary with 3d poses
    cams: dictionary with camera parameters
    cam_ids: camera_ids to consider
  Returns
    transf: dictionary with 3d poses or 2d poses if projection is True
  """

  ndim = Pcam.shape[1]
  
  if type(par) is list:
      assert len(par) == Pcam.shape[0]
     
      Pworld = np.zeros_like(Pcam)
      for i in range(len(par)):  
          tmp = np.reshape(Pcam[i,:], [-1, 3]).copy()
          tmp -= par[i]['tvec']
          tmp = np.matmul(np.linalg.inv(par[i]['R']), tmp.T).T
          Pworld[i,:] = np.reshape( tmp, [-1, ndim] )
  else:
      Pworld = np.reshape(Pcam, [-1, 3]).copy()
      Pworld -= par['tvec']
      Pworld = np.matmul(np.linalg.inv(par['R']), Pworld.T).T
      Pworld = np.reshape( Pworld, [-1, ndim] )
  
  return Pworld


def project_to_camera(P, par):
    
  ndim = P.shape[1]
  
  if type(par) is list:
      assert len(par) == P.shape[0]
      
      Pproj = np.zeros_like(P)
      for i in range(len(par)):     
          tmp = np.reshape(P[i,:], [-1, 3])  
          tmp = np.squeeze(np.matmul(par['intr'], tmp[:,:,np.newaxis]))
          tmp /= tmp[:, [2]]
          tmp = tmp[:, :2]
          Pproj[i,:] = np.reshape( tmp, [-1, int(ndim/3*2)] )
  else:
      P = np.reshape(P, [-1, 3]).copy()
      Pproj = np.squeeze(np.matmul(par['intr'], P[:,:,np.newaxis]))
      Pproj /= Pproj[:, [2]]
      Pproj = Pproj[:, :2]
  
  return Pproj


def plot_3d_graph(G, pos, ax, color_edge=None, style=None, good_keypts=None):
    
    for i, j in enumerate(reversed(list(G.edges()))):
            
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