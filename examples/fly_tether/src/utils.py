import numpy as np
import scipy.signal as scs


def get_coords_in_dim(targets, dim):
    """
    Get keypoint indices in spatial dimension 'dim'
    
    Args
        targets: list of lists of keypoints to be converted
        dim: spatial dimension of data (1, 2 or 3)
        
    Returns
        dim_to_use: list of keypoint indices in dimension dim
    """
    
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


def anchor_to_root(poses, roots, target_sets, dim):
    """
    Center points in targset sets around roots
    
    Args
        poses: dictionary of experiments each with array of size n_frames x n_dimensions
        roots: list of dimensions to be pulled to the origin
        target_sets: list of lists of indexes that are computer relative to respective roots
        dim: spatial dimension of data (1, 2 or 3)
  
    Returns
        poses: dictionary of anchored poses
        offset: offset of each root from origin
    """
    
    assert len(target_sets)==len(roots), 'We need the same # of roots as target sets!'
  
    offset = {}
    for k in poses.keys():
        offset[k] = np.zeros_like(poses[k])
        for i, root in enumerate(roots):
            for j in [root]+target_sets[i]:
                offset[k][:, dim*j:dim*(j+1)] += poses[k][:, dim*root:dim*(root+1)]

    for k in poses.keys():
        poses[k] -= offset[k]
      
    return poses, offset


def remove_roots(data, targets, n_dim, vis=None):
    """
    Normalizes a dictionary of poses
  
    Args
        data: dictionary of experiments each with array of size n_frames x n_dimensions
        targets: list of list of dimensions to be considered
        n_dim: number of spatial dimensions (e.g., 1,2 or 3)
        
    Returns
        data: dictionary of experiments with roots removed
        dim_to_use: list of dimensions in use for lifting
    """        
    
    dim_to_use = get_coords_in_dim(targets, n_dim)
 
    for key in data.keys():
        if vis is not None:
            data[ key ] = data[ key ][ :, vis ]  
        data[ key ] = data[ key ][ :, dim_to_use ]  

    return data, dim_to_use


def add_roots(data, dim_to_use, n_dim):
    """
    Add back the root dimensions
    
    Args
        data: array of size n_frames x (n_dim-n_roots)
        dim_to_use: list of indices of dimenions that are not roots 
        n_dim: integer number of dimensions including roots
    
    Returns
        orig_data: array of size n_frames x n_dim
    """
    
    T = data.shape[0]
    D = n_dim
    orig_data = np.zeros((T, D), dtype=np.float32)
    orig_data[:, dim_to_use] = data
    
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
    """
    Object to compute statistics during optimization
    """
    
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
    """
    Set learning rate
    
    Args
        optimizer: optimizer object
        step: optimization step
        decay_step: # steps of lr decay
        gamma: learning rate gain factor
        
    Returns
        lr: learning rate
    """
    
    lr = lr * gamma ** (step/decay_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
    return lr


def read_convergence_info(file):
    """
    Read the convergence file
    
    Args
        file: string containing filename (e.g.,log_train.txt)
        
    Returns
        epoch: list of optimization epochs
        lr: list of learning rates
        loss_train: list of RMSE training losses
        loss_test: list of RMSE test losses
        err_test: list of absolute test errors
    """
    
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