import numpy as np

def normalize_data(data, data_mean, data_std, targets, dim ):
  """
  Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """

  dim_to_use = get_coords_in_dim(targets, dim)

  for key in data.keys():
    data[ key ] -= data_mean
    data[ key ] /= data_std
    data[ key ] = data[ key ][ :, dim_to_use ]

  return data


def unNormalizeData(data, data_mean, data_std, dim_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """

  data *= data_std[dim_to_use]
  data += data_mean[dim_to_use]
  
  T = data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:, dim_to_use] = data
  
  return orig_data


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


def anchor(poses, 
           anchors = [0, 5, 10], 
           target_sets = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], 
           dim=3):
  """
  Center points in targset sets around anchors
  """
  
  for k in poses.keys():
      for i, anch in enumerate(anchors):
          for j in target_sets[i]:
              poses[k][:, dim*j:dim*j+dim] -= poses[k][:, dim*anch:dim*anch+dim]

  return poses


def de_anchor(old_poses, new_poses, 
           anchors = [0, 5, 10], 
           target_sets = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], 
           dim=3):
  """
  Center 3d points around root(s)
  """
  
  for k in old_poses.keys():
      for i, anch in enumerate(anchors):
          for j in target_sets[i]:
              new_poses[k][:, dim*j:dim*j+dim] += old_poses[k][:, dim*anch:dim*anch+dim]

  return new_poses