import numpy as np

def normalization_stats(data):
    """
    Computes mean and stdev
    
    Args
        data: dictionary containing data of all experiments
    Returns
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
    """

    complete_data = np.concatenate([v for k,v in data.items()], 0)
    mean = np.nanmean(complete_data, axis=0)
    std  = np.nanstd(complete_data, axis=0)
  
    return mean, std


def normalize(data, mean, std):
    """
    Normalizes a dictionary of poses
  
    Args
        data: dictionary containing data of all experiments
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
      
    Returns
        data: dictionary containing normalized data
    """
  
    np.seterr(divide='ignore', invalid='ignore')
 
    for key in data.keys():
        data[ key ] -= mean
        data[ key ] /= std

    return data


def unNormalize(data_norm, mean, std):
    """
    Un-normalizes a matrix whose mean has been substracted and that has been divided by
    standard deviation
  
    Args
        data: dictionary containing normalized data of all experiments
        mean: array with the mean of the data for all dimensions
        std: array with the stdev of the data for all dimensions
      
    Returns
        data: dictionary containing normalized data
    """
    
    data_norm *= std
    data_norm += mean
    
    return data_norm