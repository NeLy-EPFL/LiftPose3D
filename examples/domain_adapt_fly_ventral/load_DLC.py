import numpy as np
import pandas as pd


leg_joints = ['body-coxa front L', 'femur-tibia front L', 'tibia-tarsus front L', 'tarsus tip front L',
              'body-coxa mid L', 'femur-tibia mid L', 'tibia-tarsus mid L', 'tarsus tip mid L',
              'body-coxa back L', 'femur-tibia back L', 'tibia-tarsus back L', 'tarsus tip back L',
              'body-coxa front R', 'femur-tibia front R', 'tibia-tarsus front R', 'tarsus tip front R',
              'body-coxa mid R', 'femur-tibia mid R', 'tibia-tarsus mid R', 'tarsus tip mid R',
              'body-coxa back R', 'femur-tibia back R', 'tibia-tarsus back R', 'tarsus tip back R']

def load_2D(data_dir, experiments, scorer):
    """
    Load 2D data

    Args
        path: string. Directory where to load the data from
        experiments: experiment names
        scorer: DLC network identifier (scorer)
    Returns
        poses: dictionary of poses
        good_keypts: dictionary of high quality (>0.95 confidence) keypoints
    """

    poses, good_keypts = {}, {}
    for i, exp in enumerate(experiments):
        data = pd.read_hdf(data_dir + exp + scorer + '.h5') #load data
        data = data.droplevel('scorer',axis=1) #drop scorer column label
    
        xy = data.loc[:,(slice(None),['x','y'])].to_numpy().copy()
        likelihood = data.loc[:,(slice(None),['likelihood'])].to_numpy().copy()
    
        data.loc[:,(slice(None),['x','y'])] = xy
    
        #select only leg joints
        data_np = data.loc[:,(leg_joints,['x','y'])].to_numpy()
    
        #save in DF3D format
        xy = np.stack((data_np[:,::2], data_np[:,1::2]), axis=2)
        poses[exp] = xy
        good_keypts[exp] = likelihood>0.95

    return poses, good_keypts
