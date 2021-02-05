import numpy as np
from scipy.spatial.transform import Rotation as Rot
from liftpose.vision_3d import world_to_camera, project_to_camera
from liftpose.preprocess import anchor_to_root, remove_roots
import torch

def random_project(outputs, 
                   eangle,
                   axsorder, 
                   mean_2d, 
                   std_2d, 
                   mean_3d, 
                   std_3d, 
                   tvec, 
                   intr, 
                   roots, 
                   target_sets):
    
    #obtain random rotation matrices
    if len(eangle)==2: #in case we have 2 cameras with different ranges
        lr = np.random.binomial(1,0.5)
        if lr:
            eangle = eangle[0]
        else:
            eangle = eangle[1]
    else:
        eangle = eangle[0]
                    
    a = np.random.uniform(low=eangle[0][0], high=eangle[0][1])
    b = np.random.uniform(low=eangle[1][0], high=eangle[1][1])
    c = np.random.uniform(low=eangle[2][0], high=eangle[2][1])
    R = Rot.from_euler(axsorder, [[a, b, c]], degrees=True).as_matrix()
                
    #do random projection
    outputs = outputs[None,:]
    outputs = world_to_camera(outputs, R, tvec)
    inputs = project_to_camera(outputs, intr)
                
    # anchor points to body-coxa (to predict legjoints wrt body-coxas)
    inputs, _ = anchor_to_root( {'inputs': inputs}, roots, target_sets, 2)
    outputs, _ = anchor_to_root( {'outputs': outputs}, roots, target_sets, 3)
                
    inputs = inputs['inputs']
    outputs = outputs['outputs']
                
    # Standardize each dimension independently
    np.seterr(divide='ignore', invalid='ignore')
    inputs -= mean_2d
    inputs /= std_2d
    outputs -= mean_3d
    outputs /= std_3d
      
    #remove roots
    inputs, _ = remove_roots({'inputs': inputs}, target_sets, 2)
    outputs, _ = remove_roots({'outputs': outputs}, target_sets, 3)
                 
    inputs = inputs['inputs']
    outputs = outputs['outputs']
                
    #get torch tensors
    inputs = torch.from_numpy(inputs[0,:]).float()
    outputs = torch.from_numpy(outputs[0,:]).float()
    
    return inputs, outputs


def add_noise(inputs, noise_amplitude, std_2d, targets_2d):
    std = std_2d[targets_2d]
    inputs += torch.from_numpy(
                np.random.normal(0, noise_amplitude / std, size=inputs.shape)
              ).float()
    
    return inputs
    
    
    