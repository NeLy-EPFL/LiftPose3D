import numpy as np
import os
import glob
import pickle


def load_3D( path, par=None, cam_id=None, subjects='all', actions='all' ):
    """
    Load 3D ground truth

    Args
        path: String. Path where to load the data from
        par: dictionary of parameters
        subjects: List of strings coding for strings in filename
        actions: List of strings coding for strings in filename
    Returns
        data: Dictionary with keys (subject, action, filename)
        good_keypts: Dictionary with keys (subject, action, filename)
        cam_par: Dictionary with keys (subject, action, filename)
    """
    
    path = os.path.join(path, '*.pkl')
    fnames = glob.glob( path )
    
    data, cam_par, good_keypts = {}, {}, {}
    for subject in subjects:
        for action in actions:
            
            fname = fnames.copy()
            
            #select files 
            if subject!='all':
                fname = [file for file in fname if str(subject) in file]   
                
            if action!='all':
                fname = [file for file in fname if action in file]
            
            assert len(fname)!=0, 'No files found. Check path!'
      
            for fname_ in fname:
        
                #load
                poses = pickle.load(open(fname_, "rb"))
                poses3d = poses['points3d']
                
                #only take data in a specified interval
                if (par is not None) and ('interval' in par.keys()):
                    frames = np.arange(par['interval'][0], par['interval'][1])
                    poses3d = poses3d[frames, :,:] #only load the stimulation interval
                    
                #remove specified dimensions
                if (par is not None) and ('dims_to_exclude' in par.keys()):
                    dimensions = [i for i in range(par['ndims']) if i not in par['dims_to_exclude']]   
                    poses3d = poses3d[:, dimensions,:]
                    
                #collect data
                seqname = os.path.basename( fname_ )  
                data[ (subject, action, seqname[:-4]) ] = poses3d #[:-4] is to get rid of .pkl extension
                
                if 'good_keypts' in poses.keys():
                    good_keypts[ (subject, action, seqname[:-4]) ] = poses['good_keypts'][:,:,None]
                    
                if cam_id is not None:
                    cam_par[(subject, action, seqname[:-4])] = [poses[cam_id] for i in range(poses3d.shape[0])]
                
    #sort
    data = dict(sorted(data.items()))
    good_keypts = dict(sorted(good_keypts.items()))
    cam_par = dict(sorted(cam_par.items()))

    return data, good_keypts, cam_par


def load_2D(path, par=None, cam_id=0, subjects='all', actions='all'):
    """
    Load 2D data
    
    Args
        path: string. Directory where to load the data from,
        subjects: List of strings coding for strings in filename
        actions: List of strings coding for strings in filename
    Returns
        data: dictionary with keys k=(subject, action, filename)
    """

    path = os.path.join(path, '*.pkl')
    fnames = glob.glob( path )

    data = {}
    for subject in subjects:
        for action in actions:
            
            fname = fnames.copy()
        
            if subject!='all':
                fname = [file for file in fname if str(subject) in file]   
                
            if action!='all':
                fname = [file for file in fname if action in file]   
                
            assert len(fname)!=0, 'No files found. Check path!'

            for fname_ in fname:
          
                seqname = os.path.basename( fname_ )  
        
                poses = pickle.load(open(fname_, "rb"))
                poses = poses['points2d']
                
                #only take data in a specified interval
                if (par is not None) and ('interval' in par.keys()):
                    frames = np.arange(par['interval'][0], par['interval'][1])
                    poses = poses[:,frames,:,:]
                    
                #remove specified dimensions
                if (par is not None) and ('dims_to_exclude' in par.keys()):
                    dimensions = [i for i in range(par['ndims']) if i not in par['dims_to_exclude']]      
                    poses = poses[:,:,dimensions,:]
                    
                #reshape data
                poses_cam = poses[cam_id,:,:,:]
        
                #collect data
                data[ (subject, action, seqname[:-4]) ] = poses_cam
            
    #sort
    data = dict(sorted(data.items()))

    return data