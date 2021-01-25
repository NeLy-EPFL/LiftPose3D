import numpy as np
import os
import glob
import pickle


def load_3D( path, par=None, cam_id=None, subjects='all', actions='all', verbose=False):
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
    for s in subjects:
        for a in actions:
            
            fname = fnames.copy()
            
            #select subjects 
            if s!='all':
                fname = [file for file in fname if str(s) + '_' in file]   
                
            if a!='all':
                fname = [file for file in fname if a + '_' in file]
            
            assert len(fname)!=0, 'No files found. Check path!'
      
            for fname_ in fname:
                
                f = os.path.basename( fname_ )[:-4]#[:-4] is to get rid of .pkl extension
        
                #load
                poses = pickle.load(open(fname_, "rb"))
                poses3d = poses['points3d']
                poses3d -= np.mean(poses3d, axis=1, keepdims=True)
                
                #only take data in a specified interval
                if (par is not None) and ('interval' in par.keys()):
                    frames = np.arange(par['interval'][0], par['interval'][1])
                    poses3d = poses3d[frames, :, :] #only load the stimulation interval
                    
                #remove specified dimensions
                if (par is not None) and ('dims_to_exclude' in par.keys()):
                    dimensions = [i for i in range(par['ndims']) if i not in par['dims_to_exclude']]   
                    try:
                        poses3d = poses3d[:, dimensions, :]
                        for i, c in enumerate(cam_id):
                            poses[c]['vis'] = poses[c]['vis'][dimensions]
                    except:
                        if verbose:
                            print('Obsolete dimensions cannot be removed. Continuing.')
                    
                if 'good_keypts' in poses.keys():
                    good_keypts[ (s, a, f)] = poses['good_keypts']
                    
                if cam_id is not None:
                    try:
                        cam_par[(s, a, f)] = {c: poses[c] for c in cam_id}
                    except:
                        cam_par[(s, a, f)] = None
                    
                #reshape data
                poses3d = np.reshape(poses3d, 
                                    (poses3d.shape[0], poses3d.shape[1]*poses3d.shape[2]))
                data[ (s, a, f) ] = poses3d 
                
    #sort
    data = dict(sorted(data.items()))
    good_keypts = dict(sorted(good_keypts.items()))
    cam_par = dict(sorted(cam_par.items()))

    return data, good_keypts, cam_par


def load_2D(path, par=None, cam_id=None, subjects='all', actions='all',verbose=False):
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
          
                f = os.path.basename( fname_ )  
        
                poses = pickle.load(open(fname_, "rb"))
                poses2d = poses['points2d']
                
                #only take data in a specified interval
                if (par is not None) and ('interval' in par.keys()):
                    frames = np.arange(par['interval'][0], par['interval'][1])
                    poses2d = poses2d[:,frames,:,:]
                    
                #remove specified dimensions
                if (par is not None) and ('dims_to_exclude' in par.keys()):
                    dimensions = [i for i in range(par['ndims']) if i not in par['dims_to_exclude']]      
                    try:
                        poses2d = poses2d[:,:,dimensions,:]
                    except:
                        if verbose:
                            print('Obsolete dimensions cannot be removed. Continuing.')
                    
                for c in cam_id:
                    poses_cam = poses2d[c,:,:,:]
                    poses_cam -= np.mean(poses_cam, axis=1, keepdims=True)
                    
                    if c in poses.keys():
                        ids = poses[c]['vis'][dimensions]
                        ids = np.array(ids).astype(bool)
                        poses_cam = poses_cam[:,ids,:]
                    else:
                        ids = [1]*poses2d.shape[2]

                    poses_cam = np.reshape(poses_cam, 
                            (poses2d.shape[1], sum(ids)*poses2d.shape[3]))   
        
                    data[ (subject, action, f[:-4] + '.cam_' + str(c)) ] = poses_cam
            
    #sort
    data = dict(sorted(data.items()))

    return data