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
                dimensions = [i for i in range(30) if i not in [0,5,10,15,20,25]]  
                poses3d = poses['points3d'][:, dimensions, :]

                #collect data
                seqname = os.path.basename( fname_ )  
                data[ (subject, action, seqname[:-4]) ] = np.copy(poses3d) #[:-4] is to get rid of .pkl extension
                
                if 'good_keypts' in poses.keys():
                    kp = poses['good_keypts'][:,dimensions,None]
                    good_keypts[ (subject, action, seqname[:-4]) ] = np.tile(kp, reps=(1,1,3))
                
    #sort
    data = dict(sorted(data.items()))
    good_keypts = dict(sorted(good_keypts.items()))
    cam_par = dict(sorted(cam_par.items()))

    return data, good_keypts, cam_par