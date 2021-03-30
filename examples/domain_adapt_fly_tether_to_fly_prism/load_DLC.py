import numpy as np
import os
import glob
import pickle


def load_3D(path, par=None, subjects="all", actions="all"):
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

    path = os.path.join(path, "*.pkl")
    fnames = glob.glob(path)

    data = {}
    for s in subjects:
        for a in actions:
            fname = fnames.copy()

            # select subjects
            if s != "all":
                fname = [file for file in fname if str(s) + '_' in file]
            if a != "all":
                fname = [file for file in fname if a in file]

            assert len(fname) != 0, "No files found. Check path!"

            for fname_ in fname:
                f = os.path.basename(fname_)[:-4]

                # load
                poses = pickle.load(open(fname_, "rb"))
                
                dimensions = [i for i in range(38) if i not in [15,16,17,18,34,35,36,37]]  
                poses3d = poses["points3d"][:899, dimensions, :]

                k = (s, a, f)
                data[k] = np.copy(poses3d)

    return data


def load_2D(path, par=None, subjects="all", actions="all"):
    """
    Load 2D data

    Args
        path: string. Directory where to load the data from,
        subjects: List of strings coding for strings in filename
        actions: List of strings coding for strings in filename
    Returns
        data: dictionary with keys k=(subject, action, filename)
    """

    path = os.path.join(path, "*.pkl")
    fnames = glob.glob(path)

    data = {}
    for subject in subjects:
        for action in actions:
            fname = fnames.copy()

            if subject != "all":
                fname = [file for file in fname if str(subject) in file]
            if action != "all":
                fname = [file for file in fname if action in file]

            assert len(fname) != 0, "No files found. Check path!"
            for fname_ in fname:
                f = os.path.basename(fname_)[:-4]

                poses = pickle.load(open(fname_, "rb"))
                poses2d = poses["points2d"]
                dimensions = [i for i in range(38) if i not in [15,16,17,18,34,35,36,37]]   

                    # ind = np.arange(0,15) if c < 3 else np.arange(19,19+15)
                data[(subject, action, f)] = poses2d[0][:899,dimensions,:]

    return data
