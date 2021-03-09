import numpy as np
import os
import glob
import pickle


def load_3D(path, par=None, cam_id=None, subjects="all", actions="all"):
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

    data, cam_par, good_keypts = {}, {}, {}
    for s in subjects:
        for a in actions:
            fname = fnames.copy()

            # select subjects
            if s != "all":
                fname = [file for file in fname if str(s) in file]
            if a != "all":
                fname = [file for file in fname if a in file]

            assert len(fname) != 0, "No files found. Check path!"

            for fname_ in fname:
                f = os.path.basename(fname_)[
                    :-4
                ]  # [:-4] is to get rid of .pkl extension

                # load
                poses = pickle.load(open(fname_, "rb"))
                poses3d = poses["points3d"][:899]
                
                for c in cam_id:
                    k = (s, a, f + ".cam_" + str(c))
                    ind = np.arange(19) if c < 3 else np.arange(19,38)
                    data[k] = poses3d
                    cam_par[k] = poses[c]
                    good_keypts[k] = np.zeros_like(data[k], dtype=bool)
                    good_keypts[k][:,ind] = True

    return data, good_keypts, cam_par


def load_2D(path, par=None, cam_id=None, subjects="all", actions="all"):
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
                f = os.path.basename(fname_)

                poses = pickle.load(open(fname_, "rb"))
                poses2d = poses["points2d"]

                for c in cam_id:
                    #ind = np.arange(19) if c < 3 else np.arange(19,38)
                    data[(subject, action, f[:-4] + ".cam_" + str(c))] = poses2d[c][:899]

    return data