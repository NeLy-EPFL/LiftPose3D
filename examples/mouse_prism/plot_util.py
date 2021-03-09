import numpy as np
def undistort(im,idx):
    shape = im.shape
    im_flat = im.flatten('F')
    im_flat = im_flat[idx]
    return np.reshape(im_flat,(shape[0], shape[1]),'F')

def xy_to_flat(x, y, n_rows, n_cols):
    return x * n_rows + y

def flat_to_xy(f, n_rows, n_cols):
    x = f // n_rows
    y = f % n_rows
    return y, x