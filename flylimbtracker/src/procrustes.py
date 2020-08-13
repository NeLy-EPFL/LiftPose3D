import numpy as np
import cv2
from math import atan2
from src.utils import read_crop_pos
import os
#from scipy import ndimage

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
  """
  A port of MATLAB's `procrustes` function to Numpy.
  Adapted from http://stackoverflow.com/a/18927641/1884420
  Args
    X: array NxM of targets, with N number of points and M point dimensionality
    Y: array NxM of inputs
    compute_optimal_scale: whether we compute optimal scale or force it to be 1
  Returns:
    d: squared error after transformation
    Z: transformed Y
    T: computed rotation
    b: scaling
    c: translation
  """

  muX = X.mean(0)
  muY = Y.mean(0)

  X0 = X - muX
  Y0 = Y - muY

  ssX = (X0**2.).sum()
  ssY = (Y0**2.).sum()

  # centred Frobenius norm
  normX = np.sqrt(ssX)
  normY = np.sqrt(ssY)

  # scale to equal (unit) norm
  X0 = X0 / normX
  Y0 = Y0 / normY

  # optimum rotation matrix of Y
  A = np.dot(X0.T, Y0)
  U,s,Vt = np.linalg.svd(A,full_matrices=False)
  V = Vt.T
  T = np.dot(V, U.T)

  # Make sure we have a rotation
  detT = np.linalg.det(T)
  V[:,-1] *= np.sign( detT )
  s[-1]   *= np.sign( detT )
  T = np.dot(V, U.T)

  traceTA = s.sum()

  if compute_optimal_scale:  # Compute optimum scaling of Y.
    b = traceTA * normX / normY
    d = 1 - traceTA**2
    Z = normX*traceTA*np.dot(Y0, T) + muX
  else:  # If no scaling allowed
    b = 1
    d = 1 + ssY/ssX - 2 * traceTA * normY / normX
    Z = normY*np.dot(Y0, T) + muX

  c = muX - b*np.dot(muY, T)

  return d, Z, T, b, c


def orientation(img):
    img_th = img.copy()
    img_th[img_th < 130] = 0 # was 140
    
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1 : return None
    for i, contour in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(contour)
        # Ignore contours that are too small or too large
        if area > 10000:
            break

    # Find the orientation of each shape
    img_pts = np.empty((len(contour), 2), dtype=np.float64)
    img_pts[:,0], img_pts[:,1] = contour[:,0,0], contour[:,0,1]

    # PCA analysis
    mean = np.empty((0))
    _, eigenvectors, _ = cv2.PCACompute2(img_pts, mean)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    
    return angle


def center_and_align(pts2d, angle, c):
    '''rotate align data'''
    
    idx = pts2d.name
    angle = angle[idx]
    
    #rotate points
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array(((cos, -sin), (sin, cos)))    
    tmp = pts2d.to_numpy().reshape(-1, 2)
    tmp = np.matmul(tmp-c,R) + c   
    pts2d.iloc[:] = tmp.reshape(-1,tmp.shape[0]*2).flatten()
        
    return pts2d


def procrustes_on_epochs(data, epochs):
    xy = data.loc[:,(slice(None),['x','y'])].copy()
    for e in epochs:
        for step in range(1,len(e)):
            X = xy.loc[e[step]-1,:]
            Xtransf = xy.loc[e[step],:]
                
            X = X.to_numpy().reshape([-1, 2])
            Xtransf = Xtransf.to_numpy().reshape([-1, 2])
                
            _, _, T, _, c = compute_similarity_transform(Xtransf, X)
            Xtransf = (T@Xtransf.T).T

            xy.loc[e[step],:] = Xtransf.reshape(-1, 60).flatten()
                
    data.loc[:,(slice(None),['x','y'])] = xy
    
    return data


def get_orientation(path_crop_pos, path_img, index, flip_idx):
    im_file, _ = read_crop_pos(path_crop_pos)
    
    angle_old = []
    for i, idx in enumerate(index):
        assert os.path.exists(path_img + im_file[idx]), 'File does not exist: %s' % path_img + im_file[idx]
        im_crop = cv2.imread(path_img + im_file[idx],0)
        
        #get orientation and centre
        angle = orientation(im_crop)
        c = np.array(im_crop.shape)/2
        
        if angle_old==[]:
            angle_old = [angle]
        else:
            if (np.abs(angle_old[-1] - angle)>np.pi/2) & (not flip_idx[i]):
                angle += np.pi
            angle_old.append(angle)
    
    return angle_old, c