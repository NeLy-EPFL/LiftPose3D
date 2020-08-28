import numpy as np
import cv2
#from math import atan2
from scipy import ndimage

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


def orientation(img, th=10, k=30):
    img_th = img.copy()
    
    #threshold
    _, img_thresh = cv2.threshold(img_th, th, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    flybody = get_largest_conncomp(img_thresh) #mask of fly
    flybody = cv2.morphologyEx(flybody, cv2.MORPH_OPEN, kernel) #chop legs off with kernel

    contour, _ = cv2.findContours(flybody, 1, 2)
    contour = max(contour, key=cv2.contourArea)
    
    ellipse = cv2.fitEllipse(contour)   
    cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
    angle = ellipse[2]+90
    
    h, w = img.shape
    M_tr = np.float32([[1,0,w/2-cx],[0,1,h/2-cy]])
    flybody = cv2.warpAffine(flybody,M_tr,(w,h))
    img = cv2.warpAffine(img,M_tr,(w,h))
    img = ndimage.rotate(img, angle, reshape=False)
    
    return angle, (cx,cy), img


def get_largest_conncomp(img):
    '''
    In a binary image, compute the biggest component (with connected components analysis)
    Input:
        img: binary image
    Output:   
        biggestComponent: binary image of biggest component in img
    '''
    output = cv2.connectedComponentsWithStats(img, 4, cv2.CV_32S)
    stats = np.transpose(output[2])
    sizes = stats[4]
    labelMax = np.where(sizes == np.amax(sizes[1:]))
    biggestComponent = np.zeros_like(img)
    biggestComponent[np.where(output[1] == labelMax[0])] = 255
    
    return biggestComponent


def center_and_align(pts2d, angle, shape, c):
    '''rotate align data'''
    
    pts2d = pts2d.copy()
    pts2d += shape
    pts2d -= c
    
    #rotate points
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array(((cos, -sin), (sin, cos)))    
    pts2d = np.matmul(pts2d-shape,R)+shape  
    
    if pts2d[0,0]>pts2d[-1,-1]:
        cos, sin = np.cos(np.pi), np.sin(np.pi)
        R = np.array(((cos, -sin), (sin, cos)))    
        pts2d = np.matmul(pts2d-shape,R)+shape
        angle += np.pi
        
    return pts2d, angle


def procrustes_on_keypoints(data, keypoints):
    xy = data.copy()
    template = xy[:,keypoints,:].mean(0)
    transf = {'R':[],'t':[]}
    for step in range(data.shape[0]):
        
        A = xy[step,keypoints,:]
        
        xytrans = xy[step,:,:]

        _, _, R,_, t = compute_similarity_transform(template, A)
        xytrans = np.dot(xytrans, R) + t
        transf['R'].append(R)
        transf['t'].append(t)

        xy[step,:,:] = xytrans
    
    return xy, transf


#def get_orientation(img_paths, index,th):
#    
#    angle_old = []
#    for i, idx in enumerate(index):
#        im_crop = cv2.imread(img_paths[idx],0)
#        
#        #get orientation and centre
#        angle = orientation(im_crop,th)
#        c = np.array(im_crop.shape)/2
#        
#        if angle_old==[]:
#            angle_old = [angle]
#        else:
#            if np.abs(angle_old[-1] - angle)>np.pi/2:
#                angle += np.pi
#            angle_old.append(angle)
#    
#    return angle_old, c