import numpy as np
import cv2
import os
from scipy import ndimage


def read_crop_pos(file):
    
    assert os.path.exists(file), 'File does not exist: %s' % file
    f=open(file, "r")
    contents =f.readlines()
    im_file = []
    x_pos = []
    for i in range(4,len(contents)):
        line = contents[i][:-1].split(' ')
        im_file.append(line[0])
        x_pos.append(line[1])
        
    return im_file, x_pos


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
    
    idx = pts2d.name
    angle = angle[idx]
    
    tmp = pts2d.to_numpy().reshape(-1, 2)
    tmp += shape//2
    tmp -= c[idx]
    
    #rotate points
    cos, sin = np.cos(angle), np.sin(angle)
    R = np.array(((cos, -sin), (sin, cos)))    
     
    rot = np.matmul(tmp-shape//2,R) + shape[::-1]//2
        
    if rot[0,0]>rot[-1,-1]:
        cos, sin = np.cos(angle+np.pi), np.sin(angle+np.pi)
        R = np.array(((cos, -sin), (sin, cos)))    
        rot = np.matmul(tmp-shape//2,R)+shape[::-1]//2
    
    pts2d.iloc[:] = rot.reshape(-1,rot.shape[0]*2).flatten()
        
    return pts2d


def get_orientation(path_crop_pos, path_img, index):
    im_file, _ = read_crop_pos(path_crop_pos)
    
    angles, imgs_reg, centers, imgs = [], [], [], []
    for i, idx in enumerate(index):
        assert os.path.exists(path_img + im_file[idx]), 'File does not exist: %s' % path_img + im_file[idx]
        im_crop = cv2.imread(path_img + im_file[idx],0)
              
        #get orientation and centre
        angle, c, img_rot = orientation(im_crop, 10)

        angles.append(angle)
        imgs_reg.append(img_rot)
        centers.append(c)
        imgs.append(im_crop)
        
    return angles, centers, imgs_reg, im_crop.shape