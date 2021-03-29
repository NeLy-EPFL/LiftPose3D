import numpy as np
import cv2
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def undistort(im,idx):
    shape = im.shape
    im_flat = im.flatten('F')
    im_flat = im_flat[idx]
    return np.reshape(im_flat,(shape[0], shape[1]),'F')

def video_to_imgs(vid_path):
    '''Convert video to a list of images'''
    
    cap = cv2.VideoCapture(vid_path)         
    imgs = []            
    while True:
        flag, frame = cap.read()
        if flag:                            
            imgs.append(frame)       
        else:
            break        
    
    return imgs