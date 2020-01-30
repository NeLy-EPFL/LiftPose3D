from os import listdir
from os.path import isfile, isdir, join
from os import mkdir
import sys
from tqdm import tqdm

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2

DEBUG = False
IMG_PREV = None

args = sys.argv
data_dir = args[1]
if not data_dir.endswith("/") : data_dir += "/"

imgs_dir_spl = data_dir.split("/")
post_top = "_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]
post_side = "_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]

top_dd = data_dir+"top_view"+ post_top +"/"
side_dd = data_dir+"side_view"+ post_side +"/"
fcrop_loc_name = "crop_location"+ post_top +".txt"

if not isdir(top_dd):
    mkdir(top_dd)
if not isdir(side_dd):
    mkdir(side_dd)

threshold = 45
DIST_TH = 50
border_width = 105
bbox_width = 550
horiz_crop = 440

def _remove_borders(img, bwidth):
    img = img[:, bwidth:-bwidth]

    return img

def _separate_top_side_flies(img, th):
    fl_img = np.mean(img, axis=0)
    '''
    if DEBUG:
        fig = plt.figure(figsize=(5,3))
        plt.plot(fl_img)
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)
    '''
    fl_img_comp = fl_img > th
    left_vert_crop = np.argmax(fl_img_comp)
    right_vert_crop = len(fl_img) - np.argmax(fl_img_comp[::-1])

    bbox_dim = right_vert_crop - left_vert_crop
    mean_bbox_pad = (bbox_width - bbox_dim) / 2
    left_bbox_pad = int(np.floor(mean_bbox_pad))
    right_bbox_pad = int(np.ceil(mean_bbox_pad))
    print(left_bbox_pad)

    return img[:horiz_crop, left_vert_crop-left_bbox_pad : right_vert_crop+right_bbox_pad ],\
           img[horiz_crop:, left_vert_crop-left_bbox_pad : right_vert_crop+right_bbox_pad ],\
           left_vert_crop

def _get_orientation(contour, img):
    sz = len(contour)
    img_pts = np.empty((sz, 2), dtype=np.float64)
    img_pts[:,0] = contour[:,0,0]
    img_pts[:,1] = contour[:,0,1]

    # PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(img_pts, mean)

    angle = atan2(eigenvectors[0,1], eigenvectors[0,0])
    return angle

def _find_orientation(img):
    #_, img_th = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_th = img.copy()
    img_th[img_th < 130] = 0 # was 140
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1 : return None
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area > 10000:
            break
    contour = c
    # Find the orientation of each shape
    angle = np.degrees(_get_orientation(contour, img))

    if DEBUG:
        img_debug = img.copy()
        cv2.drawContours(img_debug, contours, i, (255, 0, 0), 2)
        cv2.putText(img_debug, "%.2f degree"%angle, (10,horiz_crop-50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("top", img_debug)
        cv2.waitKey(1)
    return angle

def _save_top_side_images(top_img, side_img, orientation, dd, img_name):
    img_name = img_name.split(".jpg")[0]

    cv2.imwrite(top_dd + img_name + "_%.2f"%orientation + ".jpg", top_img)
    cv2.imwrite(side_dd + img_name + "_%.2f"%orientation + ".jpg", side_img)

if __name__ == '__main__':
    print(f"\n[*] reading images name from {data_dir:s}")
    img_names = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith(".jpg")]
    img_names.sort()
    imgs_len = len(img_names)
    img_prev = None
    n_skip_dist = 0
    n_skip_out = 0
    print(f"[*] splitting the images into top and side views\n")
    print(top_dd, side_dd)

    fcrop_loc = open(data_dir+fcrop_loc_name, 'w')
    print(fcrop_loc_name)
    fcrop_loc.write("border_width = "+str(border_width)+"\n"+\
                    "threshold = "+str(threshold)+"\n"+\
                    "bbox_width = "+str(bbox_width)+"\n"+\
                    "horiz_crop = "+str(horiz_crop)+"\n")
    for counter in tqdm(range(imgs_len)):
        img_name = img_names[counter]

        img = cv2.imread(data_dir + img_name)
        
        # Convert to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Remove border with very bright light
        img = _remove_borders(img, border_width)
        img_dist = np.copy(img)
        #img_dist[img_dist < 60] = 0
        #img_bin = (img > 60).astype(float)
        
        if np.any(IMG_PREV == None):
            IMG_PREV = img_dist
            dist = 999
        else:
            dist = np.mean((img_dist - IMG_PREV)**2)
            if dist < DIST_TH:
                n_skip_dist += 1
                continue
            IMG_PREV = img_dist
        
        if DEBUG:
            cv2.putText(img_dist, "distance %.5f"%dist, (10,img_dist.shape[0]-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("row", img_dist)
            cv2.waitKey(2)
        
        top_img, side_img, left_vert_crop = _separate_top_side_flies(img, threshold)
        if top_img.shape[1] != bbox_width or side_img.shape[1] != bbox_width:
            if DEBUG:
                cv2.imshow("outlier", img)
                cv2.waitKey(1)
            n_skip_out += 1
            continue 
        
        orientation = _find_orientation(top_img)
        if orientation == None : continue
        if DEBUG:
            cv2.imshow("side", side_img)
            cv2.waitKey(1)

        if not DEBUG:
            fcrop_loc.write(img_name+" "+str(left_vert_crop)+"\n")
            _save_top_side_images(top_img, side_img, orientation, data_dir, img_name)
    fcrop_loc.close()
    print(f"\n[*] skipped {n_skip_dist:n}, {n_skip_out:n} frames because the fly was doing nothing or because it was an outlier")
    print("\n[+] done\n")

    with open(data_dir+"info.txt", 'w') as info:
        info.write("n_skip_dist: %d"%n_skip_dist)
        info.write("\nn_skip_out: %d"%n_skip_out)
