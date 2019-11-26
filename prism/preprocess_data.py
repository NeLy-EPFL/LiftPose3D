from os import listdir
from os.path import isfile, isdir, join
from os import mkdir
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2

DEBUG = True

args = sys.argv
data_dir = args[1]
if not data_dir.endswith("/") : data_dir += "/"

top_dd = data_dir + "top_view/"
side_dd = data_dir + "side_view/"
if not isdir(top_dd):
    mkdir(top_dd)
if not isdir(side_dd):
    mkdir(side_dd)

border_width = 60
threshold = 27
bbox_width = 600
horiz_crop = 400

def _remove_borders(img, bwidth):
    img = img[:, bwidth:-bwidth]

    return img

def _separate_top_side_flies(img, th):
    fl_img = np.mean(img, axis=0)

    #if DEBUG:
    #    fig = plt.figure(figsize=(5,3))
    #    plt.plot(fl_img)
    #    plt.show(block=False)
    #    plt.pause(2)
    #    plt.close(fig)

    fl_img_comp = fl_img > th
    #pick_idx = np.argmax(fl_img)
    left_vert_crop = np.argmax(fl_img_comp)
    right_vert_crop = len(fl_img) - np.argmax(fl_img_comp[::-1])

    bbox_dim = right_vert_crop - left_vert_crop
    mean_bbox_pad = (bbox_width - bbox_dim) / 2
    left_bbox_pad = int(np.floor(mean_bbox_pad))
    right_bbox_pad = int(np.ceil(mean_bbox_pad))

    return img[:horiz_crop, left_vert_crop-left_bbox_pad : right_vert_crop+right_bbox_pad ],\
           img[horiz_crop:, left_vert_crop-left_bbox_pad : right_vert_crop+right_bbox_pad ]

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
    _, img_th = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv2.contourArea(c)
        # Ignore contours that are too small or too large
        if area > 40000:
            break
    contour = c
    # Find the orientation of each shape
    angle = np.degrees(_get_orientation(contour, img))

    if DEBUG:
        img_debug = img.copy()
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
    img_names = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith(".jpg")]
    img_names.sort()
    #img_names = img_names[0::5]
    img_prev = None
    n_skip = 0
    for counter, img_name in enumerate(img_names):
        if counter % 100 == 0 : print(f"{counter:n} / {len(img_names):n}")

        img = cv2.imread(data_dir + img_name)
        
        # Convert to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Remove border with very bright light
        img = _remove_borders(img, border_width)
        
        img_dist = img.copy()
        img_dist[img_dist < 50] = 0
        if np.any(img_prev == None):
            img_prev = img_dist
            dist = 999
        else:
            dist = np.mean((img_dist - img_prev)**2)
            if dist < 6.57:
                n_skip += 1
                if DEBUG : print("skipped", n_skip)
                continue
            img_prev = img_dist

        if DEBUG:
            img_debug = img.copy()
            cv2.putText(img_debug, "distance %.5f"%dist, (10,img_debug.shape[0]-50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.imshow("row", img_debug)
            cv2.waitKey(1)

        top_img, side_img = _separate_top_side_flies(img, threshold)
        if top_img.shape[1] != bbox_width or side_img.shape[1] != bbox_width:
            if DEBUG:
                cv2.imshow("outlier", img)
                cv2.waitKey(1)
            continue 
        
        orientation = _find_orientation(top_img)
        _save_top_side_images(top_img, side_img, orientation, data_dir, img_name)
