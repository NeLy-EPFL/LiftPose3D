from os import listdir
from os.path import isfile, isdir, join
from os import mkdir
import shutil
# import sys
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import atan2
from skimage.filters.rank import median
from skimage.morphology import disk

from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte

from scipy import ndimage
from skimage.util import pad

import os
import subprocess

def _remove_borders(img, bwidth1, bwidth2):
    img = img[:, bwidth1:-bwidth2]

    return img


def _separate_flies_vertically(img, th):
    # denoise with median filter
    # _img = median(img, disk(1))
    _img = img[horiz_crop_middle_1:horiz_crop_middle_2, :]
    fl_img = np.mean(_img, axis=0)

    if DEBUG:
        fig = plt.figure(figsize=(5, 3))
        plt.plot(fl_img)
        plt.show(block=False)
        plt.pause(2)
        plt.close(fig)

    fl_img_comp = fl_img > th 
    left_vert_crop = np.argmax(fl_img_comp)
    right_vert_crop = len(fl_img) - np.argmax(fl_img_comp[::-1])

    bbox_dim = right_vert_crop - left_vert_crop
    mean_bbox_pad = (bbox_width - bbox_dim) / 2
    left_bbox_pad = int(np.floor(mean_bbox_pad))
    right_bbox_pad = int(np.ceil(mean_bbox_pad))
    print(left_bbox_pad)

    # corrections when the fly is near both ends
    if (left_vert_crop - left_bbox_pad)<0:
        img = pad(img, pad_width=((0, 0), (-(left_vert_crop - left_bbox_pad), 0)), mode='constant')
        return img[horiz_crop_middle_1:horiz_crop_middle_2, 0: right_vert_crop + right_bbox_pad - (left_vert_crop - left_bbox_pad)], \
               img[horiz_crop_right_1:horiz_crop_right_2, 0: right_vert_crop + right_bbox_pad - (left_vert_crop - left_bbox_pad)], \
               img[horiz_crop_left_1:horiz_crop_left_2, 0: right_vert_crop + right_bbox_pad - (left_vert_crop - left_bbox_pad)], \
               0, \
               right_vert_crop + right_bbox_pad
    elif (right_vert_crop + right_bbox_pad)>img.shape[1]:
        img = pad(img, pad_width=((0, 0), (0, (right_vert_crop + right_bbox_pad)-img.shape[1])), mode='constant')
        return img[horiz_crop_middle_1:horiz_crop_middle_2, left_vert_crop - left_bbox_pad: img.shape[1]], \
               img[horiz_crop_right_1:horiz_crop_right_2, left_vert_crop - left_bbox_pad: img.shape[1]], \
               img[horiz_crop_left_1:horiz_crop_left_2, left_vert_crop - left_bbox_pad: img.shape[1]], \
               left_vert_crop - left_bbox_pad, \
               img.shape[1]

    return img[horiz_crop_middle_1:horiz_crop_middle_2, left_vert_crop - left_bbox_pad: right_vert_crop + right_bbox_pad], \
           img[horiz_crop_right_1:horiz_crop_right_2, left_vert_crop - left_bbox_pad: right_vert_crop + right_bbox_pad], \
           img[horiz_crop_left_1:horiz_crop_left_2, left_vert_crop - left_bbox_pad: right_vert_crop + right_bbox_pad], \
           left_vert_crop - left_bbox_pad, \
           right_vert_crop + right_bbox_pad


def _get_orientation(contour, img):
    sz = len(contour)
    img_pts = np.empty((sz, 2), dtype=np.float64)
    img_pts[:, 0] = contour[:, 0, 0]
    img_pts[:, 1] = contour[:, 0, 1]

    # PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(img_pts, mean)

    angle = atan2(eigenvectors[0, 1], eigenvectors[0, 0])
    return angle


def _find_orientation(img):
    # _, img_th = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_th = img.copy()
    img_th[img_th < 130] = 0  # was 140
    # Find all the contours in the thresholded image
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 1: return None
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
        cv2.putText(img_debug, "%.2f degree" % angle, (10, horiz_crop_middle_1 - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("bottom", img_debug)
        cv2.waitKey(500)
    return angle


def _save_images(ventral_view_img, right_view_img, left_view_img, fly_number, behaviour, behaviour_subfolder_name):
    # ventral_view = VV, Right View = RV, Left View = LV
    cv2.imwrite(ventral_view_dd + str(fly_number)+ '_' + behaviour + '_' + behaviour_subfolder_name + '_' + 'VV' + '_'  + "%05d"%name_counter + ".tiff", ventral_view_img)
    cv2.imwrite(right_view_dd + str(fly_number)+ '_' + behaviour + '_' + behaviour_subfolder_name + '_'  + 'RV' + '_'  + "%05d"%name_counter + ".tiff", right_view_img)
    cv2.imwrite(left_view_dd + str(fly_number)+ '_' + behaviour + '_' + behaviour_subfolder_name + '_'  + 'LV' + '_'  + "%05d"%name_counter + ".tiff", left_view_img)


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

def orientation(img, th=10, k=30):
    img_th = img.copy()

    # threshold
    _, img_thresh = cv2.threshold(img_th, th, 255, cv2.THRESH_BINARY)
    if DEBUG:
        imS = cv2.resize(img_thresh, (img_thresh.shape[0] // 2, img_thresh.shape[1] // 2))
        cv2.imshow("threshold", imS)
        cv2.waitKey(3000)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    flybody = get_largest_conncomp(img_thresh)  # mask of fly
    flybody = cv2.morphologyEx(flybody, cv2.MORPH_OPEN, kernel)  # chop legs off with kernel

    if DEBUG:
        imS = cv2.resize(flybody, (img_thresh.shape[0] // 2, img_thresh.shape[1] // 2))
        cv2.imshow("openning", imS)
        cv2.waitKey(3000)

    contour, _ = cv2.findContours(flybody, 1, 2)
    contour = max(contour, key=cv2.contourArea)

    ellipse = cv2.fitEllipse(contour)
    cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
    angle = ellipse[2] + 90 + 180

    h, w = img.shape
    M_tr = np.float32([[1, 0, w / 2 - cx], [0, 1, h / 2 - cy]])
    img = cv2.warpAffine(img, M_tr, (w, h))
    img = ndimage.rotate(img, angle, reshape=False)

    return angle, cx, cy, img



if __name__ == '__main__':

    # str at position -14 at each side should be fly number
    # -4:-2 should be two letters
    # -1 should be behaviour_subfolder_name
    # folders = [['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/AG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/AG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/AG/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/AG/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/FW/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/FW/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/FW/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/FW/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/PE/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/PE/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/PE/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/PG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/PG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_2_clipped/PG/3', True],
    #
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/5', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/6', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/7', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/8', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/PE/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/PE/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/PG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/PG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/PG/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/PG/4', True],
    #
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/AG/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/AG/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/AG/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/FW/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/FW/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/FW/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PE/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PE/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PE/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/5', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/6', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/7', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/8', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_4_clipped/PG/9', False],
    #
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/AG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/AG/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/AG/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/5', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/6', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/7', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/8', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/9', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/10', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/11', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/12', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/13', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/FW/14', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PE/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PE/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PG/2', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PG/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PG/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/PG/5', True],
    #
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/AG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/AG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/AG/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/AG/4', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/FW/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/FW/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/FW/3', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PE/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PE/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/4', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/5', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/6', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/7', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_5_clipped/PG/8', True],
    #
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/AG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/AG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/AG/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/AG/4', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/FW/1', False],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/FW/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/FW/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PE/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PE/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PE/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PE/4', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PG/1', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PG/2', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PG/3', True],
    #            ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/PG/4', True]
    #            ]

    folders = [['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/FW/1', False]]


    # video_id = 1
    #
    # # # # import imageio
    # # # # from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    # # # # ffmpeg_extract_subclip("/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_1_clipped/{}.mp4".format(video_id), 0.473, 2.220, targetname="/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/raw_fly_1/clipped_{}.mp4".format(video_id))
    # # #
    # # uncomment to convert video to images
    # import cv2
    # vidcap = cv2.VideoCapture("/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/{}.mp4".format(video_id))
    # success, image = vidcap.read()
    # count = 0
    # while success:
    #     cv2.imwrite("/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_6_clipped/tmp/frame%d.tiff" % count, image)  # save frame as JPEG file
    #     success, image = vidcap.read()
    #     print('Read a new frame: ', success)
    #     count += 1


    # folders = [['/home/mahdi/Pictures/fly_1_clipped/AG/1', True]]


    DEBUG = False
    IMG_PREV = None

    for data_dir, flip_switch in folders:

        fly_number = data_dir[-14]
        behaviour = data_dir[-4:-2]  # Forward Walking = FW, Proboscis Expansion = PE, Anterior Grooming = AG, Posterior Grooming = PG
        behaviour_subfolder_name = data_dir[-1]

        if fly_number=='1':
            threshold = 30
            DIST_TH = 10
            border_width_1 = 120
            border_width_2 = 228
            bbox_width = 620
            horiz_crop_right_1 = 8
            horiz_crop_right_2 = 266
            horiz_crop_middle_1 = 348
            horiz_crop_middle_2 = 798
            horiz_crop_left_1 = 888
            horiz_crop_left_2 = 1162
        elif fly_number=='2':
            threshold = 30 # for mean value on column
            DIST_TH = 10
            border_width_1 = 200
            border_width_2 = 95
            bbox_width = 550
            horiz_crop_right_1 = 20
            horiz_crop_right_2 = 286
            horiz_crop_middle_1 = 386
            horiz_crop_middle_2 = 858
            horiz_crop_left_1 = 926
            horiz_crop_left_2 = 1186
        elif fly_number=='3':
            threshold = 30
            DIST_TH = 10
            border_width_1 = 200
            border_width_2 = 214
            bbox_width = 550
            horiz_crop_right_1 = 32
            horiz_crop_right_2 = 290
            horiz_crop_middle_1 = 392
            horiz_crop_middle_2 = 830
            horiz_crop_left_1 = 950
            horiz_crop_left_2 = 1182
        elif fly_number=='4':
            threshold = 30
            DIST_TH = 10
            border_width_1 = 172
            border_width_2 = 288
            bbox_width = 550
            horiz_crop_right_1 = 20
            horiz_crop_right_2 = 280
            horiz_crop_middle_1 = 398
            horiz_crop_middle_2 = 824
            horiz_crop_left_1 = 908
            horiz_crop_left_2 = 1166
        elif fly_number=='5':
            threshold = 30
            DIST_TH = 10
            border_width_1 = 122
            border_width_2 = 202
            bbox_width = 620
            horiz_crop_right_1 = 18
            horiz_crop_right_2 = 286
            horiz_crop_middle_1 = 370
            horiz_crop_middle_2 = 826
            horiz_crop_left_1 = 916
            horiz_crop_left_2 = 1182
        elif fly_number=='6':
            threshold = 25
            DIST_TH = 10
            border_width_1 = 154
            border_width_2 = 224
            bbox_width = 620
            horiz_crop_right_1 = 26
            horiz_crop_right_2 = 284
            horiz_crop_middle_1 = 376
            horiz_crop_middle_2 = 824
            horiz_crop_left_1 = 908
            horiz_crop_left_2 = 1184
        else:
            IOError('fly number properties not defined!')

        if not data_dir.endswith("/"): data_dir += "/"

        ventral_view_dd = data_dir + "VV" + "/"
        right_view_dd = data_dir + "RV" + "/"
        left_view_dd = data_dir + "LV" + "/"


        fcrop_loc_name = "crop_info" + ".txt"
        if ~DEBUG:
            if isdir(ventral_view_dd):
                shutil.rmtree(ventral_view_dd)
            if isdir(right_view_dd):
                shutil.rmtree(right_view_dd)
            if isdir(left_view_dd):
                shutil.rmtree(left_view_dd)
            mkdir(ventral_view_dd)
            mkdir(right_view_dd)
            mkdir(left_view_dd)
        else:
            if ~isdir(ventral_view_dd):
                mkdir(ventral_view_dd)
            if ~isdir(right_view_dd):
                mkdir(right_view_dd)
            if ~isdir(left_view_dd):
                mkdir(left_view_dd)

        print(f"\n[*] reading images name from {data_dir:s}")
        img_names = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f.endswith(".tiff")]
        img_names.sort()
        imgs_len = len(img_names)
        img_prev = None
        n_skip_dist = 0
        n_skip_out = 0
        print(f"[*] splitting the images into right_view and ventral_view views\n")

        fcrop_loc = open(data_dir + fcrop_loc_name, 'w')

        orientation_info_file = ventral_view_dd + 'orientation_info.txt'
        VV_orientation = open(orientation_info_file, 'w')

        orientation_info_file = left_view_dd + 'orientation_info.txt'
        LV_orientation = open(orientation_info_file, 'w')

        orientation_info_file = right_view_dd + 'orientation_info.txt'
        RV_orientation = open(orientation_info_file, 'w')

        print(fcrop_loc_name)
        fcrop_loc.write("border_width_1 = " + str(border_width_1) + "\n" + \
                        "border_width_2 = " + str(border_width_2) + "\n" + \
                        "threshold = " + str(threshold) + "\n" + \
                        "DIST_TH = " + str(DIST_TH) + "\n" + \
                        "bbox_width = " + str(bbox_width) + "\n" + \
                        "horiz_crop_middle_1 = " + str(horiz_crop_middle_1) + "\n" + \
                        "horiz_crop_middle_2 = " + str(horiz_crop_middle_1) + "\n" + \
                        "horiz_crop_right_1 = " + str(horiz_crop_right_1) + "\n" + \
                        "horiz_crop_right_2 = " + str(horiz_crop_right_2) + "\n")
        name_counter = 0
        for counter in tqdm(range(imgs_len)):
            img_name = img_names[counter]

            img = cv2.imread(data_dir + img_name)

            # Convert to gray scale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Remove border with very bright light
            img = _remove_borders(img, border_width_1, border_width_2)

            # flip to head to left for consistency
            if flip_switch==True:
                img = cv2.flip(img, 1)

            # enhance contrast
            img = equalize_adapthist(img, kernel_size=tuple([img.shape[0]//8, img.shape[1]//8]), clip_limit=0.0062, nbins=256)
            img = img_as_ubyte(img)

            img_dist = np.copy(img)
            # img_dist[img_dist < 60] = 0
            # img_bin = (img > 60).astype(float)

            if np.any(IMG_PREV == None):
                IMG_PREV = img_dist
                dist = 999
            elif img_dist.shape == IMG_PREV.shape:
                dist = np.mean((img_dist - IMG_PREV) ** 2)
                if dist < DIST_TH:
                    n_skip_dist += 1
                    continue
                IMG_PREV = img_dist

            if DEBUG:
                cv2.putText(img_dist, "distance %.5f" % dist, (10, img_dist.shape[0] - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

                imS = cv2.resize(img_dist, (1920 // 2, 1200 // 2))
                cv2.imshow("removed borders and enhanced contrast", imS)

                cv2.waitKey(100)

            # bottom_img, ventral_view_img, left_vert_crop, right_vert_crop = _separate_bottom_side_flies(img, threshold)
            ventral_view_img, right_view_img, left_view_img, left_vert_crop, right_vert_crop = _separate_flies_vertically(img, threshold)

            # img = pad(img, pad_width=((0, 0), (-(left_vert_crop - left_bbox_pad), 0)), mode='constant')

            # # pad the height to constant value
            # bbox_height = 500
            # try:
            #     right_view_img = pad(right_view_img, pad_width=(((bbox_height-right_view_img.shape[0])//2, (bbox_height-right_view_img.shape[0])//2), (0, 0)), mode='constant')
            #     left_view_img = pad(left_view_img, pad_width=(((bbox_height-left_view_img.shape[0])//2, (bbox_height-left_view_img.shape[0])//2), (0, 0)), mode='constant')
            #     ventral_view_img = pad(ventral_view_img, pad_width=(((bbox_height-ventral_view_img.shape[0])//2, (bbox_height-ventral_view_img.shape[0])//2), (0, 0)), mode='constant')
            # except:
            #     continue

            # pad the width and height to compensate the orientation(rotation)
            height_pad = 25
            width_pad = 25
            try:
                # right_view_img = pad(right_view_img, pad_width=((0, 0), (width_pad, width_pad)), mode='constant')
                # left_view_img = pad(left_view_img, pad_width=((0, 0), (width_pad, width_pad)), mode='constant')
                ventral_view_img = pad(ventral_view_img, pad_width=((height_pad, height_pad), (width_pad, width_pad)), mode='constant')
            except:
                continue


            if ventral_view_img.shape[1] != bbox_width + 2 * width_pad or right_view_img.shape[1] != bbox_width or left_view_img.shape[1] != bbox_width:
                if DEBUG:
                    imS = cv2.resize(img, (1920 // 2, 1200 // 2))
                    cv2.imshow("outlier", imS)
                    cv2.waitKey(500)
                n_skip_out += 1
                continue

            if DEBUG:
                imS = cv2.resize(ventral_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("ventral_view", imS)
                cv2.waitKey(500)
                imS = cv2.resize(right_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("right_view", imS)
                imS = cv2.resize(left_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("left_view", imS)
                cv2.waitKey(500)



            # try:
            #     angle, cx, cy, oriented_right_view_img = orientation(right_view_img, th=orientation_threshold, k=orientation_k)
            # except:
            #     angle = 'None'
            #     cx = 'None'
            #     cy = 'None'
            #     oriented_right_view_img = right_view_img
            # RV_orientation.write(right_view_dd + str(
            #     fly_number) + '_' + behaviour + '_' + behaviour_subfolder_name + '_' + 'RV' + '_' + "%05d" % name_counter + ".tiff" + ' angle = ' + '{:06.3f}'.format(
            #     angle) + ' cx = ' + str(cx) + ' cy = ' + str(cy) + "\n")



            # try:
            #     angle, cx, cy, oriented_left_view_img = orientation(left_view_img, th=orientation_threshold, k=orientation_k)
            # except:
            #     angle = 'None'
            #     cx = 'None'
            #     cy = 'None'
            #     oriented_left_view_img = left_view_img


            # LV_orientation.write(left_view_dd + str(
            #     fly_number) + '_' + behaviour + '_' + behaviour_subfolder_name + '_' + 'LV' + '_' + "%05d" % name_counter + ".tiff" + ' angle = ' + '{:06.3f}'.format(angle) + ' cx = ' + str(cx) + ' cy = ' + str(cy) + "\n")

            if not DEBUG:
                name_counter += 1
                try:
                    orientation_threshold = 30
                    orientation_k = 40
                    angle, cx, cy, oriented_ventral_view_img = orientation(ventral_view_img, th=orientation_threshold,
                                                                           k=orientation_k)
                    VV_orientation.write(ventral_view_dd + str(
                        fly_number) + '_' + behaviour + '_' + behaviour_subfolder_name + '_' + 'VV' + '_' + "%05d" % name_counter + ".tiff" + ' angle = ' + '{:06.3f}'.format(
                        angle) + ' cx = ' + str(cx) + ' cy = ' + str(cy) + ' height_pad = ' + str(
                        height_pad) + ' width_pad = ' + str(width_pad) + "\n")
                except:
                    angle = 0
                    cx = 0
                    cy = 0
                    oriented_ventral_view_img = ventral_view_img
                    VV_orientation.write(ventral_view_dd + str(
                        fly_number) + '_' + behaviour + '_' + behaviour_subfolder_name + '_' + 'VV' + '_' + "%05d" % name_counter + ".tiff" + ' angle = ' + '{:06.3f}'.format(
                        angle) + ' cx = ' + str(cx) + ' cy = ' + str(cy) + "\n")

                angle = 0
                cx = 0
                cy = 0
                oriented_right_view_img = right_view_img

                # TODO
                # flip vertically the right vie to be the same as left view!!
                oriented_right_view_img = cv2.flip(oriented_right_view_img, 0)

                angle = 0
                cx = 0
                cy = 0
                oriented_left_view_img = left_view_img
                fcrop_loc.write(
                    img_name + " " + "new name = %05d.tiff"%name_counter + " " + str(left_vert_crop) + " " + str(right_vert_crop) + " " + str(orientation) + "\n")
                _save_images(oriented_ventral_view_img, oriented_right_view_img, oriented_left_view_img, fly_number= fly_number, behaviour=behaviour, behaviour_subfolder_name=behaviour_subfolder_name)

            if DEBUG:
                imS = cv2.resize(oriented_ventral_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("oriented_ventral_view", imS)
                cv2.waitKey(3000)
                imS = cv2.resize(oriented_right_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("oriented_right_view", imS)
                cv2.waitKey(3000)
                imS = cv2.resize(oriented_left_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("oriented_left_view", imS)
                cv2.waitKey(3000)
        # create video from images
        os.chdir(ventral_view_dd)
        subprocess.call(
            'ffmpeg -r 150 -f image2 -i {}_{}_{}_VV_%05d.tiff  -vcodec libx264 -crf 0  -pix_fmt yuv420p {}_{}_{}_VV_video.mp4'.format(
                fly_number, behaviour, behaviour_subfolder_name, fly_number, behaviour, behaviour_subfolder_name), shell=True)
        os.chdir(right_view_dd)
        subprocess.call(
            'ffmpeg -r 150 -f image2 -i {}_{}_{}_RV_%05d.tiff  -vcodec libx264 -crf 0  -pix_fmt yuv420p {}_{}_{}_RV_video.mp4'.format(
                fly_number, behaviour, behaviour_subfolder_name, fly_number, behaviour, behaviour_subfolder_name), shell=True)
        os.chdir(left_view_dd)
        subprocess.call(
            'ffmpeg -r 150 -f image2 -i {}_{}_{}_LV_%05d.tiff  -vcodec libx264 -crf 0  -pix_fmt yuv420p {}_{}_{}_LV_video.mp4'.format(
                fly_number, behaviour, behaviour_subfolder_name, fly_number, behaviour, behaviour_subfolder_name), shell=True)

        fcrop_loc.close()
        VV_orientation.close()
        LV_orientation.close()
        RV_orientation.close()
        print(
            f"\n[*] skipped {n_skip_dist:n}, {n_skip_out:n} frames because the fly was doing nothing or because it was an outlier")
        print("\n[+] done\n")

        with open(data_dir + "info.txt", 'w') as info:
            info.write("n_skip_dist: %d" % n_skip_dist)
            info.write("\nn_skip_out: %d" % n_skip_out)