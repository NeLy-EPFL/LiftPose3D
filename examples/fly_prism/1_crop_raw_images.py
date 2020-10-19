from os import listdir
from os.path import isfile, isdir, join
from os import mkdir
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

def _remove_borders(img, bwidth):
    img = img[:, bwidth:-bwidth]

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


def _save_images(ventral_view_img, right_view_img, left_view_img, orientation, dd, name_counter, fly_number, behaviour, behaviour_subfolder_name):
    # ventral_view = VV, Right View = RV, Left View = LV
    cv2.imwrite(ventral_view_dd + str(fly_number)+ '_' + behaviour + '_' + behaviour_subfolder_name + '_' + 'VV' + '_'  + "%05d"%name_counter + ".tiff", ventral_view_img)
    cv2.imwrite(right_view_dd + str(fly_number)+ '_' + behaviour + '_' + behaviour_subfolder_name + '_'  + 'RV' + '_'  + "%05d"%name_counter + ".tiff", right_view_img)
    cv2.imwrite(left_view_dd + str(fly_number)+ '_' + behaviour + '_' + behaviour_subfolder_name + '_'  + 'LV' + '_'  + "%05d"%name_counter + ".tiff", left_view_img)



if __name__ == '__main__':

    # str at position -14 at each side should be fly number
    # -4:-2 should be two letters
    # -1 should be behaviour_subfolder_name
    folders = ['/media/mahdi/LaCie/Mahdi/SSD/data_2Dpose/fly_3_clipped/FW/1'
               ]

    DEBUG = False
    IMG_PREV = None

    for data_dir in folders:

        fly_number = data_dir[-14]
        behaviour = data_dir[-4:-2]  # Forward Walking = FW, Proboscis Expansion = PE, Anterior Grooming = AG, Posterior Grooming = PG
        behaviour_subfolder_name = data_dir[-1]

        if fly_number=='2':
            threshold = 20
            DIST_TH = 10
            border_width = 250
            bbox_width = 550
            horiz_crop_right_1 = 32
            horiz_crop_right_2 = 290
            horiz_crop_middle_1 = 392
            horiz_crop_middle_2 = 830
            horiz_crop_left_1 = 950
            horiz_crop_left_2 = 1182
        elif fly_number=='3':
            threshold = 20
            DIST_TH = 10
            border_width = 250
            bbox_width = 550
            horiz_crop_right_1 = 32
            horiz_crop_right_2 = 290
            horiz_crop_middle_1 = 392
            horiz_crop_middle_2 = 830
            horiz_crop_left_1 = 950
            horiz_crop_left_2 = 1182
        else:
            IOError('fly number properties not defined!')

        if not data_dir.endswith("/"): data_dir += "/"

        ventral_view_dd = data_dir + "VV" + "/"
        right_view_dd = data_dir + "RV" + "/"
        left_view_dd = data_dir + "LV" + "/"

        fcrop_loc_name = "crop_location" + ".txt"

        if not isdir(ventral_view_dd):
            mkdir(ventral_view_dd)
        if not isdir(right_view_dd):
            mkdir(right_view_dd)
        if not isdir(left_view_dd):
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
        print(fcrop_loc_name)
        fcrop_loc.write("border_width = " + str(border_width) + "\n" + \
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
            img = _remove_borders(img, border_width)

            # enhance contrast
            img = equalize_adapthist(img, kernel_size=tuple([img.shape[0]//8, img.shape[1]//8]), clip_limit=0.006, nbins=256)
            img = img_as_ubyte(img)

            img_dist = np.copy(img)
            # img_dist[img_dist < 60] = 0
            # img_bin = (img > 60).astype(float)

            if np.any(IMG_PREV == None):
                IMG_PREV = img_dist
                dist = 999
            else:
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

            if ventral_view_img.shape[1] != bbox_width or right_view_img.shape[1] != bbox_width or left_view_img.shape[1] != bbox_width:
                if DEBUG:
                    imS = cv2.resize(img, (1920 // 2, 1200 // 2))
                    cv2.imshow("outlier", imS)
                    cv2.waitKey(500)
                n_skip_out += 1
                continue

            # orientation = _find_orientation(bottom_img)
            # if orientation == None : continue
            orientation = None
            if DEBUG:
                imS = cv2.resize(ventral_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("ventral_view", imS)
                cv2.waitKey(500)
                imS = cv2.resize(right_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("right_view", imS)
                imS = cv2.resize(left_view_img, (1920 // 2, 1200 // 2))
                cv2.imshow("left_view", imS)
                cv2.waitKey(500)

            if not DEBUG:
                name_counter += 1
                fcrop_loc.write(
                    img_name + " " + "new name = %05d"%name_counter + " " + str(left_vert_crop) + " " + str(right_vert_crop) + " " + str(orientation) + "\n")
                _save_images(ventral_view_img, right_view_img, left_view_img, orientation, data_dir, name_counter = name_counter, fly_number= fly_number, behaviour=behaviour, behaviour_subfolder_name=behaviour_subfolder_name)

        fcrop_loc.close()
        print(
            f"\n[*] skipped {n_skip_dist:n}, {n_skip_out:n} frames because the fly was doing nothing or because it was an outlier")
        print("\n[+] done\n")

        with open(data_dir + "info.txt", 'w') as info:
            info.write("n_skip_dist: %d" % n_skip_dist)
            info.write("\nn_skip_out: %d" % n_skip_out)