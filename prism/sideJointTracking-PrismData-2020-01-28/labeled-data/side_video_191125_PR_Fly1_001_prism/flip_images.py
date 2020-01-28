import cv2

from os import listdir
from os.path import isfile, join

import numpy as np

import h5py
import pandas as pd
filename = "CollectedData_PrismData.h5"

bodyparts = [\
    "body-coxa front R",
    "coxa-femur front R",
    "femur-tibia front R",
    "tibia-tarsus front R",
    "tarsus tip front R",
    "body-coxa mid R",
    "coxa-femur mid R",
    "femur-tibia mid R",
    "tibia-tarsus mid R",
    "tarsus tip mid R",
    "body-coxa back R",
    "coxa-femur back R",
    "femur-tibia back R",
    "tibia-tarsus back R",
    "tarsus tip back R"\
]

def print_joints(im, ls):
    for x in range(0, len(ls), 2):
        cv2.circle(im, (int(ls[x]), int(ls[x+1])), 3, (255,0,0), -1)

def is_going_left(ls):
    left = np.sum(ls[:30])
    right = np.sum(ls[30:])
    if not np.isnan(left):
        return True
    return False

if __name__ == '__main__':
    f = pd.read_hdf(filename, 'df_with_missing')
    f.sort_index(inplace=True)

    indexes = f.index
    columns = f.columns[30:]

    new_labels = []
    
    for idx, im_all_name in enumerate(f.index):
        labels = f.loc[im_all_name, :].values
        print(labels)
        im_name = im_all_name.split("/")[2]
        im = cv2.imread(im_name)
        im_debug = im.copy()
        if is_going_left(labels):
            # flip image and labels
            labels = labels[:30]
            print_joints(im_debug, labels)
            #cv2.namedWindow("debug", cv2.WINDOW_AUTOSIZE)
            #cv2.imshow("debug", im_debug)
            #cv2.waitKey(500)

            for x in range(0, len(labels), 2):
                labels[x] = im.shape[1] - labels[x]
            im_flipped = np.flip(im, axis=1)

            im_flipped_debug = im_flipped.copy()
            print_joints(im_flipped_debug, labels)
            #cv2.imshow("debug", im_flipped_debug)
            #cv2.waitKey(500)

            new_labels.append(labels)
        else:
            new_labels.append(labels[30:])
    
    new_labels = np.vstack(new_labels)
    print(new_labels.shape)
    new_f = pd.DataFrame(data=new_labels, columns=columns, index=indexes)
    
    new_f.to_hdf(filename, 'df_with_missing', format='table', mode='w')
    print()
    print("[+] complete!")
    print()
