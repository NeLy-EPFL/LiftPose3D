import cv2

from os import listdir
from os.path import isfile, join

import numpy as np

import h5py
filename = "CollectedData_PrismData.h5"

def print_joints(im, ls):
    for x in range(0, len(ls), 2):
        cv2.circle(im, (int(ls[x]), int(ls[x+1])), 3, (255,0,0), -1)

def is_going_left(ls):
    left = np.sum(ls[:30])
    right = np.sum(ls[30:])
    print(left, right)
    if not np.isnan(left):
        return True
    return False

if __name__ == '__main__':
    with h5py.File(filename, 'r+') as f:
        labels = list(f['df_with_missing']['table'])
        new_labels = []
        # [
        #   (b'image_name.png', list of labels)
        #   ...
        # ]
        #print(labels)
        for l in labels:
            img_name, ls = l[0].decode('utf-8').split("/")[2], l[1]
            im = cv2.imread(img_name)
            im_debug = im.copy()
            if is_going_left(ls):
                # flip image and labels
                ls = ls[:30]
                print_joints(im_debug, ls)
                cv2.namedWindow("debug", cv2.WINDOW_AUTOSIZE)
                cv2.imshow("debug", im_debug)
                cv2.waitKey(500)

                for x in range(0, len(ls), 2):
                    ls[x] = im.shape[1] - ls[x]
                im_flipped = np.flip(im, axis=1)

                im_flipped_debug = im_flipped.copy()
                print_joints(im_flipped_debug, ls)
                cv2.imshow("debug", im_flipped_debug)
                cv2.waitKey(500)

                new_labels.append( (l[0], ls) )
            else:
                new_labels.append( (l[0], ls[30:]) )
        f['df_with_missing']['table'] = new_labels

    print()
    print("[+] complete!")
    print()
