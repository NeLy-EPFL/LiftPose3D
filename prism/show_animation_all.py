import cv2
import sys
import csv
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

def print_joints(img_t, joint_coord):
    # 6 legs top, 3 legs side
    for leg in range(9):
        # leg number leg
        for idx in range(5*leg*3, 5*(leg+1)*3-3, 3):
            cv2.line(img_t, (joint_coord[idx], joint_coord[idx+1]),
                     (joint_coord[idx+3], joint_coord[idx+3+1]),
                     color=(255,0,0), thickness=3)

if __name__ == '__main__':
    args = sys.argv
    imgs_dir = args[1]
    if not imgs_dir.endswith("/") : imgs_dir += "/"

    img_names = []
    joint_dic = {}
    joint_file = imgs_dir + "joint_locations.csv"
    print("Reading all joints from")
    print(joint_file)

    with open(joint_file, 'r') as joint_fp:
        joint_reader = csv.reader(joint_fp, delimiter=',')
        idx = 0
        for row in joint_reader: # each row is an image
            idx += 1
            if idx < 2 : continue
            joint_dic[row[0]] = np.array(row[1:], dtype=np.float).astype(int)
            img_names.append(row[0])
    img_names.sort()
    print("Done\n")
    print(f"[*] showing {len(img_names):n} frames")
    cv2.namedWindow("all", flags=cv2.WINDOW_AUTOSIZE)
    for idx in tqdm(range(len(img_names))):
        img_n = img_names[idx]
        
        img = cv2.imread(imgs_dir + img_n)
        print_joints(img, joint_dic[img_n])
        cv2.imshow("all", img)
        cv2.waitKey(10)
    cv2.destroyWindow("all")
