import cv2
import sys
import csv
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

def print_joints(img_t, joint_coord):
    for leg in range(3):
        # leg number leg
        for idx in range(5*leg*3, 5*(leg+1)*3-3, 3):
            cv2.line(img_t, (joint_coord[idx], joint_coord[idx+1]),
                     (joint_coord[idx+3], joint_coord[idx+3+1]),
                     color=(255,0,0), thickness=3)

if __name__ == '__main__':
    args = sys.argv
    imgs_dir = args[1]
    if not imgs_dir.endswith("/") : imgs_dir += "/"

    imgs_dir_spl = imgs_dir.split("/")
    post_side = "side_view_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]
    side_dd = imgs_dir + post_side +"/"

    img_names_side = [f for f in listdir(side_dd)\
        if isfile(join(side_dd, f)) and f.endswith(".jpg")]
    img_names_side.sort()

    joint_dic = {}
    joint_file = side_dd +\
        post_side + "DeepCut_resnet50_sideJointTrackingDec17shuffle1_200000.csv"
    with open(joint_file, 'r') as joint_fp:
        joint_reader = csv.reader(joint_fp, delimiter=',')
        idx = 0
        for row in joint_reader: # each row is an image
            idx += 1
            if idx < 4 : continue
            joint_dic[row[0]] = np.array(row[1:], dtype=np.float).astype(int)

    print(f"[*] showing {len(img_names_side):n} frames")
    cv2.namedWindow("side", flags=cv2.WINDOW_AUTOSIZE)
    for idx in tqdm(range(len(img_names_side))):
        img_ns = img_names_side[idx]
        
        img_s = cv2.imread(side_dd + img_ns)
        print_joints(img_s, joint_dic[img_ns])
        cv2.imshow("side", img_s)
        cv2.waitKey(10)
    cv2.destroyWindow("side")