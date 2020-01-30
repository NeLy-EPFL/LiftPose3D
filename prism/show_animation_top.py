import cv2
import sys
import csv
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import numpy as np

def print_joints(img_t, joint_coord):
    for leg in range(6):
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
    post_top = "top_view_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]
    top_dd = imgs_dir + post_top +"/"

    img_names_top = [f for f in listdir(top_dd)\
        if isfile(join(top_dd, f)) and f.endswith(".jpg")]
    img_names_top.sort()

    joint_dic = {}
    joint_file = top_dd +\
        post_top + "DeepCut_resnet50_jointTrackingDec13shuffle1_200000.csv"
    with open(joint_file, 'r') as joint_fp:
        joint_reader = csv.reader(joint_fp, delimiter=',')
        idx = 0
        for row in joint_reader: # each row is an image
            idx += 1
            if idx < 4 : continue
            joint_dic[row[0]] = np.array(row[1:], dtype=np.float).astype(int)

    print(f"[*] showing {len(img_names_top):n} frames")
    cv2.namedWindow("top", flags=cv2.WINDOW_AUTOSIZE)
    for idx in tqdm(range(len(img_names_top))):
        img_nt = img_names_top[idx]
        
        img_t = cv2.imread(top_dd + img_nt)
        print_joints(img_t, joint_dic[img_nt])
        cv2.imshow("top", img_t)
        cv2.waitKey(10)
    cv2.destroyWindow("top")
