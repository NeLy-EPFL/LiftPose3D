import csv
import numpy as np
import cv2

filename = "/ramdya-nas/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/top_view_191125_PR_Fly1_001_prism/top_view_191125_PR_Fly1_001_prismDeepCut_resnet50_jointTrackingDec13shuffle1_200000.csv"
dir_ = "/ramdya-nas/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/top_view_191125_PR_Fly1_001_prism/"

def draw_joints(im, top_joints):
    top_joints = top_joints[:int(len(top_joints)/2)]
    for idx in range(0, len(top_joints), 3):
        x = top_joints[idx]
        y = top_joints[idx+1]
        #if np.isnan(x) or np.isnan(y) : continue
        if y > im.shape[0] or x > im.shape[1] or y < 0 or x < 0 : continue
        im[y-2:y+2, x-2:x+2, :] = 255
    return im

with open(filename, 'r') as fp:
    top_joint_reader = csv.reader(fp, delimiter=',')

    idx = 0
    for row in top_joint_reader:
        idx += 1
        if idx < 4: continue
        
        if idx == 3000:
            im_name = row[0]
            #im_name = row[0].split(".")[0]
            #im_name = "_".join(im_name.split("_")[:-1])
            im = cv2.imread(dir_ + im_name)
            top_joints = np.array(row[1:], dtype=np.float).astype(int)

            im = draw_joints(im, top_joints)
            cv2.imwrite("debug.jpg", im)
