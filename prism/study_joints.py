import csv
import numpy as np
import cv2

filename_top = "/ramdya-nas/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/top_view_191125_PR_Fly2_001_prism/top_view_191125_PR_Fly2_001_prismDeepCut_resnet50_jointTrackingDec13shuffle1_200000.csv"
filename_side = "/ramdya-nas/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/side_view_191125_PR_Fly2_001_prism/side_view_191125_PR_Fly2_001_prismDeepCut_resnet50_sideJointTrackingDec17shuffle1_200000.csv"
crop_loc_filename = "/ramdya-nas/SG/prism_data/191125_PR/Fly2/001_prism/behData/images/crop_location_191125_PR_Fly2_001_prism.txt"
dir_ = "/ramdya-nas/SG/prism_data/191125_PR/Fly2/001_prism/behData/images/top_view_191125_PR_Fly2_001_prism/"

def get_all_joints(top_joints, side_joints, left_vert_crop, horiz_crop):
    all_joints = []
    # TOP
    for idx in range(0, len(top_joints), 3):
        x = top_joints[idx]
        y = top_joints[idx+1]
        likelihood = top_joints[idx+2]
        
        x += left_vert_crop

        all_joints += [x, y, likelihood]
    
    # SIDE
    for idx in range(0, len(side_joints), 3):
        x = side_joints[idx]
        y = side_joints[idx+1]
        likelihood = side_joints[idx+2]

        x += left_vert_crop
        y += horiz_crop

        all_joints += [x, y, likelihood]

    return all_joints

def print_joints(im, all_joints):
    for idx in range(0, len(all_joints), 3):
        cv2.circle(img_t, (all_joints[idx], all_joints[idx+1]),
                   radius=3, color=(255,0,0), thickness=-1)

crop_loc = {}
with open(crop_loc_filename, 'r') as cl_fp:
    lines = cl_fp.readlines()
    horiz_crop = int(croploc_lines[4].split(":")[1])
    print(horiz_crop)
    for l in lines[8:]:
        im_name = croploc.split(":")[0]
        im_name_noext = im_name.split(".")[0]
        crop_loc[im_name_noext] = int(croploc.split(":")[1])


with open(filename_top, 'r') as fp_top, open(filename_side, 'r') as fp_side:
    top_joint_reader = csv.reader(fp_top, delimiter=',')
    side_joint_reader = csv.reader(fp_side, delimiter=',')
    
    im_name = ""
    idx = 0
    for row in top_joint_reader:
        idx += 1
        if idx < 4: continue
        if idx == 300:
            im_name = row[0].split(".")[0]
            im_name = "_".join(im_name.split("_")[:-1])
            top_joints = np.array(row[1:], dtype=np.float).astype(int)
    idx = 0
    for row in side_joint_reader:
        idx += 1
        if idx < 4: continue
        if idx == 300:
            im_name_side = row[0].split(".")[0]
            im_name_side = "_".join(im_name_side.split("_")[:-1])
            assert(im_name == im_name_side)
            side_joints = np.array(row[1:], dtype=np.float).astype(int)

    left_vert_crop = crop_loc[im_name]
    print(left_vert_crop)
    all_joints = get_all_joints(top_joints, side_joints, left_vert_crop, horiz_crop)

    # plotting
    cv2.namedWindow("debug", flags=cv2.WINDOW_AUTOSIZE)
    im = cv2.imread(dir_ + im_name)
    print_joints(im, all_joints)
    cv2.imshow("debug", im)
    cv2.waitKey(10000)
