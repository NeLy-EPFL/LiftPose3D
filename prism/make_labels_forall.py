
def get_all_joints(top_joints, side_joints, left_vert_crop, horiz_crop):
    all_joints = []

    for idx in range(0, len(top_joints), 3):
        x = top_joints[idx]
        y = top_joints[idx+1]
        likelihood = top_joints[idx+2]
        # TODO: modify x and y

        all_joints += [x, y, likelihood]
    for idx in range(0, len(side_joints), 3):
        x = side_joints[idx]
        y = side_joints[idx+1]
        likelihood = side_joints[idx+2]
        # TODO: modify x and y

        all_joints += [x, y, likelihood]

    return all_joints

if __name__ == '__main__':
    args = sys.argv
    imgs_dir = args[1]
    if not imgs_dir.endswith("/") : imgs_dir += "/"

    imgs_dir_spl = imgs_dir.split("/")
    post_top = "top_view_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]
    top_dd = imgs_dir + post_top +"/"
    post_side = "side_view_"+ imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6]
    side_dd = imgs_dir + post_side +"/"
    crop_loc_file =\
        imgs_dir + "crop_location" + imgs_dir_spl[4] +"_"+ imgs_dir_spl[5] +"_"+ imgs_dir_spl[6] + ".txt"

    img_names_side = [f for f in listdir(side_dd)\
        if isfile(join(side_dd, f)) and f.endswith(".jpg")]
    img_names_side.sort()

    img_names_top = [f for f in listdir(top_dd)\
        if isfile(join(top_dd, f)) and f.endswith(".jpg")]
    img_names_top.sort()
    
    top_joint_file = top_dd +\
        post_top + "DeepCut_resnet50_jointTrackingDec13shuffle1_200000.csv"
    side_joint_file = side_dd +\
        post_side + "DeepCut_resnet50_jointTrackingDec13shuffle1_200000.csv"

    top_joint_dic = {}
    side_joint_dic = {}
    top_info = None
    side_info = None
    with open(top_joint_file, 'r') as top_fp,\
        open(side_joint_file, 'r') as side_fp:
        top_joint_reader = csv.reader(top_fp, delimiter=',')
        side_joint_reader = csv.reader(side_fp, delimiter=',')
        idx = 0
        for row in top_joint_reader:
            idx += 1
            if idx == 2 : top_info = row[idx]
            if idx < 4: continue
            im_name = row[0].split(".")[0]
            im_name = "_".join(im_name.split("_")[:-1])
            top_joint_dic[im_name] = np.array(row[1:], dtype=np.float).astype(int)
        idx = 0
        for row in side_joint_reader:
            idx += 1
            if idx == 2 : side_info = row[idx].split("bodyparts,")[1]
            if idx < 4: continue
            side_joint_dic[im_name] = np.array(row[1:], dtype=np.float).astype(int)

    with open(crop_loc_file, 'r') as croploc_fp,\
        open(imgs_dir + "joint_locations.csv", 'w') as jloc_fp:
        croploc_lines = croploc_fp.readlines()
        horiz_crop = int(croploc_lines[4].split(":")[1])
        bbox_width = int(croploc_lines[5].split(":")[1])
        print(horiz_crop, bbox_width)

        top_info = ','.join([s + " top" for s in top_info.split(',')])
        side_info = ','.join([s + " side" for s in side_info.split(',')])
        info = top_info + ',' + side_info
        print(info)

        skip_lines = 8
        for croploc in croploc_lines[skip_lines:]:
            im_name = croploc.split(":")[0]
            im_name_noext = im_name.split(".")[0]

            left_vert_crop = int(croploc.split(":")[1])
            print(left_vert_crop)

            # putting top and side joints together
            top_joints = top_joint_dic[im_name_noext]
            side_joints = side_joint_dic[im_name_noext]
            all_joints = get_all_joints(top_joints, side_joints, left_vert_crop, horiz_crop)

            # writing all joints to new file
            jloc_fp.write(im_name)
            all_joints_str = ','.join(str(j) for j in all_joints)
            jloc_fp.write(',')
            jloc_fp.write(all_joints_str)
            jloc_fp.write('\n')

