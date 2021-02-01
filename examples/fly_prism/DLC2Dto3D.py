#

import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import pickle
import src.procrustes as procrustes
import src.utils as utils

#
'''
calculate points3d.npy given DLC 2d pose estimations
points3d.shape=[#frames, #keypoints according to list bellow, 3 for (x,y,z)]

0- LF_body_coxa
1- LF_coxa_femur
2- LF_femur_tibia
3- LF_tibia_tarsus
4- LF_tarsal_claw
5- LM_body_coxa
6- LM_coxa_femur
7- LM_femur_tibia
8- LM_tibia_tarsus
9- LM_tarsal_claw
10- LH_body_coxa
11- LH_coxa_femur
12- LH_femur_tibia
13- LH_tibia_tarsus
14- LH_tarsal_claw
15- RF_body_coxa
16- RF_coxa_femur
17- RF_femur_tibia
18- RF_tibia_tarsus
19- RF_tarsal_claw
20- RM_body_coxa
21- RM_coxa_femur
22- RM_femur_tibia
23- RM_tibia_tarsus
24- RM_tarsal_claw
25- RH_body_coxa
26- RH_coxa_femur
27- RH_femur_tibia
28- RH_tibia_tarsus
29- RH_tarsal_claw
30- L_antenna
31- R_antenna
32- L_eye
33- R_eye
34- L_haltere
35- R_haltere
36- L_wing
37- R_wing
38- proboscis
39- neck
40- genitalia
41- scutellum
42- A1A2
43- A3
44- A4
45- A5
46- A6
'''

def main(fly_number,behaviour,video_sequence_number,AniPose_filter_enable=False, VV_net_name=None, LV_net_name=None):
    home_dir = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_{}_clipped'.format(fly_number)

    data_dir = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_{}_clipped/{}/{}'.format(fly_number,behaviour,video_sequence_number)
    scorer_bottom = '{}_{}_{}_VV_videoDLC_{}'.format(fly_number,behaviour,video_sequence_number, VV_net_name)
    scorer_side_LV = '{}_{}_{}_LV_videoDLC_{}'.format(fly_number,behaviour,video_sequence_number, LV_net_name)
    scorer_side_RV = '{}_{}_{}_RV_videoDLC_{}'.format(fly_number,behaviour,video_sequence_number, LV_net_name)
    # lateral cropped video of moving fly
    videos_side_LV = ['/{}/{}/LV/'.format(behaviour,video_sequence_number)]
    videos_side_RV = ['/{}/{}/RV/'.format(behaviour,video_sequence_number)]
    # ventral cropped video of moving fly
    videos_bottom = ['/{}/{}/VV/'.format(behaviour,video_sequence_number)]
    # joints
    leg_tips = ['LF_tarsal_claw', 'LM_tarsal_claw', 'LH_tarsal_claw',
                'RF_tarsal_claw', 'RM_tarsal_claw', 'RH_tarsal_claw']

    VV_bodyparts = ['LF_body_coxa', 'LF_coxa_femur', 'LF_femur_tibia', 'LF_tibia_tarsus', 'LF_tarsal_claw', 'LM_body_coxa',
                    'LM_coxa_femur', 'LM_femur_tibia', 'LM_tibia_tarsus', 'LM_tarsal_claw', 'LH_body_coxa', 'LH_coxa_femur',
                    'LH_femur_tibia', 'LH_tibia_tarsus', 'LH_tarsal_claw', 'RF_body_coxa', 'RF_coxa_femur',
                    'RF_femur_tibia', 'RF_tibia_tarsus', 'RF_tarsal_claw', 'RM_body_coxa', 'RM_coxa_femur',
                    'RM_femur_tibia', 'RM_tibia_tarsus', 'RM_tarsal_claw', 'RH_body_coxa', 'RH_coxa_femur',
                    'RH_femur_tibia', 'RH_tibia_tarsus', 'RH_tarsal_claw', 'L_antenna', 'R_antenna', 'L_eye', 'R_eye',
                    'L_haltier', 'R_haltier', 'L_wing', 'R_wing', 'proboscis', 'neck', 'genitalia']

    LV_bodyparts = ['LF_body_coxa', 'LF_coxa_femur', 'LF_femur_tibia', 'LF_tibia_tarsus', 'LF_tarsal_claw', 'LM_body_coxa',
                    'LM_coxa_femur', 'LM_femur_tibia', 'LM_tibia_tarsus', 'LM_tarsal_claw', 'LH_body_coxa', 'LH_coxa_femur',
                    'LH_femur_tibia', 'LH_tibia_tarsus', 'LH_tarsal_claw', 'L_antenna', 'L_eye', 'L_haltier', 'L_wing',
                    'proboscis', 'neck', 'genitalia', 'scutellum', 'A1A2', 'A3', 'A4', 'A5', 'A6']

    # find the arg of VV_bodyparts same as LV_bodyparts
    arg_VV_common_bodypart = []
    arg_LV_common_bodypart = []
    for idx, item in enumerate(LV_bodyparts):
        try:
            arg_VV_common_bodypart.append(np.where(np.asarray(VV_bodyparts) == item)[0][0])
            arg_LV_common_bodypart.append(idx)
        except:
            pass

    assert len(videos_side_LV) == len(videos_bottom), 'Number of video files must be the same from side_LV and bottom!'
    assert len(videos_side_RV) == len(videos_bottom), 'Number of video files must be the same from side_RV and bottom!'

    # %% md

    # Select mode

    # %%

    mode = 'DLC_video'  # 0: train, 1: prediction, 2: DLC_video, 3: train_low_res

    if mode == 'train':
        th1 = 0.99  # confidence threshold
        th2 = 10  # max L-R discrepancy in x coordinate
        align = 1
        nice_frames = 1
        register_floor = 1
    if mode == 'prediction':
        th1 = 0.99  # confidence threshold
        th2 = 10  # max L-R discrepancy in x coordinate
        align = 1
        nice_frames = 0
        register_floor = 1
    if mode == 'DLC_video':
        th1 = 0.1  # confidence threshold
        th2 = 15  # max L-R discrepancy in x coordinate
        align = 0
        nice_frames = 0
        register_floor = 0


    # %% md

    # Process
    # data

    # %%

    # frames mislabelled by DLC
    # bad_frames = [[],
    #               [663, 668, 676, 1012, 1013, 1014, 1015, 1016, 1017, 1019, 1024, 1294, 2099, 2114, 2149, 2152, 2860, 3506],
    #               [],
    #               [5, 306, 871, 945],
    #               [595],
    #               [],
    #               [],
    #               []]


    # %%

    def select_best_data(bottom, side_LV, side_RV, th1, th2, leg_tips):
        # select those frames with high confidence ventral view if all the tarsal_claw liklihood is larger than th1
        bottom_lk = bottom.loc[:, (leg_tips, 'likelihood')]
        mask = (bottom_lk > th1).sum(1) == 6
        bottom = bottom[mask].dropna()
        side_LV = side_LV[mask].dropna()
        side_RV = side_RV[mask].dropna()

        # find high confidence and low discrepancy keypoints in each frame
        likelihood_LV = side_LV.loc[:, (np.asarray(LV_bodyparts)[arg_LV_common_bodypart], 'likelihood')]
        likelihood_RV = side_RV.loc[:, (np.asarray(LV_bodyparts)[arg_LV_common_bodypart], 'likelihood')]
        pad_width = 25  # to compensate VV horizontal pad
        discrepancy_LV = np.abs(
            bottom.loc[:, (slice(None), 'x')].values[:, arg_VV_common_bodypart] - pad_width - side_LV.loc[:,
                                                                                              (slice(None), 'x')].values[:,
                                                                                              arg_LV_common_bodypart])
        discrepancy_RV = np.abs(
            bottom.loc[:, (slice(None), 'x')].values[:, arg_VV_common_bodypart] - pad_width - side_RV.loc[:,
                                                                                              (slice(None), 'x')].values[:,
                                                                                              arg_LV_common_bodypart])
        # find good keypoints corresponding to np.asarray(LV_bodyparts)[arg_LV_common_bodypart] for both LV and RV
        good_keypts = (likelihood_LV > th1) & (likelihood_RV > th1) & (discrepancy_LV < th2) & (discrepancy_RV < th2)
        good_keypts = good_keypts.droplevel(1, axis=1)

        assert side_LV.shape[0] == bottom.shape[0], 'Number of rows(=number of data) must match in filtered data!'

        return bottom, side_LV, side_RV, good_keypts, mask


    def flip_LR(data):
        cols = list(data.columns)
        half = int(len(cols) / 2)
        tmp = data.loc[:, cols[:half]].values
        data.loc[:, cols[:half]] = data.loc[:, cols[half:]].values
        data.loc[:, cols[half:]] = tmp

        return data


    for i in range(len(videos_side_LV)):
        # print(home_dir + videos_side_LV[i])

        if AniPose_filter_enable==False:
            # load data of side_LV and bottom view
            _side_LV = pd.read_hdf(home_dir + videos_side_LV[i] + scorer_side_LV + '.h5')
            _side_RV = pd.read_hdf(home_dir + videos_side_RV[i] + scorer_side_RV + '.h5')
            _bottom = pd.read_hdf(home_dir + videos_bottom[i] + scorer_bottom + '.h5')

            # todo testify
            # _bottom = _bottom.droplevel('scorer', axis=1)
            # bottom_manual = pd.read_hdf(
            #     '/media/mahdi/LaCie/Mahdi/DLC/projects/project_1/VV2Dpose-mahdi-2020-10-21/labeled-data/2_PG_1_VV_video/CollectedData_mahdi.h5')
            # bottom_manual = bottom_manual.droplevel('scorer', axis=1)

        else:
            # load data of side_LV and bottom view
            _side_LV = pd.read_hdf('/media/mahdi/LaCie/Mahdi/AniPose/LV/trial_6/pose-2d-filtered/{}_{}_{}_LV_video.h5'.format(fly_number, behaviour, video_sequence_number))
            _side_RV = pd.read_hdf('/media/mahdi/LaCie/Mahdi/AniPose/RV/trial_6/pose-2d-filtered/{}_{}_{}_RV_video.h5'.format(fly_number, behaviour, video_sequence_number))
            _bottom = pd.read_hdf('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/pose-2d-filtered/{}_{}_{}_VV_video.h5'.format(fly_number, behaviour, video_sequence_number))

        initial_number_frames = _bottom.shape[0]
        _side_LV = _side_LV.droplevel('scorer', axis=1)
        _side_RV = _side_RV.droplevel('scorer', axis=1)
        _bottom = _bottom.droplevel('scorer', axis=1)

        orientation_info = np.loadtxt(home_dir + videos_bottom[0] + 'orientation_info.txt', dtype=str)

        # todo keep -1 only to keep manually labeled
        angle = orientation_info[:-1, 3]
        cx = orientation_info[:-1, 6]
        cy = orientation_info[:-1, 9]
        # todo uncomment to inspect nan to keep manually labeled
        # np.argwhere(np.isnan(_side_LV.loc[:, (slice(None), ['x'])].to_numpy()))


        # #############################
        # # plot fly and joints
        # import matplotlib.pyplot as plt
        # import matplotlib
        # import matplotlib.image as mpimg
        # fig, ax = plt.subplots()
        # img = mpimg.imread(orientation_info[1,0])
        # ax.imshow(img, cmap=matplotlib.cm.jet)
        # ax.scatter(_bottom.loc[:, (slice(None), ['x'])].to_numpy()[0,:], _bottom.loc[:, (slice(None), ['y'])].to_numpy()[0,:], marker='o', c='k')
        # # plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/test.png')
        # plt.show()
        # #############################

        def rotate(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.
            The angle should be given in radians.
            """
            angle = angle * np.pi / 180
            ox, oy = w / 2, h / 2
            px = point[:, ::2]
            py = point[:, 1::2]

            qx = ox + np.cos(angle).reshape(angle.size, 1) * np.subtract(px, ox) - np.sin(angle).reshape(angle.size,
                                                                                                         1) * np.subtract(
                py, oy) - (w / 2 - origin[0].reshape(origin[0].size, 1))
            qy = oy + np.sin(angle).reshape(angle.size, 1) * np.subtract(px, ox) + np.cos(angle).reshape(angle.size,
                                                                                                         1) * np.subtract(
                py, oy) - (h / 2 - origin[1].reshape(origin[1].size, 1))
            return qx, qy


        import math


        def rotate2(origin, point, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            angle = math.radians(angle)
            ox, oy = w / 2, h / 2
            px = point[:, ::2] + (w / 2 - origin[0])
            py = point[:, 1::2] + (h / 2 - origin[1])
            qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
            qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
            return qx, qy


        # # flip left and right side_LV due to prism reflection
        # _side_LV = flip_LR(_side_LV)

        # rotate back the VV image
        import cv2
        from scipy import ndimage
        import matplotlib.image as mpimg

        img = mpimg.imread(orientation_info[1, 0])
        h, w = img.shape

        # rotate back the VV joints
        qx, qy = rotate((cx.astype(float), cy.astype(float)), _bottom.loc[:, (slice(None), ['x', 'y'])].to_numpy(),
                        angle.astype(float))
        # qx, qy =rotate2((cx.astype(float)[0], cy.astype(float)[0]), _bottom.loc[:, (slice(None), ['x', 'y'])].to_numpy(), angle.astype(float)[0])
        _bottom.loc[:, (slice(None), ['x'])] = qx
        _bottom.loc[:, (slice(None), ['y'])] = qy
        # #############################
        # # plot fly and joints
        # import matplotlib.pyplot as plt
        # import matplotlib
        # fig, ax = plt.subplots(2,1)
        # M_tr = np.float32([[1, 0, -(w / 2 - cx.astype(float)[0])], [0, 1, -(h / 2 - cy.astype(float)[0])]])
        # img = ndimage.rotate(img, -angle.astype(float)[0], reshape=False)
        # img = cv2.warpAffine(img, M_tr, (w, h))
        # img_LV = mpimg.imread('/home/mahdi/Pictures/fly_1/AG/2/LV/1_AG_2_LV_00001.tiff')
        # # img = mpimg.imread(orientation_info[1,0])
        # ax[0].imshow(img, cmap=matplotlib.cm.jet)
        # ax[0].scatter(qx[0, :], qy[0, :], marker='o', c='k')
        # # plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/test.png')
        #
        # ax[1].imshow(img_LV, cmap=matplotlib.cm.jet)
        # ax[1].scatter(_side_LV.loc[:, (slice(None), ['x'])].to_numpy()[0, :], _side_LV.loc[:, (slice(None), ['y'])].to_numpy()[0, :], marker='o', c='k')
        #
        # plt.show()
        # #############################

        # todo comment only to keep manually labeled
        # # select for high confidence datapoints
        # _bottom, _side_LV, _side_RV, good_keypts, mask_kept_data = select_best_data(_bottom, _side_LV, _side_RV, th1, th2,
        #                                                                             leg_tips)
        # name_id_kept_frames = np.arange(0, initial_number_frames)[mask_kept_data] + 1

        # todo uncomment only to keep manually labeled
        name_id_kept_frames = np.arange(0, initial_number_frames) + 1

        # take only those frames where all keypoints on at least one side_LV are correct
        if nice_frames:  # 1 for training, 0 for prediction
            print('nice frames')
            # mask if only 2 keypoints of leg_tips are with likelihood 1.0
            mask = (good_keypts.loc[:, leg_tips[:3]].sum(1) == 2)

            _side_LV = _side_LV[mask].dropna()
            _side_RV = _side_RV[mask].dropna()
            _bottom = _bottom[mask].dropna()
            good_keypts = good_keypts.loc[mask, :]

        # frame indices
        index = _bottom.index.values
        _bottom = _bottom.reset_index()
        _side_LV = _side_LV.reset_index()
        _side_RV = _side_RV.reset_index()

        # align horizontally
        # if align:  # 1 for training and prediction, 0 for making of DLC video
        #     print('align')
        #     path_crop_pos = home_dir + crop_positions[i]
        #     path_img = data_dir + images_bottom[i]
        #     angle, c, img_rot, shape = procrustes.get_orientation(path_crop_pos, path_img, index)
        #     _bottom.loc[:, (slice(None), ['x', 'y'])] = \
        #         _bottom.loc[:, (slice(None), ['x', 'y'])].apply(
        #             lambda x: procrustes.center_and_align(x, np.radians(angle), np.array(shape), np.array(c)), axis=1)

        # we find the floor of side_LV view as the confident predicted maximum tarsal_claw per each image and subtract it from the y channel
        if register_floor:
            print('align with x-y plane')
            floor = 0
            floor_RV = 0
            unreliable_tips_mask_idx = []
            for ind in _side_LV.index:
                # for ind in range(58,60):
                try:
                    good_tips = _side_LV.loc[:, (np.asarray(LV_bodyparts)[arg_LV_common_bodypart], 'y')].iloc[:,
                                good_keypts.iloc[ind, :].to_numpy()].loc[
                        ind, (leg_tips, 'y')]
                    floor_new_LV = np.max(good_tips.to_numpy())
                    if ~np.isnan(floor_new_LV):
                        floor = floor_new_LV
                    _side_LV.loc[ind, (slice(None), 'y')] = floor - _side_LV.loc[ind, (slice(None), 'y')]

                    good_tips_RV = _side_RV.loc[:, (np.asarray(LV_bodyparts)[arg_LV_common_bodypart], 'y')].iloc[:,
                                   good_keypts.iloc[ind, :].to_numpy()].loc[
                        ind, (leg_tips, 'y')]
                    floor_new_RV = np.max(good_tips_RV.to_numpy())
                    if ~np.isnan(floor_new_RV):
                        floor_RV = floor_new_RV
                    _side_RV.loc[ind, (slice(None), 'y')] = floor_RV - _side_RV.loc[ind, (slice(None), 'y')]
                except:
                    unreliable_tips_mask_idx.append(ind)
                    continue

            # remove the frames with not good tips for any of 3 tarsal claws in each side view (if good_tips is empty)
            unreliable_tips_mask = np.ones(_side_LV.shape[0])
            unreliable_tips_mask[unreliable_tips_mask_idx] = 0
            unreliable_tips_mask = pd.Series(unreliable_tips_mask > 0)
            _bottom = _bottom[unreliable_tips_mask].dropna()
            _side_LV = _side_LV[unreliable_tips_mask].dropna()
            _side_RV = _side_RV[unreliable_tips_mask].dropna()
            # frame indices
            index = _bottom.index.values
            _bottom = _bottom.reset_index()
            _side_LV = _side_LV.reset_index()
            _side_RV = _side_RV.reset_index()

        else:
            floor = 0
            floor_RV = 0
            if fly_number == '1':
                horiz_crop_right_1 = 8
                horiz_crop_right_2 = 266
                horiz_crop_middle_1 = 348
                horiz_crop_middle_2 = 798
                horiz_crop_left_1 = 888
                horiz_crop_left_2 = 1162
            elif fly_number == '2':
                horiz_crop_right_1 = 20
                horiz_crop_right_2 = 286
                horiz_crop_middle_1 = 386
                horiz_crop_middle_2 = 858
                horiz_crop_left_1 = 926
                horiz_crop_left_2 = 1186
            elif fly_number == '3':
                pad = 25
                horiz_crop_right_1 = 32
                horiz_crop_right_2 = 290
                horiz_crop_middle_1 = 392
                horiz_crop_middle_2 = 830
                horiz_crop_left_1 = 950
                horiz_crop_left_2 = 1182
            elif fly_number == '4':
                horiz_crop_right_1 = 20
                horiz_crop_right_2 = 280
                horiz_crop_middle_1 = 398
                horiz_crop_middle_2 = 824
                horiz_crop_left_1 = 908
                horiz_crop_left_2 = 1166
            elif fly_number == '5':
                horiz_crop_right_1 = 18
                horiz_crop_right_2 = 286
                horiz_crop_middle_1 = 370
                horiz_crop_middle_2 = 826
                horiz_crop_left_1 = 916
                horiz_crop_left_2 = 1182
            elif fly_number == '6':
                horiz_crop_right_1 = 26
                horiz_crop_right_2 = 284
                horiz_crop_middle_1 = 376
                horiz_crop_middle_2 = 824
                horiz_crop_left_1 = 908
                horiz_crop_left_2 = 1184
            else:
                IOError('fly number properties not defined!')

            floor_new_LV = horiz_crop_left_2 - horiz_crop_left_1
            if ~np.isnan(floor_new_LV):
                floor = floor_new_LV
            _side_LV.loc[:, (slice(None), 'y')] = floor - _side_LV.loc[:, (slice(None), 'y')]

            floor_new_RV = horiz_crop_right_2 - horiz_crop_right_1
            if ~np.isnan(floor_new_RV):
                floor_RV = floor_new_RV
            _side_RV.loc[:, (slice(None), 'y')] = floor_RV - _side_RV.loc[:, (slice(None), 'y')]

            # convert & save to DF3D format
        side_LV_np = _side_LV.loc[:, (slice(None), ['x', 'y'])].to_numpy()
        # z_LV = _side_LV.loc[:, (slice(None), 'y')].to_numpy()
        z_LV_uncommon = _side_LV.loc[:, (slice(None), 'y')].to_numpy()[:, arg_LV_common_bodypart][:, :-3]
        z_RV_uncommon = _side_RV.loc[:, (slice(None), 'y')].to_numpy()[:, arg_LV_common_bodypart][:, :-3]
        z_LRV_common = (_side_RV.loc[:, (slice(None), 'y')].to_numpy()[:, arg_LV_common_bodypart][:, -3:] + _side_LV.loc[:,
                                                                                                            (slice(None),
                                                                                                             'y')].to_numpy()[
                                                                                                            :,
                                                                                                            arg_LV_common_bodypart][
                                                                                                            :, -3:]) / 2

        # approximate 3D poses for the abdomen A keypoints and scutellum
        z_scutellum = (_side_RV.loc[:, (slice(None), 'y')].to_numpy()[:,22] + _side_LV.loc[:, (slice(None), 'y')].to_numpy()[:,22])/2
        z_abdomen_A = (_side_RV.loc[:, (slice(None), 'y')].to_numpy()[:,23:] + _side_LV.loc[:, (slice(None), 'y')].to_numpy()[:,23:])/2

        x_scutellum = (_side_RV.loc[:, (slice(None), 'x')].to_numpy()[:, 22] + _side_LV.loc[:,
                                                                               (slice(None), 'x')].to_numpy()[:, 22]) / 2
        x_abdomen_A = (
                    _side_RV.loc[:, (slice(None), 'x')].to_numpy()[:, 23:] + _side_LV.loc[:, (slice(None), 'x')].to_numpy()[
                                                                             :, 23:])/2

        bottom_np = _bottom.loc[:, (slice(None), ['x', 'y'])].to_numpy()
        bottom_np = np.stack((bottom_np[:, ::2], bottom_np[:, 1::2]), axis=2)

        # y_scutellum and abdomens is approximated as the mean of 4 H and M coxa_body predictions
        y_scutellum = np.mean(bottom_np[:, [5, 10, 20, 25], 1], axis=1)
        y_abdomen_A = np.vstack((y_scutellum, y_scutellum, y_scutellum, y_scutellum, y_scutellum)).T

        xyz_abdomen_A = np.concatenate((np.expand_dims(x_abdomen_A, axis=2),np.expand_dims(y_abdomen_A, axis=2),np.expand_dims(z_abdomen_A, axis=2)),axis=2)
        xyz_scutellum = np.expand_dims(np.concatenate((np.expand_dims(x_scutellum, axis=1),np.expand_dims(y_scutellum, axis=1),np.expand_dims(z_scutellum, axis=1)),axis=1),axis=1)


        # build and order z values corresponding VV annotation
        _tmp_legs = np.concatenate((z_LV_uncommon[:, :15], z_RV_uncommon[:, :15]), axis=1)

        _tmp_31 = np.concatenate((_tmp_legs, z_LV_uncommon[:, 15].reshape(z_LV_uncommon[:, 15].size, 1)), axis=1)
        _tmp_32 = np.concatenate((_tmp_31, z_RV_uncommon[:, 15].reshape(z_RV_uncommon[:, 15].size, 1)), axis=1)

        _tmp_33 = np.concatenate((_tmp_32, z_LV_uncommon[:, 16].reshape(z_LV_uncommon[:, 16].size, 1)), axis=1)
        _tmp_34 = np.concatenate((_tmp_33, z_RV_uncommon[:, 16].reshape(z_RV_uncommon[:, 16].size, 1)), axis=1)

        _tmp_35 = np.concatenate((_tmp_34, z_LV_uncommon[:, 17].reshape(z_LV_uncommon[:, 17].size, 1)), axis=1)
        _tmp_36 = np.concatenate((_tmp_35, z_RV_uncommon[:, 17].reshape(z_RV_uncommon[:, 17].size, 1)), axis=1)

        _tmp_37 = np.concatenate((_tmp_36, z_LV_uncommon[:, 18].reshape(z_LV_uncommon[:, 18].size, 1)), axis=1)
        _tmp_38 = np.concatenate((_tmp_37, z_RV_uncommon[:, 18].reshape(z_RV_uncommon[:, 18].size, 1)), axis=1)

        z = np.concatenate((_tmp_38, z_LRV_common), axis=1)  # all

        # side_LV_np = np.stack((side_LV_np[:, ::2], side_LV_np[:, 1::2]), axis=2)

        points3d = np.concatenate((np.concatenate((bottom_np, z[:, :, None]), axis=2), xyz_scutellum, xyz_abdomen_A) , axis=1)

        if AniPose_filter_enable==False:
            np.save(home_dir + videos_bottom[i] + 'points3d_names_id.npy', name_id_kept_frames)
            np.save(home_dir + videos_bottom[i] + 'points3d.npy', points3d)
            print(home_dir + videos_bottom[i] + 'points3d.npy' + ' successfully saved.')

        else:
            np.save('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d_names_id.npy', name_id_kept_frames)
            np.save('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d.npy', points3d)
            print('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d.npy' + ' successfully saved.')

        save_victor = False
        if save_victor:
            np.save('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/victor_final/{}_{}_{}_'.format(fly_number, behaviour,
                                                                                                  video_sequence_number) + '{}_AniPose_points3d_names_id.npy'.format(VV_net_name),
                    name_id_kept_frames[5:])
            np.save('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/victor_final/{}_{}_{}_'.format(fly_number, behaviour,
                                                                                                  video_sequence_number) + '{}_AniPose_points3d.npy'.format(VV_net_name),
                    points3d[5:,:,:])



if __name__ == "__main__":
    import traceback

    # fly_number=range(1,6+1,1)
    # behaviour=['AG','FW','PG','PE']
    # video_sequence_number=range(1,20+1,1)

    fly_number= [2]
    behaviour=['PG']
    video_sequence_number= [1]
    AniPose_filter_enable = False
    # VV_net_name = 'resnet152_VV2DposeOct21shuffle1_300000'
    # LV_net_name = 'resnet152_LV2DposeOct23shuffle1_490000'
    VV_net_name = 'gt_labels'
    LV_net_name = 'gt_labels'
    for _fly_number in zip(fly_number):
        for _behaviour in zip(behaviour):
            for _video_sequence_number in zip(video_sequence_number):
                try:
                    main(str(_fly_number[0]), _behaviour[0], str(_video_sequence_number[0]),AniPose_filter_enable, VV_net_name, LV_net_name)
                except:
                    traceback.print_exc()
                    continue

