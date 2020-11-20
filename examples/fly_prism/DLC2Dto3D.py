#

import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import pickle
import src.procrustes as procrustes
import src.utils as utils

#

home_dir = '/home/mahdi/Pictures/fly_1'
data_dir = '/home/mahdi/Pictures/fly_1/AG/1'
scorer_bottom = '1_AG_2_VV_videoDLC_resnet50_VV2DposeOct21shuffle1_390000'
scorer_side_LV = '1_AG_2_LV_videoDLC_resnet50_LV2DposeOct23shuffle1_405000'
scorer_side_RV = '1_AG_2_LV_videoDLC_resnet50_LV2DposeOct23shuffle1_405000'


# joints
leg_tips = ['LF_tarsal_claw', 'LM_tarsal_claw', 'LH_tarsal_claw',
            'RF_tarsal_claw', 'RM_tarsal_claw', 'RH_tarsal_claw']

coxa_femurs = ['coxa-femur front L', 'coxa-femur mid L', 'coxa-femur back L',
               'coxa-femur front R', 'coxa-femur mid R', 'coxa-femur back R']

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
arg_VV_common_bodypart=[]
arg_LV_common_bodypart=[]
for idx, item in enumerate(LV_bodyparts):
    try:
        arg_VV_common_bodypart.append(np.where(np.asarray(VV_bodyparts) == item)[0][0])
        arg_LV_common_bodypart.append(idx)
    except:
        pass

# # lateral images of enclosure
# images_side_LV = ['191125_PR/Fly1/001_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly1/',
#                '191125_PR/Fly1/002_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly1/',
#                '191125_PR/Fly1/003_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly1/',
#                '191125_PR/Fly1/004_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly1/',
#                '191125_PR/Fly2/001_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly2/',
#                '191125_PR/Fly2/002_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly2/',
#                '191125_PR/Fly2/003_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly2/',
#                '191125_PR/Fly2/004_prism/behData/images/side_LV_view_prism_data_191125_PR_Fly2/']

# # ventral images of enclosure
# images_bottom = ['191125_PR/Fly1/001_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1/',
#                  '191125_PR/Fly1/002_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1/',
#                  '191125_PR/Fly1/003_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1/',
#                  '191125_PR/Fly1/004_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1/',
#                  '191125_PR/Fly2/001_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2/',
#                  '191125_PR/Fly2/002_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2/',
#                  '191125_PR/Fly2/003_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2/',
#                  '191125_PR/Fly2/004_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2/']
#
# # position of crop around moving fly
# crop_positions = ['/bottom_view/videos/crop_loc_191125_PR_Fly1_001_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly1_002_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly1_003_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly1_004_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly2_001_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly2_002_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly2_003_prism.txt',
#                   '/bottom_view/videos/crop_loc_191125_PR_Fly2_004_prism.txt']

# lateral cropped video of moving fly
videos_side_LV = ['/AG/2/LV/']
videos_side_RV = ['/AG/2/RV/']

# ventral cropped video of moving fly
videos_bottom = ['/AG/2/VV/']


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
    register_floor = 1


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

def select_best_data(bottom, side_LV, th1, th2, leg_tips):
    # select those frames with high confidence ventral view if all the tarsal_claw liklihood is larger than th1
    bottom_lk = bottom.loc[:, (leg_tips, 'likelihood')]
    mask = (bottom_lk > th1).sum(1) == 6
    bottom = bottom[mask].dropna()
    side_LV = side_LV[mask].dropna()

    # find high confidence and low discrepancy keypoints in each frame
    likelihood = side_LV.loc[:, (np.asarray(LV_bodyparts)[arg_LV_common_bodypart], 'likelihood')]
    discrepancy = np.abs(bottom.loc[:, (slice(None), 'x')].values[:,arg_VV_common_bodypart] - side_LV.loc[:, (slice(None), 'x')].values[:,arg_LV_common_bodypart])
    # find good keypoints corresponding to np.asarray(LV_bodyparts)[arg_LV_common_bodypart]
    good_keypts = (likelihood > th1) & (discrepancy < th2)
    good_keypts = good_keypts.droplevel(1, axis=1)

    assert side_LV.shape[0] == bottom.shape[0], 'Number of rows(=number of data) must match in filtered data!'

    return bottom, side_LV, good_keypts


def flip_LR(data):
    cols = list(data.columns)
    half = int(len(cols) / 2)
    tmp = data.loc[:, cols[:half]].values
    data.loc[:, cols[:half]] = data.loc[:, cols[half:]].values
    data.loc[:, cols[half:]] = tmp

    return data


for i in range(len(videos_side_LV)):
    print(home_dir + videos_side_LV[i])

    # load data of side_LV and bottom view
    _side_LV = pd.read_hdf(home_dir + videos_side_LV[i] + scorer_side_LV + '.h5')
    _bottom = pd.read_hdf(home_dir + videos_bottom[i] + scorer_bottom + '.h5')
    _side_LV = _side_LV.droplevel('scorer', axis=1)
    _bottom = _bottom.droplevel('scorer', axis=1)

    orientation_info = np.loadtxt(home_dir + videos_bottom[0] + 'orientation_info.txt', dtype=str)
    angle = orientation_info[:, 3]
    cx = orientation_info[:, 6]
    cy = orientation_info[:, 9]

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
        angle = angle*np.pi/180
        ox, oy = w/2, h/2
        px = point[:,::2]
        py = point[:,1::2]

        qx = ox + np.cos(angle).reshape(angle.size,1) * np.subtract(px,ox) - np.sin(angle).reshape(angle.size,1) * np.subtract(py,oy) - (w / 2 - origin[0].reshape(origin[0].size,1))
        qy = oy + np.sin(angle).reshape(angle.size,1) * np.subtract(px,ox) + np.cos(angle).reshape(angle.size,1) * np.subtract(py,oy) - (h / 2 - origin[1].reshape(origin[1].size,1))
        return qx, qy


    import math


    def rotate2(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        angle = math.radians(angle)
        ox, oy = w/2, h/2
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
    img = mpimg.imread(orientation_info[1,0])
    h, w = img.shape
    M_tr = np.float32([[1, 0, -(w / 2 - cx.astype(float)[0])], [0, 1, -(h / 2 - cy.astype(float)[0])]])
    img = ndimage.rotate(img, -angle.astype(float)[0], reshape=False)
    img = cv2.warpAffine(img, M_tr, (w, h))


    # rotate back the VV joints
    qx, qy =rotate((cx.astype(float), cy.astype(float)), _bottom.loc[:, (slice(None), ['x', 'y'])].to_numpy(), angle.astype(float))
    # qx, qy =rotate2((cx.astype(float)[0], cy.astype(float)[0]), _bottom.loc[:, (slice(None), ['x', 'y'])].to_numpy(), angle.astype(float)[0])
    _bottom.loc[:, (slice(None), ['x'])] = qx
    _bottom.loc[:, (slice(None), ['y'])] = qy



    #############################
    # plot fly and joints
    import matplotlib.pyplot as plt
    import matplotlib
    fig, ax = plt.subplots()
    # img = mpimg.imread(orientation_info[1,0])
    ax.imshow(img, cmap=matplotlib.cm.jet)
    ax.scatter(qx[0, :], qy[0, :], marker='o', c='k')
    # plt.savefig('/home/mahdi/HVR/git_repos/deep-prior-pp/src/cache/test.png')
    plt.show()
    #############################

    # select for high confidence datapoints
    _bottom, _side_LV, good_keypts = select_best_data(_bottom, _side_LV, th1, th2, leg_tips)

    # take only those frames where all keypoints on at least one side_LV are correct
    if nice_frames:  # 1 for training, 0 for prediction
        print('nice frames')
        # mask if only 2 keypoints of leg_tips are with likelihood 1.0
        mask = (good_keypts.loc[:, leg_tips[:3]].sum(1) == 2)

        _side_LV = _side_LV[mask].dropna()
        _bottom = _bottom[mask].dropna()
        good_keypts = good_keypts.loc[mask, :]

    # frame indices
    index = _bottom.index.values
    _bottom = _bottom.reset_index()
    _side_LV = _side_LV.reset_index()

    # align horizontally
    # if align:  # 1 for training and prediction, 0 for making of DLC video
    #     print('align')
    #     path_crop_pos = home_dir + crop_positions[i]
    #     path_img = data_dir + images_bottom[i]
    #     angle, c, img_rot, shape = procrustes.get_orientation(path_crop_pos, path_img, index)
    #     _bottom.loc[:, (slice(None), ['x', 'y'])] = \
    #         _bottom.loc[:, (slice(None), ['x', 'y'])].apply(
    #             lambda x: procrustes.center_and_align(x, np.radians(angle), np.array(shape), np.array(c)), axis=1)

    # we find the floor of side_LV view as the confident predicted maximum trasal_claw per each image andsubtract it from the y channel
    if register_floor:
        print('align with x-y plane')
        floor = 0
        for ind in _side_LV.index:
            try:
                good_tips = _side_LV.loc[:, (np.asarray(LV_bodyparts)[arg_LV_common_bodypart], 'y')].iloc[:, good_keypts.iloc[ind, :].to_numpy()].loc[
                    ind, (leg_tips, 'y')]
                floor_new = np.max(good_tips.to_numpy())
                if ~np.isnan(floor_new):
                    floor = floor_new
                _side_LV.loc[ind, (slice(None), 'y')] = _side_LV.loc[ind, (slice(None), 'y')] - floor
            except:
                continue

    # convert & save to DF3D format
    side_LV_np = _side_LV.loc[:, (slice(None), ['x', 'y'])].to_numpy()
    z_LV = _side_LV.loc[:, (slice(None), 'y')].to_numpy()
    
    side_LV_np = np.stack((side_LV_np[:, ::2], side_LV_np[:, 1::2]), axis=2)

    bottom_np = _bottom.loc[:, (slice(None), ['x', 'y'])].to_numpy()
    bottom_np = np.stack((bottom_np[:, ::2], bottom_np[:, 1::2]), axis=2)
    points2d = np.stack((bottom_np, side_LV_np), axis=0)
    points3d = np.concatenate((bottom_np, -z[:, :, None]), axis=2)
    good_keypts = np.array(good_keypts)

    # # remove some bad frames manually
    # for b_frame in bad_frames[i]:
    #     points2d = np.delete(points2d, b_frame, 1)
    #     points3d = np.delete(points3d, b_frame, 0)
    #     index = np.delete(index, b_frame, 0)
    #     good_keypts = np.delete(good_keypts, b_frame, 0)

    if np.isnan(z).any():
        print('NaNs found, something went wrong...')

    poses = {'points2d': points2d,
             'points3d': points3d,
             'index': index,
             'good_keypts': good_keypts
             }

    pickle.dump(poses, open(home_dir + videos_side_LV[i].split('/')[-1][6:] + '.pkl', 'wb'))

# %%

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
# plt.xlim([100,800])
# plt.ylim([100,800])

# bottom = _bottom.loc[0,(slice(None),['x','y'])]#
# bottom_np = pts2d.to_numpy()[None,:]
# bottom_np = _bottom.loc[:,(slice(None),['x','y'])].to_numpy()
# bottom_np = np.stack((bottom_np[:,::2], bottom_np[:,1::2]), axis=2)

from skeleton import skeleton

G, color_edge = skeleton()
writer = FFMpegWriter(fps=10)
with writer.saving(fig, "cropped.mp4", 100):
    for frame_idx in tqdm(range(1500)):
        plt.cla()

        # plt.imshow(img_rot[frame_idx], cmap='gray')
        # plt.imshow(imgs[frame_idx], cmap='gray')

        utils.plot_skeleton(G, bottom_np[frame_idx, :, 0], bottom_np[frame_idx, :, 1], color_edge)
        plt.xlim([0, 400])
        plt.ylim([100, 400])

        # plt.text(120, 80, str(frame_idx), fontsize=20, color='white')

        # plt.axis('off')
        writer.grab_frame()

    # %%

shape

# %%

import matplotlib.pyplot as plt


def plot_skeleton(x, y, color_edge, ax=None, good_keypts=None):
    for i, j in enumerate(G.edges()):
        if good_keypts is not None:
            if (good_keypts[j[0]] == 0) | (good_keypts[j[1]] == 0):
                continue

        u = np.array((x[j[0]], x[j[1]]))
        v = np.array((y[j[0]], y[j[1]]))
        if ax is not None:
            ax.plot(u, v, c=color_edge[j[0]], alpha=1.0, linewidth=2)
        else:
            plt.plot(u, v, c=color_edge[j[0]], alpha=1.0, linewidth=2)


from skeleton import skeleton

G, color_edge = skeleton()
# cropped image
i = 0

fig = plt.figure(figsize=(6, 6))

bottom = _bottom.loc[:, (slice(None), ['x', 'y'])]

tmp = procrustes.center_and_align(x, angle[0], np.array(shape), np.array(c[0]), img_rot[0])

bottom_x = bottom[:, 0]
bottom_y = bottom[:, 1]

plt.imshow(img_rot[0])
plot_skeleton(bottom_x, bottom_y, color_edge)

# i = 50

# bottom_2 = poses['points2d'][0,i,:,:].copy()

# bottom_x = bottom_2[:,0]
# bottom_y = bottom_2[:,1]

# plot_skeleton(bottom_x, bottom_y, color_edge)


# %%
