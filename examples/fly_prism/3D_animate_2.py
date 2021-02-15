import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
import cv2
import matplotlib.image as mpimg

VV_bodyparts = ['LF_body_coxa', 'LF_coxa_femur', 'LF_femur_tibia', 'LF_tibia_tarsus', 'LF_tarsal_claw', 'LM_body_coxa',
                'LM_coxa_femur', 'LM_femur_tibia', 'LM_tibia_tarsus', 'LM_tarsal_claw', 'LH_body_coxa', 'LH_coxa_femur',
                'LH_femur_tibia', 'LH_tibia_tarsus', 'LH_tarsal_claw', 'RF_body_coxa', 'RF_coxa_femur',
                'RF_femur_tibia', 'RF_tibia_tarsus', 'RF_tarsal_claw', 'RM_body_coxa', 'RM_coxa_femur',
                'RM_femur_tibia', 'RM_tibia_tarsus', 'RM_tarsal_claw', 'RH_body_coxa', 'RH_coxa_femur',
                'RH_femur_tibia', 'RH_tibia_tarsus', 'RH_tarsal_claw', 'L_antenna', 'R_antenna', 'L_eye', 'R_eye',
                'L_haltere', 'R_haltere', 'L_wing', 'R_wing', 'proboscis', 'neck', 'genitalia', 'scutellum', 'A1A2', 'A3', 'A4', 'A5', 'A6']

LV_bodyparts = ['F_body_coxa', 'F_coxa_femur', 'F_femur_tibia', 'F_tibia_tarsus', 'F_tarsal_claw', 'M_body_coxa',
                'M_coxa_femur', 'M_femur_tibia', 'M_tibia_tarsus', 'M_tarsal_claw', 'H_body_coxa', 'H_coxa_femur',
                'H_femur_tibia', 'H_tibia_tarsus', 'H_tarsal_claw', 'antenna', 'eye', 'haltere', 'wing',
                'proboscis', 'neck', 'genitalia', 'scutellum', 'A1A2', 'A3', 'A4', 'A5', 'A6']

def main(fly_number,behaviour,video_sequence_number, AniPose_filter_enable=False, VV_net_name=None, LV_net_name=None):
    idx_LF_body_coxa = 0
    idx_LF_coxa_femur = 1
    idx_LF_femur_tibia = 2
    idx_LF_tibia_tarsus = 3
    idx_LF_tarsal_claw = 4
    idx_LM_body_coxa = 5
    idx_LM_coxa_femur = 6
    idx_LM_femur_tibia = 7
    idx_LM_tibia_tarsus = 8
    idx_LM_tarsal_claw = 9
    idx_LH_body_coxa = 10
    idx_LH_coxa_femur = 11
    idx_LH_femur_tibia = 12
    idx_LH_tibia_tarsus = 13
    idx_LH_tarsal_claw = 14
    idx_RF_body_coxa = 15
    idx_RF_coxa_femur = 16
    idx_RF_femur_tibia = 17
    idx_RF_tibia_tarsus = 18
    idx_RF_tarsal_claw = 19
    idx_RM_body_coxa = 20
    idx_RM_coxa_femur = 21
    idx_RM_femur_tibia = 22
    idx_RM_tibia_tarsus = 23
    idx_RM_tarsal_claw = 24
    idx_RH_body_coxa = 25
    idx_RH_coxa_femur = 26
    idx_RH_femur_tibia = 27
    idx_RH_tibia_tarsus = 28
    idx_RH_tarsal_claw = 29
    idx_L_antenna = 30
    idx_R_antenna = 31
    idx_L_eye = 32
    idx_R_eye = 33
    idx_L_haltere = 34
    idx_R_haltere = 35
    idx_L_wing = 36
    idx_R_wing = 37
    idx_proboscis = 38
    idx_neck = 39
    idx_genitalia = 40
    idx_scutellum = 41
    idx_A1A2 = 42
    idx_A3 = 43
    idx_A4 = 44
    idx_A5 = 45
    idx_A6 = 46

    home_dir = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_{}_clipped/{}/{}'.format(fly_number, behaviour,
                                                                                       video_sequence_number)
    if AniPose_filter_enable == False:
        points3d = np.load('{}/VV/{}_points3d.npy'.format(home_dir, VV_net_name))
        points3d_names_id = np.load('{}/VV/points3d_names_id.npy'.format(home_dir))
    else:
        points3d = np.load('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d.npy'.format(VV_net_name))
        points3d_names_id = np.load('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d_names_id.npy')


    # temp TODO
    # points3d = np.load('/media/mahdi/LaCie/Mahdi/AniPose/tmp/1_FW_2_AniPose_points3d.npy')
    num_repeats = 1
    # num_repeats = 2
    # points3d = np.append(points3d, points3d, axis=0)

    # # temp TODO
    # points3d = points3d[:-20,:,:]
    # points3d_names_id = points3d_names_id[:-20]

    # # temp TODO
    # transfer points to new origin
    # new_origin = np.mean((points3d[:,5,:],points3d[:,10,:],points3d[:,20,:],points3d[:,25,:]),axis=0)
    # points3d = points3d - np.expand_dims(new_origin, 1)
    # # temp TODO
    # points3d = points3d[31:49, :, :]
    # points3d_names_id = points3d_names_id[31:49]
    # # temp TODO add one frame: linear interpolation first and last frames
    # points3d = np.append(points3d, [(points3d[0, :, :] + points3d[-1, :, :]) / 2], axis=0)
    # points3d_names_id = np.append(points3d_names_id, points3d_names_id[-1])
    # #
    # # temp TODO
    # N = 100
    # points3d = np.concatenate([points3d]*N)
    # points3d_names_id = np.concatenate([points3d_names_id]*N)
    # np.save('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/victor_final/{}_repeated_31_48_{}_{}_{}_'.format(N,fly_number, behaviour, video_sequence_number) + 'AniPose_points3d.npy', points3d)

    # temp TODO repeat only a chunck of video in the MIDDLE
    frame_start = 56
    frame_end = 132
    # transfer points to new origin
    new_origin = np.mean((points3d[:, 5, :], points3d[:, 10, :], points3d[:, 20, :], points3d[:, 25, :]), axis=0)
    points3d = points3d - np.expand_dims(new_origin, 1)
    # temp TODO
    points3d_middle = points3d[(frame_start-1):frame_end, :, :]
    points3d_names_id_middle = points3d_names_id[(frame_start-1):frame_end]
    # temp TODO add one frame: linear interpolation first and last frames
    points3d_middle = np.append(points3d_middle, [(points3d_middle[0, :, :] + points3d_middle[-1, :, :]) / 2], axis=0)
    points3d_names_id_middle = np.append(points3d_names_id_middle, points3d_names_id_middle[-1])
    # temp TODO
    N = 25
    points3d_middle = np.concatenate([points3d_middle] * N)
    points3d_names_id_middle = np.concatenate([points3d_names_id_middle] * N)
    points3d = np.concatenate([points3d[:(frame_start-1),:,:], points3d_middle, points3d[frame_end:,:,:]])
    points3d_names_id = np.concatenate([points3d_names_id[:(frame_start-1)], points3d_names_id_middle, points3d_names_id[frame_end:]])
    np.save(
        '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_{}_clipped/{}/{}/refined_results/{}_repeated_{}_{}_middle_{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number, N, frame_start, frame_end,
                                                                                                                     fly_number,
                                                                                                                     behaviour,
                                                                                                                     video_sequence_number) + 'points3d.npy',
        points3d)

    # points3d = np.append(np.repeat(points3d[17:38,:,:],N,axis=0), points3d[17:-20, :, :], axis=0)
    # points3d_names_id = np.append(np.repeat(points3d_names_id[17:38],N,axis=0), points3d_names_id[17:-20], axis=0)

    # # plot gaits
    # gaits = np.random.randint(2, size=(6, 10))
    # fig, (ax0, ax1) = plt.subplots(2, 1)
    # c = ax1.pcolor(gaits, edgecolors='k', linewidths=.1, cmap='binary')
    # ax1.set_title('thick edges')
    # fig.tight_layout()
    # plt.show()

    # # TODO find best concatenation frames
    # for data_idx in range(0,points3d.shape[0]):
    #     x_dist = np.arange(points3d.shape[0])
    #     y_dist = np.sum(np.linalg.norm(points3d[:,:30,:] - points3d[data_idx, :30, :], axis=2), axis=1)
    #     fig, ax = plt.subplots()
    #     ax.plot(x_dist, y_dist)
    #     ax.set(xlabel='data', ylabel='l2 distance',
    #            title='l2 distance of data frame = {} vs data'.format(data_idx+1))
    #     ax.grid()
    #     fig.savefig("test.png")
    #     plt.show()
    #     plt.close()

    # # TODO
    # x_dist = np.arange(points3d.shape[0])
    # y_dist = points3d[:,4, 2]
    # y_dist_2 = points3d[:,9, 2]
    # y_dist_3 = points3d[:,14, 2]
    # y_dist_R = points3d[:,19, 2]
    # y_dist_2_R = points3d[:,24, 2]
    # y_dist_3_R = points3d[:,29, 2]
    # fig, ax = plt.subplots()
    # ax.plot(x_dist, y_dist, label='LF_tarsal_claw')
    # ax.plot(x_dist, y_dist_2, label='LM_tarsal_claw')
    # ax.plot(x_dist, y_dist_3, label='LH_tarsal_claw')
    # ax.plot(x_dist, y_dist_R, linestyle='--', label='RF_tarsal_claw')
    # ax.plot(x_dist, y_dist_2_R, linestyle='--', label='RM_tarsal_claw')
    # ax.plot(x_dist, y_dist_3_R, linestyle='--', label='RH_tarsal_claw')
    # ax.set(xlabel='data', ylabel='z [px]',
    #        title='predicted z coordinates of tarsal claws')
    # ax.grid()
    # ax.legend()
    # fig.savefig("test.png")
    # plt.show()

    x = points3d[:,:,0]
    y= points3d[:,:,1]
    z= points3d[:,:,2]

    if AniPose_filter_enable == False:
        # load data of side_LV and bottom view
        _side_LV = pd.read_hdf(home_dir + '/LV' + '/{}_{}_{}_LV_videoDLC_{}'.format(fly_number,behaviour,video_sequence_number, LV_net_name) + '.h5')
        _side_RV = pd.read_hdf(home_dir + '/RV' + '/{}_{}_{}_RV_videoDLC_{}'.format(fly_number,behaviour,video_sequence_number, LV_net_name) + '.h5')
        _bottom = pd.read_hdf(home_dir + '/VV' + '/{}_{}_{}_VV_videoDLC_{}'.format(fly_number,behaviour,video_sequence_number, VV_net_name) + '.h5')
    else:
        _side_LV = pd.read_hdf('/media/mahdi/LaCie/Mahdi/AniPose/LV/trial_6/pose-2d-filtered/{}_{}_{}_LV_video.h5'.format(fly_number, behaviour, video_sequence_number))
        _side_RV = pd.read_hdf('/media/mahdi/LaCie/Mahdi/AniPose/RV/trial_6/pose-2d-filtered/{}_{}_{}_RV_video.h5'.format(fly_number, behaviour, video_sequence_number))
        _bottom = pd.read_hdf('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/pose-2d-filtered/{}_{}_{}_VV_video.h5'.format(fly_number, behaviour, video_sequence_number))

    initial_number_frames = _bottom.shape[0]
    _side_LV = _side_LV.droplevel('scorer', axis=1)
    _side_RV = _side_RV.droplevel('scorer', axis=1)
    _bottom = _bottom.droplevel('scorer', axis=1)

    # load 2D used for 3D pose estimation
    x_VV_2D = _bottom.loc[:, (slice(None), ['x'])].to_numpy()[points3d_names_id-1,:]
    y_VV_2D = _bottom.loc[:, (slice(None), ['y'])].to_numpy()[points3d_names_id-1,:]
    x_LV_2D = _side_LV.loc[:, (slice(None), ['x'])].to_numpy()[points3d_names_id-1,:]
    y_LV_2D = _side_LV.loc[:, (slice(None), ['y'])].to_numpy()[points3d_names_id-1,:]
    x_RV_2D = _side_RV.loc[:, (slice(None), ['x'])].to_numpy()[points3d_names_id-1,:]
    y_RV_2D = _side_RV.loc[:, (slice(None), ['y'])].to_numpy()[points3d_names_id-1,:]


    register_floor= False
    if register_floor:
        pass
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

        # floor_new_LV = horiz_crop_left_2 - horiz_crop_left_1
        # floor_new_RV = horiz_crop_right_2 - horiz_crop_right_1


    def update_graph(num):
        graph_FL.set_data (x[num,:5], y[num,:5])
        graph_FL.set_3d_properties(z[num,:5])

        graph_ML.set_data (x[num,5:10], y[num,5:10])
        graph_ML.set_3d_properties(z[num,5:10])

        graph_HL.set_data (x[num,10:15], y[num,10:15])
        graph_HL.set_3d_properties(z[num,10:15])

        graph_FR.set_data(x[num, 15:20], y[num, 15:20])
        graph_FR.set_3d_properties(z[num, 15:20])

        graph_MR.set_data(x[num, 20:25], y[num, 20:25])
        graph_MR.set_3d_properties(z[num, 20:25])

        graph_HR.set_data(x[num, 25:30], y[num, 25:30])
        graph_HR.set_3d_properties(z[num, 25:30])

        graph_Lhead.set_data(np.hstack((x[num,idx_neck],x[num,idx_L_eye],x[num,idx_L_antenna])), np.hstack((y[num,idx_neck],y[num,idx_L_eye],y[num,idx_L_antenna])))
        graph_Lhead.set_3d_properties(np.hstack((z[num,idx_neck],z[num,idx_L_eye],z[num,idx_L_antenna])))

        graph_Rhead.set_data(np.hstack((x[num,idx_neck],x[num,idx_R_eye],x[num,idx_R_antenna])), np.hstack((y[num,idx_neck],y[num,idx_R_eye],y[num,idx_R_antenna])))
        graph_Rhead.set_3d_properties(np.hstack((z[num,idx_neck],z[num,idx_R_eye],z[num,idx_R_antenna])))

        graph_neck_proboscis.set_data(np.hstack((x[num,idx_neck],x[num,idx_proboscis])), np.hstack((y[num,idx_neck],y[num,idx_proboscis])))
        graph_neck_proboscis.set_3d_properties(np.hstack((z[num,idx_neck],z[num,idx_proboscis])))

        graph_back_L.set_data(np.hstack((x[num,idx_L_wing],x[num,idx_scutellum],x[num,idx_L_haltere])), np.hstack((y[num,idx_L_wing],y[num,idx_scutellum],y[num,idx_L_haltere])))
        graph_back_L.set_3d_properties(np.hstack((z[num,idx_L_wing],z[num,idx_scutellum],z[num,idx_L_haltere])))

        graph_back_R.set_data(np.hstack((x[num,idx_R_wing],x[num,idx_scutellum],x[num,idx_R_haltere])), np.hstack((y[num,idx_R_wing],y[num,idx_scutellum],y[num,idx_R_haltere])))
        graph_back_R.set_3d_properties(np.hstack((z[num,idx_R_wing],z[num,idx_scutellum],z[num,idx_R_haltere])))

        graph_abdomen.set_data(np.hstack((x[num,idx_A1A2],x[num,idx_A3],x[num,idx_A4],x[num,idx_A5],x[num,idx_A6],x[num,idx_genitalia])), np.hstack((y[num,idx_A1A2],y[num,idx_A3],y[num,idx_A4],y[num,idx_A5],y[num,idx_A6],y[num,idx_genitalia])))
        graph_abdomen.set_3d_properties(np.hstack((z[num,idx_A1A2],z[num,idx_A3],z[num,idx_A4],z[num,idx_A5],z[num,idx_A6],z[num,idx_genitalia])))

        # graph_back_R_scatter.set_data(np.hstack((x[num,idx_R_wing],x[num,idx_R_haltere])), np.hstack((y[num,idx_R_wing],y[num,idx_R_haltere])))
        # graph_back_R_scatter.set_3d_properties(np.hstack((z[num,idx_R_wing],z[num,idx_R_haltere])))

        # graph_back_R_color.set_data((x[num, idx_R_haltere], y[num, idx_R_haltere]))
        # graph_back_R_color.set_3d_properties(z[num, idx_R_haltere])

        graph_keypoints_scatter._offsets3d = (x[num, :], y[num, :], z[num, :])

        title.set_text('points3d, frame={}'.format(num+1))

        num_img = num%(total_number_data//num_repeats-1)
        im_LV_predictions.set_offsets(np.vstack((x_LV_2D[num_img, :],y_LV_2D[num_img, :])).T)
        im_RV_predictions.set_offsets(np.vstack((x_RV_2D[num_img, :],y_RV_2D[num_img, :])).T)
        im_VV_predictions.set_offsets(np.vstack((x_VV_2D[num_img, :],y_VV_2D[num_img, :])).T)


        try:
            img_name_LV = home_dir+'/LV/{}_{}_{}_LV_{:05d}.tiff'.format(fly_number,behaviour,video_sequence_number,points3d_names_id[num_img])
            title_LV.set_text('LV, frame={}'.format(num + 1))
            img_LV = mpimg.imread(img_name_LV)
            im_LV.set_array(img_LV)

            img_name_RV = home_dir+'/RV/{}_{}_{}_RV_{:05d}.tiff'.format(fly_number,behaviour,video_sequence_number,points3d_names_id[num_img])
            title_RV.set_text('RV, frame={}'.format(num + 1))
            img_RV = mpimg.imread(img_name_RV)
            im_RV.set_array(img_RV)

            img_name_VV = home_dir+'/VV/{}_{}_{}_VV_{:05d}.tiff'.format(fly_number,behaviour,video_sequence_number,points3d_names_id[num_img])
            title_VV.set_text('VV, frame={}'.format(num + 1))
            img_VV = mpimg.imread(img_name_VV)
            im_VV.set_array(img_VV)


        except:
            pass

        # ax.set_xlim((100, 600))
        # ax.set_ylim((50, 500))
        # ax.set_zlim((-25, 150))

        return title, graph_FL, graph_ML, graph_HL, graph_FR, graph_MR, graph_HR, graph_Lhead, graph_back_L, graph_back_R,  im_LV, title_LV, im_RV, title_RV, im_VV, title_VV, graph_keypoints_scatter, im_LV_predictions, im_RV_predictions, im_VV_predictions, graph_abdomen, graph_neck_proboscis

    fig = plt.figure(figsize=(16, 12))

    from matplotlib import gridspec
    gs = gridspec.GridSpec(3, 4)

    ax = fig.add_subplot(gs[:,1:4], projection='3d', aspect='equal')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    ax.set_ylabel('y')


    title = ax.set_title('points3d')

    graph_FL, = ax.plot(x[0,:5], y[0,:5], z[0,:5], linestyle="-", marker="", color='gray')
    graph_ML, = ax.plot(x[0,5:10], y[0,5:10], z[0,5:10], linestyle="-", marker="", color='gray')
    graph_HL, = ax.plot(x[0,10:15], y[0,10:15], z[0,10:15], linestyle="-", marker="", color='gray')

    graph_FR, = ax.plot(x[0,:5], y[0,:5], z[0,:5], linestyle="-", marker="", color='k')
    graph_MR, = ax.plot(x[0,5:10], y[0,5:10], z[0,5:10], linestyle="-", marker="", color='k')
    graph_HR, = ax.plot(x[0,10:15], y[0,10:15], z[0,10:15], linestyle="-", marker="", color='k')

    graph_Lhead, = ax.plot(np.hstack((x[0,idx_neck],x[0,idx_L_eye],x[0,idx_L_antenna])), np.hstack((y[0,idx_neck],y[0,idx_L_eye],y[0,idx_L_antenna])), np.hstack((z[0,idx_neck],z[0,idx_L_eye],z[0,idx_L_antenna])), linestyle="-", marker="", color='gray')
    graph_Rhead, = ax.plot(np.hstack((x[0,idx_neck],x[0,idx_R_eye],x[0,idx_R_antenna])), np.hstack((y[0,idx_neck],y[0,idx_R_eye],y[0,idx_R_antenna])), np.hstack((z[0,idx_neck],z[0,idx_R_eye],z[0,idx_R_antenna])), linestyle="-", marker="", color='k')

    graph_neck_proboscis, = ax.plot(np.hstack((x[0,idx_neck],x[0,idx_proboscis])), np.hstack((y[0,idx_neck],y[0,idx_proboscis])), np.hstack((z[0,idx_neck],z[0,idx_proboscis])), linestyle="-", marker="", color='b')

    graph_back_L, = ax.plot(np.hstack((x[0,idx_L_wing],x[0,idx_scutellum],x[0,idx_L_haltere])), np.hstack((y[0,idx_L_wing],y[0,idx_scutellum],y[0,idx_L_haltere])), np.hstack((z[0,idx_L_wing],z[0,idx_scutellum],z[0,idx_L_haltere])), linestyle="-", marker="", color='gray')

    graph_back_R, = ax.plot(np.hstack((x[0,idx_R_wing],x[0,idx_scutellum],x[0,idx_R_haltere])), np.hstack((y[0,idx_R_wing],y[0,idx_scutellum],y[0,idx_R_haltere])), np.hstack((z[0,idx_R_wing],z[0,idx_scutellum],z[0,idx_R_haltere])), linestyle="-", marker="", color='k')

    graph_abdomen, = ax.plot(np.hstack((x[0,idx_A1A2],x[0,idx_A3],x[0,idx_A4],x[0,idx_A5],x[0,idx_A6],x[0,idx_genitalia])), np.hstack((y[0,idx_A1A2],y[0,idx_A3],y[0,idx_A4],y[0,idx_A5],y[0,idx_A6],y[0,idx_genitalia])), np.hstack((z[0,idx_A1A2],z[0,idx_A3],z[0,idx_A4],z[0,idx_A5],z[0,idx_A6],z[0,idx_genitalia])), linestyle="-", marker="", color='b')


    # graph_back_R_color, = ax.plot(x[0,idx_R_haltere], y[0,idx_R_haltere], z[0,idx_R_haltere], linestyle="", marker="", color='k', markerfacecolor='r')

    # graph_back_R_scatter, = ax.plot(np.hstack((x[0,idx_R_haltere],x[0,idx_R_wing])), np.hstack((y[0,idx_R_haltere],y[0,idx_R_wing])), np.hstack((z[0,idx_R_haltere],z[0,idx_R_wing])), linestyle="-", marker="", color='k', markerfacecolor='r')

    # ax2 = fig.add_subplot(gs[:,0:3], projection='3d')

    colors=np.arange(x.shape[1])
    graph_keypoints_scatter = ax.scatter(x[0,:], y[0,:], z[0,:], marker='o', zorder=2, c=colors, cmap=matplotlib.cm.jet)
    cbar = fig.colorbar(graph_keypoints_scatter, ax=ax, spacing="proportional", ticks=np.arange(len(VV_bodyparts)))
    cbar.set_ticklabels(VV_bodyparts)

    # graph_Rhead, = ax.plot(np.hstack((x[0,idx_neck],[x[0,idx_R_eye]],[x[0,idx_R_antenna]])), np.hstack((y[0,idx_neck],[y[0,idx_R_eye]],[y[0,idx_R_antenna]])), np.hstack((z[0,idx_neck],[z[0,idx_R_eye]],[z[0,idx_R_antenna]])), linestyle="-", marker="", color='g')

    # graph_Rhead, = ax.plot(np.hstack((x[0,idx_neck],[x[0,idx_L_eye]])), np.hstack((y[0,idx_neck],[y[0,idx_L_eye]])), np.hstack((z[0,idx_neck],[z[0,idx_L_eye]])), linestyle="-", marker="", color='g')


    labels = ["LV", "RV"]
    handles = [graph_back_L, graph_back_R]
    plt.legend(handles, labels)


    # ax_LV = fig.add_subplot(212)
    img_name_RV = home_dir+'/RV/{}_{}_{}_RV_{:05d}.tiff'.format(fly_number,behaviour,video_sequence_number, points3d_names_id[0])
    img_RV = mpimg.imread(img_name_RV)
    # ax_RV = plt.subplot(gs[2,3:4])
    ax_RV = fig.add_subplot(gs[0,0:1])
    # plt.imshow(img_RV, cmap='gray')
    title_RV = plt.title('RV')
    im_RV = ax_RV.imshow(img_RV, animated=True, cmap='gray')
    colors_RV=np.arange(x_RV_2D.shape[1])
    im_RV_predictions = ax_RV.scatter(x_RV_2D[0, :], y_RV_2D[0, :], zorder=2, s=3, c=colors_RV, cmap=matplotlib.cm.Wistia)


    # ax_VV = fig.add_subplot(212)
    img_name_VV = home_dir+'/VV/{}_{}_{}_VV_{:05d}.tiff'.format(fly_number,behaviour,video_sequence_number, points3d_names_id[0])
    img_VV = mpimg.imread(img_name_VV)
    # ax_VV = plt.subplot(gs[2,3:4])
    ax_VV = fig.add_subplot(gs[1,0:1])
    # plt.imshow(img_VV, cmap='gray')
    title_VV = plt.title('VV')
    im_VV = ax_VV.imshow(img_VV, animated=True, cmap='gray')
    colors_VV=np.arange(x_VV_2D.shape[1])
    im_VV_predictions = ax_VV.scatter(x_VV_2D[0, :], y_VV_2D[0, :], zorder=2, s=3, c=colors_VV, cmap=matplotlib.cm.jet)


    # ax_LV = fig.add_subplot(212)
    img_name_LV = home_dir+'/LV/{}_{}_{}_LV_{:05d}.tiff'.format(fly_number,behaviour,video_sequence_number,points3d_names_id[0])
    img_LV = mpimg.imread(img_name_LV)
    # ax_LV = plt.subplot(gs[2,3:4])
    ax_LV = fig.add_subplot(gs[2,0:1])
    # plt.imshow(img_LV, cmap='gray')
    title_LV = plt.title('LV')
    im_LV = ax_LV.imshow(img_LV, animated=True, cmap='gray')
    colors_LV=np.arange(x_LV_2D.shape[1])
    im_LV_predictions = ax_LV.scatter(x_LV_2D[0, :], y_LV_2D[0, :], zorder=2, s=3, c=colors_LV, cmap=matplotlib.cm.Wistia)
    cbar_LV = fig.colorbar(im_LV_predictions, ax=ax_LV, spacing="proportional", ticks=np.arange(len(LV_bodyparts)), orientation='horizontal')
    plt.setp(cbar_LV.ax.get_xticklabels(),rotation=90)
    cbar_LV.set_ticklabels(LV_bodyparts)

    total_number_data = x.shape[0]
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, total_number_data-1,
                                   interval=.1, blit=False)
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max())
    ax.set_zlim(z.min(),z.max())

    plt.show()

    # Set up formatting for the movie files
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=2, metadata=dict(artist='Me'))
    if AniPose_filter_enable==False:
        ani.save('{}/VV/points3d_animation.mp4'.format(home_dir), writer=writer)
        print('{}/VV/points3d_animation.mp4'.format(home_dir) + ' successfully saved.')

    else:
        ani.save('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_repeated_2_16_middle_{}_{}_{}_'.format(N,fly_number, behaviour, video_sequence_number) + 'AniPose_points3d_animation.mp4', writer=writer)
        print('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_6/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d_animation.mp4' + ' successfully saved.')


if __name__ == "__main__":

    import traceback

    # fly_number=range(1,6+1,1)
    # behaviour=['AG','FW','PG','PE']
    # video_sequence_number=range(1,20+1,1)

    AniPose_filter_enable = False
    fly_number= [4]
    behaviour=['PG']
    video_sequence_number= [4]
    # VV_net_name = 'resnet152_VV2DposeOct21shuffle1_300000'
    # LV_net_name = 'resnet152_LV2DposeOct23shuffle1_490000'

    VV_net_name = 'gt_labels'
    LV_net_name = 'gt_labels'

    for _fly_number in zip(fly_number):
        for _behaviour in zip(behaviour):
            for _video_sequence_number in zip(video_sequence_number):
                try:
                    main(str(_fly_number[0]), _behaviour[0], str(_video_sequence_number[0]), AniPose_filter_enable, VV_net_name, LV_net_name)
                except:
                    traceback.print_exc()
                    continue