import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
import cv2
import matplotlib.image as mpimg

VV_bodyparts = ['LF_body_coxa', 'LF_coxa_femur', 'LF_femur_tarsus', 'LF_tibia_tarsus', 'LF_tarsal_claw', 'LM_body_coxa',
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

def main(fly_number,behaviour,video_sequence_number, AniPose_filter_enable=False):
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

    # points3d = np.load('{}/VV/points3d.npy'.format(home_dir))
    # points3d_names_id = np.load('{}/VV/points3d_names_id.npy'.format(home_dir))
    points3d = np.load(
        '/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_2/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour,
                                                                                      video_sequence_number) + 'AniPose_points3d.npy')
    points3d_names_id = np.load(
        '/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_2/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour,
                                                                                      video_sequence_number) + 'AniPose_points3d_names_id.npy')
    points3d_filtered = np.load('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_3/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d.npy')
    points3d_names_id_filtered = np.load('/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_3/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour, video_sequence_number) + 'AniPose_points3d_names_id.npy')

    points3d_filtered_2 = np.load(
        '/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_5/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour,
                                                                                      video_sequence_number) + 'AniPose_points3d.npy')
    points3d_names_id_filtered_2 = np.load(
        '/media/mahdi/LaCie/Mahdi/AniPose/VV/trial_5/DLC_animations/{}_{}_{}_'.format(fly_number, behaviour,
                                                                                      video_sequence_number) + 'AniPose_points3d_names_id.npy')
    # labels = ["Resnet-152", "Resnet-50"]
    # labels = ["Resnet-50-without-filter","Resnet-50-with_median-filter"]
    labels = ["Resnet-152-median", "Resnet-152-viterbi", "Resnet-152-autoencoder"]

    x = points3d[:,:,0]
    y= points3d[:,:,1]
    z= points3d[:,:,2]

    # limb_lengths
    LF_coxa = np.linalg.norm((points3d[:,idx_LF_body_coxa,:] - points3d[:,idx_LF_coxa_femur,:]),axis=1)
    LF_femur = np.linalg.norm((points3d[:, idx_LF_coxa_femur, :] - points3d[:, idx_LF_femur_tibia, :]), axis=1)
    LF_tibia = np.linalg.norm((points3d[:, idx_LF_femur_tibia, :] - points3d[:, idx_LF_tibia_tarsus, :]), axis=1)
    LF_tarsus = np.linalg.norm((points3d[:, idx_LF_tibia_tarsus, :] - points3d[:, idx_LF_tarsal_claw, :]), axis=1)
    RF_coxa = np.linalg.norm((points3d[:,idx_RF_body_coxa,:] - points3d[:,idx_RF_coxa_femur,:]),axis=1)
    RF_femur = np.linalg.norm((points3d[:, idx_RF_coxa_femur, :] - points3d[:, idx_RF_femur_tibia, :]), axis=1)
    RF_tibia = np.linalg.norm((points3d[:, idx_RF_femur_tibia, :] - points3d[:, idx_RF_tibia_tarsus, :]), axis=1)
    RF_tarsus = np.linalg.norm((points3d[:, idx_RF_tibia_tarsus, :] - points3d[:, idx_RF_tarsal_claw, :]), axis=1)

    LM_coxa = np.linalg.norm((points3d[:,idx_LM_body_coxa,:] - points3d[:,idx_LM_coxa_femur,:]),axis=1)
    LM_femur = np.linalg.norm((points3d[:, idx_LM_coxa_femur, :] - points3d[:, idx_LM_femur_tibia, :]), axis=1)
    LM_tibia = np.linalg.norm((points3d[:, idx_LM_femur_tibia, :] - points3d[:, idx_LM_tibia_tarsus, :]), axis=1)
    LM_tarsus = np.linalg.norm((points3d[:, idx_LM_tibia_tarsus, :] - points3d[:, idx_LM_tarsal_claw, :]), axis=1)
    RM_coxa = np.linalg.norm((points3d[:,idx_RM_body_coxa,:] - points3d[:,idx_RM_coxa_femur,:]),axis=1)
    RM_femur = np.linalg.norm((points3d[:, idx_RM_coxa_femur, :] - points3d[:, idx_RM_femur_tibia, :]), axis=1)
    RM_tibia = np.linalg.norm((points3d[:, idx_RM_femur_tibia, :] - points3d[:, idx_RM_tibia_tarsus, :]), axis=1)
    RM_tarsus = np.linalg.norm((points3d[:, idx_RM_tibia_tarsus, :] - points3d[:, idx_RM_tarsal_claw, :]), axis=1)

    LH_coxa = np.linalg.norm((points3d[:,idx_LH_body_coxa,:] - points3d[:,idx_LH_coxa_femur,:]),axis=1)
    LH_femur = np.linalg.norm((points3d[:, idx_LH_coxa_femur, :] - points3d[:, idx_LH_femur_tibia, :]), axis=1)
    LH_tibia = np.linalg.norm((points3d[:, idx_LH_femur_tibia, :] - points3d[:, idx_LH_tibia_tarsus, :]), axis=1)
    LH_tarsus = np.linalg.norm((points3d[:, idx_LH_tibia_tarsus, :] - points3d[:, idx_LH_tarsal_claw, :]), axis=1)
    RH_coxa = np.linalg.norm((points3d[:,idx_RH_body_coxa,:] - points3d[:,idx_RH_coxa_femur,:]),axis=1)
    RH_femur = np.linalg.norm((points3d[:, idx_RH_coxa_femur, :] - points3d[:, idx_RH_femur_tibia, :]), axis=1)
    RH_tibia = np.linalg.norm((points3d[:, idx_RH_femur_tibia, :] - points3d[:, idx_RH_tibia_tarsus, :]), axis=1)
    RH_tarsus = np.linalg.norm((points3d[:, idx_RH_tibia_tarsus, :] - points3d[:, idx_RH_tarsal_claw, :]), axis=1)

    # limb_lengths
    LF_coxa_filtered = np.linalg.norm((points3d_filtered[:,idx_LF_body_coxa,:] - points3d_filtered[:,idx_LF_coxa_femur,:]),axis=1)
    LF_femur_filtered = np.linalg.norm((points3d_filtered[:, idx_LF_coxa_femur, :] - points3d_filtered[:, idx_LF_femur_tibia, :]), axis=1)
    LF_tibia_filtered = np.linalg.norm((points3d_filtered[:, idx_LF_femur_tibia, :] - points3d_filtered[:, idx_LF_tibia_tarsus, :]), axis=1)
    LF_tarsus_filtered = np.linalg.norm((points3d_filtered[:, idx_LF_tibia_tarsus, :] - points3d_filtered[:, idx_LF_tarsal_claw, :]), axis=1)
    RF_coxa_filtered = np.linalg.norm((points3d_filtered[:,idx_RF_body_coxa,:] - points3d_filtered[:,idx_RF_coxa_femur,:]),axis=1)
    RF_femur_filtered = np.linalg.norm((points3d_filtered[:, idx_RF_coxa_femur, :] - points3d_filtered[:, idx_RF_femur_tibia, :]), axis=1)
    RF_tibia_filtered = np.linalg.norm((points3d_filtered[:, idx_RF_femur_tibia, :] - points3d_filtered[:, idx_RF_tibia_tarsus, :]), axis=1)
    RF_tarsus_filtered = np.linalg.norm((points3d_filtered[:, idx_RF_tibia_tarsus, :] - points3d_filtered[:, idx_RF_tarsal_claw, :]), axis=1)

    LM_coxa_filtered = np.linalg.norm((points3d_filtered[:, idx_LM_body_coxa, :] - points3d_filtered[:, idx_LM_coxa_femur, :]), axis=1)
    LM_femur_filtered = np.linalg.norm((points3d_filtered[:, idx_LM_coxa_femur, :] - points3d_filtered[:, idx_LM_femur_tibia, :]),
                       axis=1)
    LM_tibia_filtered = np.linalg.norm((points3d_filtered[:, idx_LM_femur_tibia, :] - points3d_filtered[:, idx_LM_tibia_tarsus, :]),
                       axis=1)
    LM_tarsus_filtered = np.linalg.norm((points3d_filtered[:, idx_LM_tibia_tarsus, :] - points3d_filtered[:, idx_LM_tarsal_claw, :]),
                       axis=1)
    RM_coxa_filtered = np.linalg.norm((points3d_filtered[:, idx_RM_body_coxa, :] - points3d_filtered[:, idx_RM_coxa_femur, :]), axis=1)
    RM_femur_filtered = np.linalg.norm((points3d_filtered[:, idx_RM_coxa_femur, :] - points3d_filtered[:, idx_RM_femur_tibia, :]),
                       axis=1)
    RM_tibia_filtered = np.linalg.norm((points3d_filtered[:, idx_RM_femur_tibia, :] - points3d_filtered[:, idx_RM_tibia_tarsus, :]),
                       axis=1)
    RM_tarsus_filtered = np.linalg.norm((points3d_filtered[:, idx_RM_tibia_tarsus, :] - points3d_filtered[:, idx_RM_tarsal_claw, :]),
                       axis=1)

    LH_coxa_filtered = np.linalg.norm((points3d_filtered[:, idx_LH_body_coxa, :] - points3d_filtered[:, idx_LH_coxa_femur, :]), axis=1)
    LH_femur_filtered = np.linalg.norm((points3d_filtered[:, idx_LH_coxa_femur, :] - points3d_filtered[:, idx_LH_femur_tibia, :]),
                       axis=1)
    LH_tibia_filtered = np.linalg.norm((points3d_filtered[:, idx_LH_femur_tibia, :] - points3d_filtered[:, idx_LH_tibia_tarsus, :]),
                       axis=1)
    LH_tarsus_filtered = np.linalg.norm((points3d_filtered[:, idx_LH_tibia_tarsus, :] - points3d_filtered[:, idx_LH_tarsal_claw, :]),
                       axis=1)
    RH_coxa_filtered = np.linalg.norm((points3d_filtered[:, idx_RH_body_coxa, :] - points3d_filtered[:, idx_RH_coxa_femur, :]), axis=1)
    RH_femur_filtered = np.linalg.norm((points3d_filtered[:, idx_RH_coxa_femur, :] - points3d_filtered[:, idx_RH_femur_tibia, :]),
                       axis=1)
    RH_tibia_filtered = np.linalg.norm((points3d_filtered[:, idx_RH_femur_tibia, :] - points3d_filtered[:, idx_RH_tibia_tarsus, :]),
                       axis=1)
    RH_tarsus_filtered = np.linalg.norm((points3d_filtered[:, idx_RH_tibia_tarsus, :] - points3d_filtered[:, idx_RH_tarsal_claw, :]),
                       axis=1)

    # limb_lengths
    LF_coxa_filtered_2 = np.linalg.norm((points3d_filtered_2[:,idx_LF_body_coxa,:] - points3d_filtered_2[:,idx_LF_coxa_femur,:]),axis=1)
    LF_femur_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LF_coxa_femur, :] - points3d_filtered_2[:, idx_LF_femur_tibia, :]), axis=1)
    LF_tibia_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LF_femur_tibia, :] - points3d_filtered_2[:, idx_LF_tibia_tarsus, :]), axis=1)
    LF_tarsus_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LF_tibia_tarsus, :] - points3d_filtered_2[:, idx_LF_tarsal_claw, :]), axis=1)
    RF_coxa_filtered_2 = np.linalg.norm((points3d_filtered_2[:,idx_RF_body_coxa,:] - points3d_filtered_2[:,idx_RF_coxa_femur,:]),axis=1)
    RF_femur_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RF_coxa_femur, :] - points3d_filtered_2[:, idx_RF_femur_tibia, :]), axis=1)
    RF_tibia_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RF_femur_tibia, :] - points3d_filtered_2[:, idx_RF_tibia_tarsus, :]), axis=1)
    RF_tarsus_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RF_tibia_tarsus, :] - points3d_filtered_2[:, idx_RF_tarsal_claw, :]), axis=1)

    LM_coxa_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LM_body_coxa, :] - points3d_filtered_2[:, idx_LM_coxa_femur, :]), axis=1)
    LM_femur_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LM_coxa_femur, :] - points3d_filtered_2[:, idx_LM_femur_tibia, :]),
                       axis=1)
    LM_tibia_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LM_femur_tibia, :] - points3d_filtered_2[:, idx_LM_tibia_tarsus, :]),
                       axis=1)
    LM_tarsus_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LM_tibia_tarsus, :] - points3d_filtered_2[:, idx_LM_tarsal_claw, :]),
                       axis=1)
    RM_coxa_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RM_body_coxa, :] - points3d_filtered_2[:, idx_RM_coxa_femur, :]), axis=1)
    RM_femur_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RM_coxa_femur, :] - points3d_filtered_2[:, idx_RM_femur_tibia, :]),
                       axis=1)
    RM_tibia_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RM_femur_tibia, :] - points3d_filtered_2[:, idx_RM_tibia_tarsus, :]),
                       axis=1)
    RM_tarsus_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RM_tibia_tarsus, :] - points3d_filtered_2[:, idx_RM_tarsal_claw, :]),
                       axis=1)

    LH_coxa_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LH_body_coxa, :] - points3d_filtered_2[:, idx_LH_coxa_femur, :]), axis=1)
    LH_femur_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LH_coxa_femur, :] - points3d_filtered_2[:, idx_LH_femur_tibia, :]),
                       axis=1)
    LH_tibia_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LH_femur_tibia, :] - points3d_filtered_2[:, idx_LH_tibia_tarsus, :]),
                       axis=1)
    LH_tarsus_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_LH_tibia_tarsus, :] - points3d_filtered_2[:, idx_LH_tarsal_claw, :]),
                       axis=1)
    RH_coxa_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RH_body_coxa, :] - points3d_filtered_2[:, idx_RH_coxa_femur, :]), axis=1)
    RH_femur_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RH_coxa_femur, :] - points3d_filtered_2[:, idx_RH_femur_tibia, :]),
                       axis=1)
    RH_tibia_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RH_femur_tibia, :] - points3d_filtered_2[:, idx_RH_tibia_tarsus, :]),
                       axis=1)
    RH_tarsus_filtered_2 = np.linalg.norm((points3d_filtered_2[:, idx_RH_tibia_tarsus, :] - points3d_filtered_2[:, idx_RH_tarsal_claw, :]),
                       axis=1)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(4, 6)
    # axs.title('after AniPose MedianFilter')
    frame_idx = np.arange(LF_tarsus.shape[0])+1
    axs[0,0].plot(frame_idx, LF_coxa, LF_coxa_filtered)
    axs[0,0].set_xlabel('Frame')
    axs[0,0].set_ylabel('Limb Length')
    axs[0,0].grid(True)
    fig.legend(labels,loc='upper center')
    # axs[0,0].legend(labels)
    axs[0,0].set_title('LF_coxa')
    axs[0,0].set_ylim(0,200)
    axs[0,1].plot(frame_idx, RF_coxa, RF_coxa_filtered)
    # axs[0,1].set_xlabel('Frame')
    # axs[0,1].set_ylabel('Limb Length')
    axs[0,1].grid(True)
    # labels = ["Resnet-152", "Resnet-50"]
    # axs[0,1].legend(labels)
    axs[0,1].set_title('RF_coxa')
    axs[0,1].set_ylim(0,200)

    axs[1,0].plot(frame_idx, LF_femur, LF_femur_filtered)
    axs[1,0].grid(True)
    axs[1,0].set_title('LF_femur')
    axs[1,0].set_xlabel('Frame')
    axs[1,0].set_ylabel('Limb Length')
    axs[1,0].set_ylim(0,200)
    axs[1,1].plot(frame_idx, RF_femur, RF_femur_filtered)
    axs[1,1].grid(True)
    axs[1,1].set_title('RF_femur')
    axs[1,1].set_xlabel('Frame')
    axs[1,1].set_ylabel('Limb Length')
    axs[1,1].set_ylim(0,200)

    axs[2,0].plot(frame_idx, LF_tibia, LF_tibia_filtered)
    axs[2,0].grid(True)
    axs[2,0].set_title('LF_tibia')
    axs[2,0].set_xlabel('Frame')
    axs[2,0].set_ylabel('Limb Length')
    axs[2,0].set_ylim(0,200)
    axs[2,1].plot(frame_idx, RF_tibia, RF_tibia_filtered)
    axs[2,1].grid(True)
    axs[2,1].set_title('RF_tibia')
    axs[2,1].set_xlabel('Frame')
    axs[2,1].set_ylabel('Limb Length')
    axs[2,1].set_ylim(0,200)


    axs[3,0].plot(frame_idx, LF_tarsus, LF_tarsus_filtered)
    axs[3,0].grid(True)
    axs[3,0].set_title('LF_tarsus')
    axs[3,0].set_xlabel('Frame')
    axs[3,0].set_ylabel('Limb Length')
    axs[3,0].set_ylim(0,200)
    axs[3,1].plot(frame_idx, RF_tarsus, RF_tarsus_filtered)
    axs[3,1].grid(True)
    axs[3,1].set_title('RF_tarsus')
    axs[3,1].set_xlabel('Frame')
    axs[3,1].set_ylabel('Limb Length')
    axs[3,1].set_ylim(0,200)

    axs[0,2].plot(frame_idx, LM_coxa, LM_coxa_filtered)
    axs[0,2].set_xlabel('Frame')
    axs[0,2].set_ylabel('Limb Length')
    axs[0,2].grid(True)
    axs[0,2].set_title('LM_coxa')
    axs[0,2].set_ylim(0,200)
    axs[0,3].plot(frame_idx, RM_coxa, RM_coxa_filtered)
    axs[0,3].grid(True)
    axs[0,3].set_title('RM_coxa')
    axs[0,3].set_ylim(0,200)


    axs[1,2].plot(frame_idx, LM_femur, LM_femur_filtered)
    axs[1,2].grid(True)
    axs[1,2].set_title('LM_femur')
    axs[1,2].set_xlabel('Frame')
    axs[1,2].set_ylabel('Limb Length')
    axs[1,2].set_ylim(0,200)
    axs[1,3].plot(frame_idx, RM_femur, RM_femur_filtered)
    axs[1,3].grid(True)
    axs[1,3].set_title('RM_femur')
    axs[1,3].set_xlabel('Frame')
    axs[1,3].set_ylabel('Limb Length')
    axs[1,3].set_ylim(0,200)

    axs[2,2].plot(frame_idx, LM_tibia, LM_tibia_filtered)
    axs[2,2].grid(True)
    axs[2,2].set_title('LM_tibia')
    axs[2,2].set_xlabel('Frame')
    axs[2,2].set_ylabel('Limb Length')
    axs[2,2].set_ylim(0,200)
    axs[2,3].plot(frame_idx, RM_tibia, RM_tibia_filtered)
    axs[2,3].grid(True)
    axs[2,3].set_title('RM_tibia')
    axs[2,3].set_xlabel('Frame')
    axs[2,3].set_ylabel('Limb Length')
    axs[2,3].set_ylim(0,200)

    axs[3,2].plot(frame_idx, LM_tarsus, LM_tarsus_filtered)
    axs[3,2].grid(True)
    axs[3,2].set_title('LM_tarsus')
    axs[3,2].set_xlabel('Frame')
    axs[3,2].set_ylabel('Limb Length')
    axs[3,2].set_ylim(0,200)
    axs[3,3].plot(frame_idx, RM_tarsus, RM_tarsus_filtered)
    axs[3,3].grid(True)
    axs[3,3].set_title('RM_tarsus')
    axs[3,3].set_xlabel('Frame')
    axs[3,3].set_ylabel('Limb Length')
    axs[3,3].set_ylim(0,200)



    axs[0,4].plot(frame_idx, LH_coxa, LH_coxa_filtered)
    axs[0,4].grid(True)
    axs[0,4].set_title('LH_coxa')
    axs[0,4].set_xlabel('Frame')
    axs[0,4].set_ylabel('Limb Length')
    axs[0,4].set_ylim(0,200)
    axs[0,5].plot(frame_idx, RH_coxa, RH_coxa_filtered)
    axs[0,5].grid(True)
    axs[0,5].set_title('RH_coxa')
    axs[0,5].set_xlabel('Frame')
    axs[0,5].set_ylabel('Limb Length')
    axs[0,5].set_ylim(0,200)

    axs[1,4].plot(frame_idx, LH_femur, LH_femur_filtered)
    axs[1,4].grid(True)
    axs[1,4].set_title('LH_femur')
    axs[1,4].set_xlabel('Frame')
    axs[1,4].set_ylabel('Limb Length')
    axs[1,4].set_ylim(0,200)
    axs[1,5].plot(frame_idx, RH_femur, RH_femur_filtered)
    axs[1,5].grid(True)
    axs[1,5].set_title('RH_femur')
    axs[1,5].set_xlabel('Frame')
    axs[1,5].set_ylabel('Limb Length')
    axs[1,5].set_ylim(0,200)

    axs[2,4].plot(frame_idx, LH_tibia, LH_tibia_filtered)
    axs[2,4].grid(True)
    axs[2,4].set_title('LH_tibia')
    axs[2,4].set_xlabel('Frame')
    axs[2,4].set_ylabel('Limb Length')
    axs[2,4].set_ylim(0,200)
    axs[2,5].plot(frame_idx, RH_tibia, RH_tibia_filtered)
    axs[2,5].grid(True)
    axs[2,5].set_title('RH_tibia')
    axs[2,5].set_xlabel('Frame')
    axs[2,5].set_ylabel('Limb Length')
    axs[2,5].set_ylim(0,200)

    axs[3,4].plot(frame_idx, LH_tarsus, LH_tarsus_filtered)
    axs[3,4].grid(True)
    axs[3,4].set_title('LH_tarsus')
    axs[3,4].set_xlabel('Frame')
    axs[3,4].set_ylabel('Limb Length')
    axs[3,4].set_ylim(0,200)
    axs[3,5].plot(frame_idx, RH_tarsus, RH_tarsus_filtered)
    axs[3,5].grid(True)
    axs[3,5].set_title('RH_tarsus')
    axs[3,5].set_xlabel('Frame')
    axs[3,5].set_ylabel('Limb Length')
    axs[3,5].set_ylim(0,200)


    fig.tight_layout(pad=1.5,w_pad=-2,h_pad=-2)

    # plot std
    bar_labels = ['LF_coxa', 'LF_femur', 'LF_tibia', 'LF_tarsus',
                 'LM_coxa', 'LM_femur', 'LM_tibia', 'LM_tarsus',
                 'LH_coxa', 'LH_femur', 'LH_tibia', 'LH_tarsus',
                 'RF_coxa', 'RF_femur', 'RF_tibia', 'RF_tarsus',
                 'RM_coxa', 'RM_femur', 'RM_tibia', 'RM_tarsus',
                 'RH_coxa', 'RH_femur', 'RH_tibia', 'RH_tarsus'
                 ]
    means_filtered = [LF_coxa_filtered.mean(), LF_femur_filtered.mean(), LF_tibia_filtered.mean(), LF_tarsus_filtered.mean(),
                 LM_coxa_filtered.mean(), LM_femur_filtered.mean(), LM_tibia_filtered.mean(), LM_tarsus_filtered.mean(),
                 LH_coxa_filtered.mean(), LH_femur_filtered.mean(), LH_tibia_filtered.mean(), LH_tarsus_filtered.mean(),
                 RF_coxa_filtered.mean(), RF_femur_filtered.mean(), RF_tibia_filtered.mean(), RF_tarsus_filtered.mean(),
                 RM_coxa_filtered.mean(), RM_femur_filtered.mean(), RM_tibia_filtered.mean(), RM_tarsus_filtered.mean(),
                 RH_coxa_filtered.mean(), RH_femur_filtered.mean(), RH_tibia_filtered.mean(), RH_tarsus_filtered.mean()
                 ]

    means_filtered_2 = [LF_coxa_filtered_2.mean(), LF_femur_filtered_2.mean(), LF_tibia_filtered_2.mean(), LF_tarsus_filtered_2.mean(),
                 LM_coxa_filtered_2.mean(), LM_femur_filtered_2.mean(), LM_tibia_filtered_2.mean(), LM_tarsus_filtered_2.mean(),
                 LH_coxa_filtered_2.mean(), LH_femur_filtered_2.mean(), LH_tibia_filtered_2.mean(), LH_tarsus_filtered_2.mean(),
                 RF_coxa_filtered_2.mean(), RF_femur_filtered_2.mean(), RF_tibia_filtered_2.mean(), RF_tarsus_filtered_2.mean(),
                 RM_coxa_filtered_2.mean(), RM_femur_filtered_2.mean(), RM_tibia_filtered_2.mean(), RM_tarsus_filtered_2.mean(),
                 RH_coxa_filtered_2.mean(), RH_femur_filtered_2.mean(), RH_tibia_filtered_2.mean(), RH_tarsus_filtered_2.mean()
                 ]
    means = [LF_coxa.mean(), LF_femur.mean(), LF_tibia.mean(), LF_tarsus.mean(),
                 LM_coxa.mean(), LM_femur.mean(), LM_tibia.mean(), LM_tarsus.mean(),
                 LH_coxa.mean(), LH_femur.mean(), LH_tibia.mean(), LH_tarsus.mean(),
                 RF_coxa.mean(), RF_femur.mean(), RF_tibia.mean(), RF_tarsus.mean(),
                 RM_coxa.mean(), RM_femur.mean(), RM_tibia.mean(), RM_tarsus.mean(),
                 RH_coxa.mean(), RH_femur.mean(), RH_tibia.mean(), RH_tarsus.mean()
                 ]

    means_filtered_2_std = [LF_coxa_filtered_2.std(), LF_femur_filtered_2.std(), LF_tibia_filtered_2.std(), LF_tarsus_filtered_2.std(),
                 LM_coxa_filtered_2.std(), LM_femur_filtered_2.std(), LM_tibia_filtered_2.std(), LM_tarsus_filtered_2.std(),
                 LH_coxa_filtered_2.std(), LH_femur_filtered_2.std(), LH_tibia_filtered_2.std(), LH_tarsus_filtered_2.std(),
                 RF_coxa_filtered_2.std(), RF_femur_filtered_2.std(), RF_tibia_filtered_2.std(), RF_tarsus_filtered_2.std(),
                 RM_coxa_filtered_2.std(), RM_femur_filtered_2.std(), RM_tibia_filtered_2.std(), RM_tarsus_filtered_2.std(),
                 RH_coxa_filtered_2.std(), RH_femur_filtered_2.std(), RH_tibia_filtered_2.std(), RH_tarsus_filtered_2.std()
                 ]
    means_filtered_std = [LF_coxa_filtered.std(), LF_femur_filtered.std(), LF_tibia_filtered.std(), LF_tarsus_filtered.std(),
                 LM_coxa_filtered.std(), LM_femur_filtered.std(), LM_tibia_filtered.std(), LM_tarsus_filtered.std(),
                 LH_coxa_filtered.std(), LH_femur_filtered.std(), LH_tibia_filtered.std(), LH_tarsus_filtered.std(),
                 RF_coxa_filtered.std(), RF_femur_filtered.std(), RF_tibia_filtered.std(), RF_tarsus_filtered.std(),
                 RM_coxa_filtered.std(), RM_femur_filtered.std(), RM_tibia_filtered.std(), RM_tarsus_filtered.std(),
                 RH_coxa_filtered.std(), RH_femur_filtered.std(), RH_tibia_filtered.std(), RH_tarsus_filtered.std()
                 ]
    means_std = [LF_coxa.std(), LF_femur.std(), LF_tibia.std(), LF_tarsus.std(),
                 LM_coxa.std(), LM_femur.std(), LM_tibia.std(), LM_tarsus.std(),
                 LH_coxa.std(), LH_femur.std(), LH_tibia.std(), LH_tarsus.std(),
                 RF_coxa.std(), RF_femur.std(), RF_tibia.std(), RF_tarsus.std(),
                 RM_coxa.std(), RM_femur.std(), RM_tibia.std(), RM_tarsus.std(),
                 RH_coxa.std(), RH_femur.std(), RH_tibia.std(), RH_tarsus.std()
                 ]
    x = np.arange(len(bar_labels))  # the label locations
    width = 0.25  # the width of the bars
    fig_std, ax_std = plt.subplots()
    rects1 = ax_std.bar(x - width, means, width, yerr=means_std , label=labels[0])
    rects2 = ax_std.bar(x, means_filtered, width, yerr=means_filtered_std , label=labels[1])
    rects3 = ax_std.bar(x + width, means_filtered_2, width, yerr=means_filtered_2_std , label=labels[2])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax_std.set_ylabel('Mean Limb length [px]')
    ax_std.set_title('Mean Limb length for Video fly_4_FW_1')
    ax_std.set_xticks(x)
    plt.setp(ax_std.get_xticklabels(), rotation=90)
    ax_std.set_xticklabels(bar_labels)
    ax_std.legend(loc='upper left')
    fig_std.tight_layout()

    width = .5
    x_std_summary = np.arange(3)
    fig_std_summary, ax_std_summary = plt.subplots()
    ax_std_summary.bar(x_std_summary , [np.mean(means_std), np.mean(means_filtered_std), np.mean(means_filtered_2_std)] , width , label=labels[0])
    # ax_std_summary.bar(x_std_summary , np.mean(means_filtered_std), width , label=labels[1])
    # ax_std_summary.bar(x_std_summary + width, np.mean(means_filtered_2_std), width, label=labels[2])
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax_std_summary.set_ylabel('Mean standard deviation [px]')
    ax_std_summary.set_title('Mean standard deviation limb length for video fly_4_FW_1')
    ax_std_summary.set_xticks(x_std_summary)
    plt.setp(ax_std_summary.get_xticklabels(), rotation=90)
    ax_std_summary.set_xticklabels(labels)
    # ax_std_summary.legend(loc='upper right')
    fig_std_summary.tight_layout()


    plt.show()


if __name__ == "__main__":
    import traceback

    # fly_number=range(1,6+1,1)
    # behaviour=['AG','FW','PG','PE']
    # video_sequence_number=range(1,20+1,1)

    AniPose_filter_enable = True
    fly_number= [1]
    behaviour=['FW']
    video_sequence_number= [2]

    for _fly_number in zip(fly_number):
        for _behaviour in zip(behaviour):
            for _video_sequence_number in zip(video_sequence_number):
                try:
                    main(str(_fly_number[0]), _behaviour[0], str(_video_sequence_number[0]), AniPose_filter_enable)
                except:
                    traceback.print_exc()
                    continue