#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:32:53 2020

@author: adamgosztolai
"""
import os

# =============================================================================
# Crop images
# =============================================================================
#folder = ['/mnt/NAS/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/002_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/003_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/004_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/001_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/002_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/003_prism/behData/images/',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/004_prism/behData/images/']
#
#os.chdir('prism')
#
#for dir_ in folder:
#    print(dir_)
#    os.system('python prism_data_preprocess.py ' + dir_)

# =============================================================================
# Create videos
# =============================================================================
#folder = ['/mnt/NAS/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/side_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/001_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/002_prism/behData/images/side_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/002_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/003_prism/behData/images/side_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/003_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/004_prism/behData/images/side_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly1/004_prism/behData/images/bottom_view_prism_data_191125_PR_Fly1',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/001_prism/behData/images/side_view_prism_data_191125_PR_Fly2',
#          '/mnt/NAS/SG/prism_data/191125_PR/Fly2/001_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/002_prism/behData/images/side_view_prism_data_191125_PR_Fly2',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/002_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/003_prism/behData/images/side_view_prism_data_191125_PR_Fly2',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/003_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/004_prism/behData/images/side_view_prism_data_191125_PR_Fly2',
#         '/mnt/NAS/SG/prism_data/191125_PR/Fly2/004_prism/behData/images/bottom_view_prism_data_191125_PR_Fly2']
#
#
#for dir_ in folder:
#    os.chdir(dir_)
#    where = dir_.split('/')[-1].split('_')[0]
#    
#    exp = dir_.split('/')[5]
#    fly = dir_.split('/')[6]
#    run = dir_.split('/')[7]
#    print('/data/LiftFly3D/prism/' + where + '_view/video_' + exp + '_' + fly + '_' + run + '.mp4')
#    os.system('ffmpeg -pattern_type glob -i "*.jpg" -pix_fmt yuv420p /data/LiftFly3D/prism/' + where + '_view/videos/video_' + exp + '_' + fly + '_' + run + '.mp4')