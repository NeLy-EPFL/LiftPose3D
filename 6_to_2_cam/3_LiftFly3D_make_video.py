#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:53:14 2020

@author: adamgosztolai
"""

import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
from matplotlib.legend_handler import HandlerTuple
from matplotlib.animation import FFMpegWriter
matplotlib.use('Agg')
import src.utils as utils
from skeleton import skeleton
import pickle
from tqdm import tqdm

def plot_3d_graph(pos, ax, color_edge, style = '-'):
    
    pos = np.array(pos)
           
    for i, j in enumerate(G.edges()): 
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
        ax.plot(x, -z, -y, style, c=color_edge[i], alpha=1.0, linewidth = 2)
    
    return ax


def average_cameras(tar):
    #average over cameras
    
    tar = np.stack( tar, axis=2 )
    tar[tar == 0] = np.nan
    tar_avg = np.nanmean(tar, axis=2)  
    tar_avg[np.isnan(tar_avg)] = 0
    
    return tar_avg


#cameras = [0,4]
cameras = [1,5]
#cameras = [2,6] #keep order, they come in L-R pairs!

root_dir = '/data/LiftFly3D/DF3D/cam_angles/cam'

#import
G, color_edge = skeleton() #skeleton
cam_par = pickle.load(open('cameras.pkl', "rb")) #camera parameters
offset = pickle.load(open('joint_locations.pkl', "rb")) #absolute joint locations

print('making video using cameras' + str(cameras))

#load data / statistics for cameras
out, tar = [], []
for cam in cameras:
    data_dir = root_dir + str(cam)
    
    #load stats
    tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
    tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
    targets_3d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_3d']
    
    #load predictions
    data = torch.load(data_dir + '/test_results.pth.tar')
    tar_ = data['target']
    out_ = data['output']
    
    #transform back to world
    tar_ = utils.camera_to_world(tar_,cam_par,cam)
    out_ = utils.camera_to_world(out_,cam_par,cam)
    
    #unnormalise
    tar_ = utils.unNormalizeData(tar_, tar_mean[targets_3d], tar_std[targets_3d]) 
    out_ = utils.unNormalizeData(out_, tar_mean[targets_3d], tar_std[targets_3d]) 
    tar_ = utils.expand(tar_,targets_3d,len(tar_mean))
    out_ = utils.expand(out_,targets_3d,len(tar_mean))

    out.append(out_)
    tar.append(tar_)
    
#combine cameras
out = average_cameras(out)
tar = average_cameras(tar)

#translate legs back to their original places
tar += offset
out += offset

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=30, azim=10)

writer = FFMpegWriter(fps=25)
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(900)):
        pos_pred, pos_tar = [], []
        
        ax.cla()
        
        for j in range(tar.shape[1]//3):
            pos_tar.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_pred.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
            
        ax = plot_3d_graph(pos_tar, ax, color_edge = color_edge)
        ax = plot_3d_graph(pos_pred, ax, color_edge = color_edge, style='--')
            
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p1, = ax.plot(pts, pts, pts, 'r-')
        p2, = ax.plot(pts, pts, pts, 'b-')
        p3, = ax.plot(pts, pts, pts, 'r--', dashes=(2, 2))
        p4, = ax.plot(pts, pts, pts, 'b--', dashes=(2, 2))
        ax.legend([(p1, p2), (p3, p4)], 
            ['Triangulated 3D pose (x0.25 real time)', 'LiftFly3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.9),
            frameon=False)    
        p1.remove()
        p2.remove()
        p3.remove()
        p4.remove()
        ####    
            
        ax.set_xlim([-2.5,2.5])
        ax.set_ylim([-2.5,2.5])
        ax.set_zlim([-1,0.75])
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True)
        
        writer.grab_frame()