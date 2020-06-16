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
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.animation import FFMpegWriter
matplotlib.use('Agg')
import src.utils as utils
from skeleton import skeleton
import pickle

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
joint_locations = pickle.load(open('joint_locations.pkl', "rb")) #absolute joint locations

print('making video using cameras' + str(cameras))

#load data / statistics for cameras
out, tar = [], []
data = {}
offset = None
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
    tar_ = utils.unNormalizeData(tar_, tar_mean, tar_std, targets_3d) 
    out_ = utils.unNormalizeData(out_, tar_mean, tar_std, targets_3d) 

    out.append(out_)
    tar.append(tar_)
    
#combine cameras
out = average_cameras(out)
tar = average_cameras(tar)

tar += joint_locations
out += joint_locations

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev = 20, azim=20)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=25, metadata=metadata)
xlim, ylim, zlim = None,None,None
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(900):
        pos_pred, pos_tar = [], []
        
        ax.cla()
        
        for j in range(tar.shape[1]//3):
            pos_tar.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_pred.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
            
        ax = plot_3d_graph(pos_tar, ax, color_edge = color_edge)
        ax = plot_3d_graph(pos_pred, ax, color_edge = color_edge, style='--')
        
        if xlim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
            
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p1, = ax.plot(pts, pts, pts, 'r-')
        p2, = ax.plot(pts, pts, pts, 'b-')
        p3, = ax.plot(pts, pts, pts, 'r--', dashes=(2, 2))
        p4, = ax.plot(pts, pts, pts, 'b--', dashes=(2, 2))
        ax.legend([(p1, p2), (p3, p4)], 
            ['Triangulated 3D pose', 'LiftFly3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0,0.9))    
        p1.remove()
        p2.remove()
        p3.remove()
        p4.remove()
        ####    
            
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True)
        
#        plt.savefig('test.png')
#        
#        import sys
#        sys.exit()
        
        writer.grab_frame()