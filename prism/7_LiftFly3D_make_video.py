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
from matplotlib.animation import FFMpegWriter
from matplotlib.legend_handler import HandlerTuple
import matplotlib
matplotlib.use('Agg')
from skeleton import skeleton
from tqdm import tqdm
import src.utils as utils

print('making video')

#specify folder
data_dir = '/data/LiftFly3D/prism/data_oriented/test_data/'

#load
G, color_edge = skeleton()
legtips = [4, 9, 14, 19, 24, 29]
data = torch.load(data_dir + '/test_results.pth.tar')

tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
targets_1d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_1d']
tar_offset = np.vstack(torch.load(data_dir + '/stat_3d.pth.tar')['offset'].values())[0,:]

inp_mean = torch.load(data_dir + '/stat_2d.pth.tar')['mean']
inp_std = torch.load(data_dir + '/stat_2d.pth.tar')['std']
targets_2d = torch.load(data_dir + '/stat_2d.pth.tar')['targets_2d']
inp_offset = np.vstack(torch.load(data_dir + '/stat_2d.pth.tar')['offset'].values())[0,:]

#unnormalize
tar = utils.unNormalizeData(data['target'], tar_mean[targets_1d], tar_std[targets_1d])
tar = utils.expand(tar,targets_1d,len(tar_mean))
tar += tar_offset
out = utils.unNormalizeData(data['output'], tar_mean[targets_1d], tar_std[targets_1d])
out = utils.expand(out,targets_1d,len(tar_mean))
out += tar_offset
inp = utils.unNormalizeData(data['input'], inp_mean[targets_2d], inp_std[targets_2d])
inp = utils.expand(inp,targets_2d,len(inp_mean))
inp += inp_offset

good_keypts = utils.expand(data['good_keypts'],targets_1d,len(tar_mean))

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40, azim=140)

writer = FFMpegWriter(fps=10)
xlim, ylim, zlim = None,None,None
l1, l2, trail = {}, {}, {}
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(600,1000)):
        
        thist = 5
        pos_pred, pos_tar = [], []
        for j in range(out.shape[1]):
            tmin = max(0,t-thist)
            pos_pred.append((inp[tmin:t, 2*j], inp[tmin:t, 2*j+1], -out[tmin:t, j]))
            pos_tar.append((inp[tmin:t, 2*j], inp[tmin:t, 2*j+1], -tar[tmin:t, j]))
                
        pos_pred, pos_tar = np.array(pos_pred), np.array(pos_tar)
                    
        l1 = utils.plot_3d_graph(G, pos_tar[:,:,-1], ax, l1, color_edge=color_edge, good_keypts=good_keypts[t,:])    
        l2 = utils.plot_3d_graph(G, pos_pred[:,:,-1], ax, l2, color_edge=color_edge, style='--') 
            
        for leg in legtips:
            if leg not in trail.keys():
                trail[leg], = ax.plot(pos_pred[leg,0,:], pos_pred[leg,1,:], pos_pred[leg,2,:], c='y', alpha=0.9)
            else:
                trail[leg].set_xdata(pos_pred[leg,0,:])
                trail[leg].set_ydata(pos_pred[leg,1,:])
                trail[leg].set_3d_properties(pos_pred[leg,2,:], zdir='z')

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
            ['Triangulated 3D pose (x0.2 real time)', 'LiftFly3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.9),
            frameon=False)    
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
    
        writer.grab_frame()