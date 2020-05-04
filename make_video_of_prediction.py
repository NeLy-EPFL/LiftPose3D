#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:53:14 2020

@author: adamgosztolai
"""

import torch
import networkx as nx
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FFMpegWriter
import matplotlib
matplotlib.use('Agg')
from src.normalize import unNormalizeData, get_coords_in_dim

def plot_3d_graph(pos, ax, color = 'k'):
    
    pos = np.array(pos)    
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=color, s=10, edgecolors='k', alpha=0.7)
           
    for _, j in enumerate(G.edges()): 
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
        ax.plot(x, y, z, c=color, alpha=0.3, linewidth = 1)
        
    ax.view_init(elev = 60, azim=100)

    ax.set_axis_off()
    
    return ax

'''
Joints
------
0:  BODY_COXA,    :19 
1:  COXA_FEMUR,   :20 
2:  FEMUR_TIBIA,  :21
3:  TIBIA_TARSUS, :22
4:  TARSUS_TIP,   :23

5:  BODY_COXA,    :24
6:  COXA_FEMUR,   :25
7:  FEMUR_TIBIA,  :26
8:  TIBIA_TARSUS, :27
9:  TARSUS_TIP,   :28
    
10: BODY_COXA,    :29
11: COXA_FEMUR,   :30
12: FEMUR_TIBIA,  :31
13: TIBIA_TARSUS, :32
14: TARSUS_TIP,   :33

15: ANTENNA,      :34
16: STRIPE,       :35
17: STRIPE,       :36
18: STRIPE,       :37
'''

n = 38
#nodes = list(range(38))
edges = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),(10,11),(11,12),(12,13),(13,14),(16,17),(17,18),(19,20),
         (20,21),(21,22),(22,23),(24,25),(25,26),(26,27),(27,28),(29,30),(30,31),(31,32),(32,33),(35,36),(36,37)]

edges = [e for e in edges if (e[0]<15 and e[1]<15) or (e[0]>18 and e[1]>18)]
nodes = list(range(29))

anchors = [0, 5, 10, 19, 24, 29]
target_sets = [[ 1,  2,  3,  4], [ 6,  7,  8,  9], [11, 12, 13, 14],
               [20, 21, 22, 23], [25, 26, 27, 28], [30, 31, 32, 33]]

#build graph
G=nx.Graph()
G.add_edges_from(edges)
G.add_nodes_from(nodes)

cameras = [1,5]

#load data / statistics for cameras
out, tar = [], []
for cam in cameras:
    ind = 'cam_' + str(cam)
    data_dir = '/data/LiftFly3D/DF3D/cam_' + str(cam)
    tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
    tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
    targets = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']
    offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
    offset = np.vstack( offset.values() )

    #load predictions
    data = torch.load(data_dir + '/test_results.pth.tar')
 
#    inp_mean[ind] = torch.load(data_dir + 'stat_2d.pth.tar')['mean']
#    inp_std[ind] = torch.load(data_dir + 'stat_2d.pth.tar')['std']
#    targets_2d = get_coords_in_dim(targets, 2)
#    xy = unNormalizeData(data['input'], inp_mean, inp_std, targets_2d)
    
    targets_3d = get_coords_in_dim(targets, 3)
    out_ = unNormalizeData(data['output'], tar_mean, tar_std, targets_3d)
    tar_ = unNormalizeData(data['target'], tar_mean, tar_std, targets_3d)
        
    #remove offset
    out_ += offset
    tar_ += offset
    
    out.append(out_)
    tar.append(tar_)   
    
#average over cameras
out = np.stack( out, axis=2 ) 
tar = np.stack( tar, axis=2 )
print(tar.shape)
out[out == 0] = np.nan
tar[tar == 0] = np.nan
out = np.nanmean(out, axis=2)
tar = np.nanmean(tar, axis=2)

#put back the anchor points
#output_full = np.zeros((output.shape[0], 15))
#target_full = np.zeros((target.shape[0], 15))
#input_full = np.zeros((inp.shape[0], 2*15))
#
#for i, anch in enumerate(anchors):
#    for j in target_sets[i]:            
#        output_full[:, j] = output[:, j-i-1]
#        target_full[:, j] = target[:, j-i-1]
#        input_full[:, 2*j:2*(j+1)] = inp[:, 2*(j-i-1):2*(j-i)]

#plot
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection = '3d')

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=10, metadata=metadata)
with writer.saving(fig, "LiftFly3d_prediction.mp4", 100):
    for t in range(1000):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
        
        for j in range(int(out.shape[1]/3)):
            pos_pred.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_tar.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
            
        ax = plot_3d_graph(pos_pred, ax, color = 'r')
        ax = plot_3d_graph(pos_tar, ax, color = 'b')
        
        writer.grab_frame()