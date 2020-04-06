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

#load data statistics
data_dir = '/data/DF3D/'
xy_mean = torch.load(data_dir +'stat_2d.pth.tar')['mean']
z_mean = torch.load(data_dir +'stat_z.pth.tar')['mean']
xy_std = torch.load(data_dir +'stat_2d.pth.tar')['std']
z_std = torch.load(data_dir +'stat_z.pth.tar')['std']
targets = torch.load(data_dir +'stat_z.pth.tar')['target_sets']
anchors = torch.load(data_dir +'stat_z.pth.tar')['anchors']

#load predictions
data = torch.load('checkpoint/LiftFly3D/MDN_CsCh_test.pth.tar')
z_out = data['output']
z_tar = data['target']
xy = data['input']

targets_xy = get_coords_in_dim(targets, 2)
targets_z = get_coords_in_dim(targets, 1)

xy = unNormalizeData(xy, xy_mean, xy_std, targets_xy)
z_out = unNormalizeData(z_out, z_mean, z_std, targets_z)
z_tar = unNormalizeData(z_tar, z_mean, z_std, targets_z)

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
        
        for j in range(int(z_out.shape[1])):
            pos_pred.append((xy[t, 2*j], xy[t, 2*j+1], z_out[t, j]))
            pos_tar.append((xy[t, 2*j], xy[t, 2*j+1], z_tar[t, j]))
            
        ax = plot_3d_graph(pos_pred, ax, color = 'r')
        ax = plot_3d_graph(pos_tar, ax, color = 'b')
        
        writer.grab_frame()