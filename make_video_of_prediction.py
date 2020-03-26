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

def plot_fly(graph, pos_pred, pos_tar, ax=None):
    
    ax.cla()
    
    pos_pred = np.array(pos_pred)    
    ax.scatter(-pos_pred[:,0], -pos_pred[:,1], -pos_pred[:,2], c='b', s=10, edgecolors='k', alpha=0.7)
           
    for _, j in enumerate(G.edges()): 
        x = np.array((-pos_pred[j[0]][0], -pos_pred[j[1]][0]))
        y = np.array((-pos_pred[j[0]][1], -pos_pred[j[1]][1]))
        z = np.array((-pos_pred[j[0]][2], -pos_pred[j[1]][2]))
                   
        ax.plot(x, y, z, c='b', alpha=0.3, linewidth = 1)
        
    pos_tar = np.array(pos_tar)    
    ax.scatter(-pos_tar[:,0], -pos_tar[:,1], -pos_tar[:,2], c='r', s=10, edgecolors='k', alpha=0.7)
           
    for _, j in enumerate(G.edges()): 
        x = np.array((-pos_tar[j[0]][0], -pos_tar[j[1]][0]))
        y = np.array((-pos_tar[j[0]][1], -pos_tar[j[1]][1]))
        z = np.array((-pos_tar[j[0]][2], -pos_tar[j[1]][2]))
                   
        ax.plot(x, y, z, c='r', alpha=0.3, linewidth = 1)    

    ax.view_init(elev = 10, azim=290)

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

edges = [e for e in edges if e[0]<15 and e[1]<15]
nodes = list(range(15))

anchors = [0, 5, 10]
target_sets = [[1, 2, 3, 4], [6, 7, 8, 9], [11, 12, 13, 14]]

#build graph
G=nx.Graph()
G.add_edges_from(edges)
G.add_nodes_from(nodes)

#load data and prediction
data = torch.load('checkpoint/test/MDN_CsCh_test.pth.tar')
output = data['output']
target = data['target']

#put back the anchor points
output_full = np.zeros((output.shape[0], 3*15))
target_full = np.zeros((target.shape[0], 3*15))
for i, anch in enumerate(anchors):
    for j in target_sets[i]:            
        output_full[:, 3*j:3*j+3] = output[:, 3*(j-i-1):3*(j-i-1)+3]
        target_full[:, 3*j:3*j+3] = target[:, 3*(j-i-1):3*(j-i-1)+3]

#plot
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection = '3d')

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=10, metadata=metadata)
with writer.saving(fig, "LiftFly3d_prediction.mp4", 100):
    for t in range(100):
        pos_pred = []
        pos_tar = []
        for j in range(int(output_full.shape[1]/3)):
            pos_pred.append((output_full[t, 3*j], output_full[t, 3*j+1], output_full[t, 3*j+2]))
            pos_tar.append((target_full[t, 3*j], target_full[t, 3*j+1], target_full[t, 3*j+2]))
            
        ax = plot_fly(G, pos_pred, pos_tar, ax=ax)
        writer.grab_frame()