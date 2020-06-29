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


def plot_3d_graph(pos, ax, color_edge = 'k', style = '-', LR=None):
    
    pos = np.array(pos)
           
    n = pos.shape[0]

    for i, j in enumerate(G.edges()): 
        if LR is not None:
            if LR[-1] & (j[0] < int(n/2)) & (j[1] < int(n/2)):   
                
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))
                
                ax.plot(x, y, -z, style, c=color_edge[j[0]], alpha=1.0, linewidth = 2)
                
            elif LR[0] & (j[0] >= int(n/2)) & (j[1] >= int(n/2)):
                
                x = np.array((pos[j[0]][0], pos[j[1]][0]))
                y = np.array((pos[j[0]][1], pos[j[1]][1]))
                z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
                ax.plot(x, y, -z, style, c=color_edge[j[0]], alpha=1.0, linewidth = 2)
                
        if LR is None:
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
            ax.plot(x, y, -z, style, c=color_edge[j[0]], alpha=1.0, linewidth = 2) 
    
    return ax



print('making video')

#load
G, color_edge = skeleton() 
data_dir = '/data/LiftFly3D/prism/data_oriented/test_data/'
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

#import pickle 
#pickle.dump([tar_offset,inp_offset], open('joint_locations.pkl','wb'))

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev = 40, azim=140)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=25, metadata=metadata)
xlim, ylim, zlim = None,None,None
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(1100)):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
        
        for j in range(out.shape[1]):
            pos_pred.append((inp[t, 2*j], inp[t, 2*j+1], out[t, j]))
            pos_tar.append((inp[t, 2*j], inp[t, 2*j+1], tar[t, j])) 
           
        ax = plot_3d_graph(pos_tar, ax, color_edge = color_edge, LR=data['bool_LR'][t,:])
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