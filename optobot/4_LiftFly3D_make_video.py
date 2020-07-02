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
import matplotlib
matplotlib.use('Agg')
from skeleton import skeleton
from matplotlib.legend_handler import HandlerTuple
import pickle
from tqdm import tqdm
import src.utils as utils


def plot_3d_graph(pos, ax, color_edge = 'k', style = '-'):
    
    pos = np.array(pos)
    
    for i, j in enumerate(G.edges()): 
        xid, yid = j[0], j[1]
        if xid is not None and yid is not None:
            x = np.array((pos[xid][0], pos[yid][0]))
            y = np.array((pos[xid][1], pos[yid][1]))
            z = np.array((pos[xid][2], pos[yid][2]))
                   
            ax.plot(x, y, -z, style, c=color_edge[j[0]], alpha=1.0, linewidth = 2) 
    
    return ax


G, color_edge = skeleton() #skeleton of fly

print('making video')

data_dir = '/data/LiftFly3D/optobot/102906_s1a5_p6-0' #data directory
template_dir = '/data/LiftFly3D/optobot/network/'
        
#predictions
data = torch.load(data_dir + '/test_results.pth.tar')
out_offset, inp_offset = pickle.load(open('joint_locations.pkl','rb'))

#output
targets_1d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_1d']
out_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
out_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
out = utils.unNormalizeData(data['output'], out_mean[targets_1d], out_std[targets_1d])

#inputs
targets_2d = torch.load(template_dir + '/stat_2d.pth.tar')['targets_2d']    
inp_mean = torch.load(template_dir + 'stat_2d.pth.tar')['mean']
inp_std = torch.load(template_dir + 'stat_2d.pth.tar')['std']
inp = utils.unNormalizeData(data['input'], inp_mean[targets_2d], inp_std[targets_2d])

targets_1d = torch.load(template_dir + '/stat_3d.pth.tar')['targets_1d'] 
targets_2d = torch.load(template_dir + '/stat_2d.pth.tar')['targets_2d'] 
out = utils.expand(out,targets_1d,len(out_mean))
inp = utils.expand(inp,targets_2d,len(inp_mean))

out += out_offset
inp += inp_offset

# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev = 40, azim=140)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=15, metadata=metadata)
xlim, ylim, zlim = None,None,None

with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(799)):
        pos_pred = []
        
        ax.cla()
        
        for j in range(out.shape[1]):
            pos_pred.append((inp[t, 2*j], inp[t, 2*j+1], out[t, j]))
           
        ax = plot_3d_graph(pos_pred, ax, color_edge = color_edge, style='--')       
        
        if xlim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
              
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p3, = ax.plot(pts, pts, pts, 'r--', dashes=(2, 2))
        p4, = ax.plot(pts, pts, pts, 'b--', dashes=(2, 2))
        ax.legend([(p3, p4)], 
            ['LiftFly3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.9))    
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
        
#        ax.set_axis_off()
#        plt.savefig('test.svg')
#        import sys
#        sys.exit()
    
        writer.grab_frame()