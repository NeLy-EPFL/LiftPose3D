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
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import matplotlib
matplotlib.use('Agg')
from skeleton import skeleton


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


def get_coords_in_dim(targets, dim):
    
    if len(targets)>1:
      dim_to_use = []
      for i in targets:
          dim_to_use += i
    else:
      dim_to_use = targets
  
    dim_to_use = np.array(dim_to_use)
    if dim == 2:    
      dim_to_use = np.sort( np.hstack( (dim_to_use*2, 
                                        dim_to_use*2+1)))
  
    return dim_to_use


def unNormalizeData(data, data_mean, data_std):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing
  """
  data *= data_std
  data += data_mean
    
  return data


def expand(data,dim_to_use,dim):
    
    T = data.shape[0]
    D = dim
    orig_data = np.zeros((T, D), dtype=np.float32)
    orig_data[:,dim_to_use] = data
    
    return orig_data


cameras = [0]

#import skeleton of fly
G, color_edge = skeleton()

print('making video using cameras' + str(cameras))

#load stats
data_dir = '/data/LiftFly3D/prism/data_oriented'
        
#load predictions
data = torch.load(data_dir + '/test_results.pth.tar')

#target
target_sets = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']    
tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
tar_offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
tar_offset = np.vstack( tar_offset.values() ) 
targets_1d = get_coords_in_dim(target_sets, 1)

tar = unNormalizeData(data['target'], tar_mean[targets_1d], tar_std[targets_1d])
tar = expand(tar,targets_1d,len(tar_mean))
tar += tar_offset[0,:]

#output
out = unNormalizeData(data['output'], tar_mean[targets_1d], tar_std[targets_1d])
out = expand(out,targets_1d,len(tar_mean))
out += tar_offset[0,:] 

bool_LR = data['bool_LR']

#inputs
inp_mean = torch.load(data_dir + '/stat_2d.pth.tar')['mean']
inp_std = torch.load(data_dir + '/stat_2d.pth.tar')['std']
inp_offset = torch.load(data_dir + '/stat_2d.pth.tar')['offset']

inp_offset = np.vstack( inp_offset.values() )
targets_2d = get_coords_in_dim(target_sets, 2)
inp = unNormalizeData(data['input'], inp_mean[targets_2d], inp_std[targets_2d])
inp = expand(inp,targets_2d,len(inp_mean))
inp += inp_offset[0,:] 

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev = 40, azim=140)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=25, metadata=metadata)
xlim, ylim, zlim = None,None,None
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(1000):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
        
        for j in range(int(out.shape[1])):
            pos_pred.append((inp[t, 2*j], inp[t, 2*j+1], out[t, j]))
            pos_tar.append((inp[t, 2*j], inp[t, 2*j+1], tar[t, j])) 
           
        ax = plot_3d_graph(pos_tar, ax, color_edge = color_edge, LR=bool_LR[t,:])
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
    
        writer.grab_frame()