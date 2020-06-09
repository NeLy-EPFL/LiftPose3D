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


G, color_edge = skeleton() #skeleton of fly

print('making video')

data_dir = '/data/LiftFly3D/optobot' #data directory
        
#predictions
data = torch.load(data_dir + '/test_results.pth.tar')

#output
coords_1 = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']    
out_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
out_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
out_offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
out_offset = np.vstack( out_offset.values() ) 
out_coords = get_coords_in_dim(coords_1, 1)
out = unNormalizeData(data['output'], out_mean[out_coords], out_std[out_coords])

#inputs
coords_2 = torch.load(data_dir + '/stat_2d.pth.tar')['target_sets']    
inp_mean = torch.load(data_dir + '/stat_2d.pth.tar')['mean']
inp_std = torch.load(data_dir + '/stat_2d.pth.tar')['std']
inp_offset = torch.load(data_dir + '/stat_2d.pth.tar')['offset']
inp_offset = np.vstack( inp_offset.values() )
inp_coords = get_coords_in_dim(coords_2, 2)
inp = unNormalizeData(data['input'], inp_mean[inp_coords], inp_std[inp_coords])

out_coords = get_coords_in_dim(coords_2, 1) 
inp_coords = get_coords_in_dim(coords_2, 2)

out = expand(out,out_coords,int(len(inp_mean)/2))
inp = expand(inp,inp_coords,len(inp_mean))
#out += out_offset[0,:]
#inp += inp_offset[0,:] 

# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev = 40, azim=140)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=10, metadata=metadata)
xlim, ylim, zlim = None,None,None

with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(80):
        pos_pred = []
        
        ax.cla()
        
        for j in range(int(out.shape[1])):
            pos_pred.append((inp[t, 2*j], inp[t, 2*j+1], out[t, j]))
           
        ax = plot_3d_graph(pos_pred, ax, color_edge = color_edge, style='--')       
        
        if xlim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True)
    
        writer.grab_frame()