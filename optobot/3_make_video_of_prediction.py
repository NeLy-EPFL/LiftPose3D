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
        if (j[0] < pos.shape[0]) & (j[1] < pos.shape[0]):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
            ax.plot(x, y, -z, style, c=color_edge[i], alpha=1.0, linewidth = 2)
        
    ax.view_init(elev = 40, azim=140)
    
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


def unNormalizeData(data, data_mean, data_std, dim_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing
  """
  data *= data_std[dim_to_use]
  data += data_mean[dim_to_use]
  
  
  T = data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:,dim_to_use] = data
    
  return orig_data


cameras = [0]

#import skeleton of fly
G, color_edge = skeleton()

print('making video using cameras' + str(cameras))

#load stats
data_dir = '/data/LiftFly3D/prism/cam_0'
        
#load predictions
data = torch.load(data_dir + '/test_results.pth.tar')
_, bool_LR = torch.load(data_dir + '/test_3d.pth.tar')
bool_LR = np.hstack( bool_LR.values())
    
#target
target_sets = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']    
tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
tar_offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
tar_offset = np.vstack( tar_offset.values() ) 
targets_1d = get_coords_in_dim(target_sets, 1)

tar = unNormalizeData(data['target'], tar_mean, tar_std, targets_1d)
tar += tar_offset

#output
out = unNormalizeData(data['output'], tar_mean, tar_std, targets_1d)
out += tar_offset  
    
#inputs
inp_mean = torch.load(data_dir + '/stat_2d.pth.tar')['mean']
inp_std = torch.load(data_dir + '/stat_2d.pth.tar')['std']
inp_offset = torch.load(data_dir + '/stat_2d.pth.tar')['offset']
inp_offset = np.vstack( inp_offset.values() )
targets_2d = get_coords_in_dim(target_sets, 2)
inp = unNormalizeData(data['input'], inp_mean, inp_std, targets_2d)
inp += inp_offset  

#plot
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection = '3d')

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=10, metadata=metadata)
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(400):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
    
        for j in range(int(out.shape[1])):
            pos_pred.append((inp[t, 2*j], inp[t, 2*j+1], out[t, j]))
            
        for j in range(int(tar.shape[1]/2)):
            pos_tar.append((inp[t, 2*j], inp[t, 2*j+1], tar[t, j]))    
                 
        ax = plot_3d_graph(pos_tar, ax, color_edge = color_edge)
        ax = plot_3d_graph(pos_pred, ax, color_edge = color_edge, style='--')
        
        ax.set_axis_off()
        plt.savefig('pred.svg')
        import sys
        sys.exit()
    
        writer.grab_frame()