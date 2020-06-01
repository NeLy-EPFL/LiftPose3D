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


def plot_3d_graph(pos, ax, color_node = 'k', color_edge = 'k', style = '-'):
    
    pos = np.array(pos)

    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=color_node, s=20, alpha=0.7)
           
    for i, j in enumerate(G.edges()): 
        if (j[0] < pos.shape[0]) & (j[1] < pos.shape[0]):
            x = np.array((pos[j[0]][0], pos[j[1]][0]))
            y = np.array((pos[j[0]][1], pos[j[1]][1]))
            z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
            ax.plot(x, y, z, style, c=color_edge[i], alpha=1.0, linewidth = 2)
        
    ax.view_init(elev = 60, azim=100)

    ax.set_axis_off()
    
    return ax


def average_cameras(out, tar):
    #average over cameras
    
    out = np.stack( out, axis=2 ) 
    tar = np.stack( tar, axis=2 )
    out[out == 0] = np.nan
    tar[tar == 0] = np.nan
    out_avg = np.nanmean(out, axis=2)
    tar_avg = np.nanmean(tar, axis=2)
    
    sqerr = (out_avg - tar_avg) ** 2
    sqerr[np.isnan(sqerr)] = 0
    
    n_pts = int(out.shape[1]/(3*2))
    err = np.zeros([sqerr.shape[0],n_pts])
    for k in range(n_pts):
        err[:, k] = np.sqrt(np.sum(sqerr[:, 3*k:3*k + 3], axis=1))
            
    joint_err = np.mean(err, axis=0)
    ttl_err = np.mean(joint_err[joint_err>0])
        
    return out_avg, tar_avg, joint_err, ttl_err


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
  
    elif dim == 3:
      dim_to_use = np.sort( np.hstack( (dim_to_use*3,
                                        dim_to_use*3+1,
                                        dim_to_use*3+2)))
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


cameras = [0,1]

#import skeleton of fly
G, color_edge = skeleton()

print('making video using cameras' + str(cameras))

_out, _tar, joint_errs, ttl_errs = [],[],[],[]
for cam in cameras:
    #load stats
    data_dir = '/data/LiftFly3D/prism/'
    
    ind = 'cam_' + str(cam)
    data_dir += 'cam_' + str(cam)
    
    inp_mean = torch.load(data_dir + '/stat_2d.pth.tar')['mean']
    inp_std = torch.load(data_dir + '/stat_2d.pth.tar')['std']
    tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
    tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
    targets = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']

    tar_offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
    tar_offset = np.vstack( tar_offset.values() )
    
    #load predictions
    data = torch.load(data_dir + '/test_results.pth.tar')
    
    #unnormalise    
    targets_3d = get_coords_in_dim(targets, 3)
    out = unNormalizeData(data['output'], tar_mean, tar_std, targets_3d)
    tar = unNormalizeData(data['target'], tar_mean, tar_std, targets_3d)

    #remove offset
    tar += tar_offset
    out += tar_offset
    
    if cam ==1:
        out = np.reshape(out, [-1, 3])
        out[:,1] = -1*out[:,1]
        out = np.reshape(out, [-1, 90] )
        tar = np.reshape(tar, [-1, 3])
        tar[:,1] = -1*tar[:,1]
        tar = np.reshape(tar, [-1, 90] )

    _out.append(out)
    _tar.append(tar)
    
tar = _tar[0]    
print(tar.shape)
out, _, joint_err, ttl_err = average_cameras(_out, _tar)
joint_errs.append(joint_err)
ttl_errs.append(ttl_err)
    
print(joint_errs)    
print(ttl_errs)  

#plot
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection = '3d')

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=10, metadata=metadata)
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(100):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
    
        for j in range(int(tar.shape[1]/3)):
            pos_tar.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_pred.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
                 
        ax = plot_3d_graph(pos_tar, ax, color_node='C1', color_edge = color_edge)
        ax = plot_3d_graph(pos_pred, ax, color_node='C0', color_edge = color_edge, style='--')
        
        writer.grab_frame()