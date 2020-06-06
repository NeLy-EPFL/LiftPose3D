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
import matplotlib
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.animation import FFMpegWriter
matplotlib.use('Agg')
from src.normalize import unNormalizeData, get_coords_in_dim
from skeleton import skeleton
import pickle

#from model import LinearModel

def plot_3d_graph(pos, ax, color_node = 'k', color_edge = 'k', style = '-'):
    
    pos = np.array(pos)
           
    for i, j in enumerate(G.edges()): 
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
        ax.plot(x, y, z, style, c=color_edge[i], alpha=1.0, linewidth = 2)
    
    return ax


def average_cameras(out, tar):
    #average over cameras
    
    out = np.stack( out, axis=2 ) 
    tar = np.stack( tar, axis=2 )
    out[out == 0] = np.nan
    tar[tar == 0] = np.nan
    out_avg = np.nanmean(out, axis=2)
    tar_avg = np.nanmean(tar, axis=2)   
        
    return out_avg, tar_avg


def camera_to_world( data, cam_par, cam ):
  """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    cam_ids: camera_ids to consider
  Returns
    transf: dictionary with 3d poses or 2d poses if projection is True
  """


  R, T, _, _ = cam_par[cam]
    
  Pcam = np.reshape(data, [-1, 3]).copy()
  Pcam -= T
  Pworld = np.matmul(np.linalg.inv(R), Pcam.T).T
  
  return np.reshape( Pworld, [-1, data.shape[1]] )


#cameras = [0,4]
cameras = [1,5]
#cameras = [2,6] #keep order, they come in L-R pairs!

#import
G, color_edge = skeleton() #skeleton
cam_par = pickle.load(open('cameras.pkl', "rb")) #camera parameters

print('making video using cameras' + str(cameras))

#load data / statistics for cameras
out, tar = [], []
data = {}
for cam in cameras:
    data_dir = '/data/LiftFly3D/DF3D/cam_angles/cam_' + str(cam)
    
    #load stats
    tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
    tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
    targets = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']
    offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']    
    offset = np.vstack( offset.values() )
    
    #load predictions
    data = torch.load(data_dir + '/test_results.pth.tar')

    #unnormalise    
    targets_3d = get_coords_in_dim(targets, 3)
    out_ = unNormalizeData(data['output'], tar_mean, tar_std, targets_3d)
    tar_ = unNormalizeData(data['target'], tar_mean, tar_std, targets_3d)
    
    #remove offset
#    out_ += offset
#    tar_ += offset
    
    #transform back to world
    out_ = camera_to_world(out_,cam_par,cam)
    tar_ = camera_to_world(tar_,cam_par,cam)
        
    out.append(out_)
    tar.append(tar_)
    
    
#combine cameras
out, tar = average_cameras(out, tar)

#plot
fig = plt.figure(figsize=(6, 5))

left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005
ax = fig.add_axes([left, bottom, width, height])
ax.set_axis_off()
plt.axis('equal')
ax = plt.gca(projection = '3d')
ax.view_init(elev = 70, azim=100)
ax_upper = fig.add_axes([left, bottom + height + spacing, width, 0.2])
ax_upper.set_axis_off()

p1, = ax_upper.plot([1, 2.5, 3], 'r-')
p2, = ax_upper.plot([3, 2, 1], 'b-')
p3, = ax_upper.plot([1, 2.5, 3], 'r--', dashes=(2, 2))
p4, = ax_upper.plot([3, 2, 1], 'b--', dashes=(2, 2))

ax_upper.legend([(p1, p2), (p3, p4)], ['Triangulated 3D pose', 'LiftFly3D prediction'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})
p1.remove()
p2.remove()
p3.remove()
p4.remove()

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=25, metadata=metadata)
xlim, ylim, zlim = None,None,None

with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(900):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
        
        for j in range(int(out.shape[1]/3)):
            pos_pred.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_tar.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
            
        ax = plot_3d_graph(pos_tar, ax, color_node='C1', color_edge = color_edge)
        ax = plot_3d_graph(pos_pred, ax, color_node='C0', color_edge = color_edge, style='--')
        ax.set_axis_off()
        
        if xlim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
        
#        plt.savefig('text.svg')
#        import sys
#        sys.exit()
        
        writer.grab_frame()