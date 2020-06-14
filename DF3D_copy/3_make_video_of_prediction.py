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


def plot_3d_graph(pos, ax, color_edge, style = '-'):
    
    pos = np.array(pos)

    for i, j in enumerate(G.edges()): 
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
        ax.plot(x, y, z, style, c=color_edge[i], alpha=1.0, linewidth = 2)
    
    return ax


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

  R, T, _, _, _ = cam_par[cam]
    
  Pcam = np.reshape(data, [-1, 3]).copy()
  Pcam -= T
  Pworld = np.matmul(np.linalg.inv(R), Pcam.T).T
  
  return np.reshape( Pworld, [-1, data.shape[1]] )


def average_cameras(out):
    #average over cameras
    
    out = np.stack( out, axis=2 ) 
    out[out == 0] = np.nan
    out_avg = np.nanmean(out, axis=2)
        
    return out_avg


cameras = [1, 5]
data_dir = '/data/LiftFly3D/DF3D/cam_angles_2/cams15'

#import
G, color_edge = skeleton() #skeleton
cam_par = pickle.load(open('cameras.pkl', "rb")) #camera parameters

print('making video')

#load predictions
data = torch.load(data_dir + '/test_results.pth.tar')
keys = data['keys']

cam = [int(keys[i][2][j][-1]) for i in range(len(keys)) for j in range(len(keys[i][2]))]

#load stats
tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
targets = torch.load(data_dir + '/stat_3d.pth.tar')['target_sets']
targets_3d = get_coords_in_dim(targets, 3)
offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']    
offset = np.vstack( offset.values() )
    
#split to camera L and R
cam_L = np.where(np.array(cam)==cameras[0])[0]
cam_R = np.where(np.array(cam)==cameras[1])[0]

#unnormalise    
out = unNormalizeData(data['output'], tar_mean, tar_std, targets_3d)
tar = unNormalizeData(data['target'], tar_mean, tar_std, targets_3d)

out_L, tar_L = out[cam_L,:], tar[cam_L,:]
out_R, tar_R = out[cam_R,:], tar[cam_R,:]
#out_L += offset[cam_L[0],:]
#tar_L += offset[cam_L[0],:]
#out_R += offset[cam_R[0],:]
#tar_R += offset[cam_R[0],:]

#transform back to world
out_L = camera_to_world(out_L, cam_par, cameras[0])
out_R = camera_to_world(out_R, cam_par, cameras[1])
tar_L = camera_to_world(tar_L, cam_par, cameras[0])
tar_R = camera_to_world(tar_R, cam_par, cameras[1])

out = average_cameras([out_L, out_R])
tar = average_cameras([tar_L, tar_R])  
    
# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev = 00, azim=80)

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=25, metadata=metadata)
xlim, ylim, zlim = None,None,None
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in range(10):
        pos_pred = []
        pos_tar = []
        
        ax.cla()
        
        for j in range(int(out.shape[1]/3)):
            pos_tar.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_pred.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
            
        ax = plot_3d_graph(pos_tar, ax, color_edge = color_edge)
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