#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 06:30:06 2020

@author: adamgosztolai
"""

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

def unNormalizeData(data, data_mean, data_std, dim_to_use):
  """
  Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions that were removed from the original data
  Returns
    orig_data: the input normalized_data, but unnormalized
  """
      
  data *= data_std[dim_to_use]
  data += data_mean[dim_to_use]
  
  T = data.shape[0] # Batch size
  D = data_mean.shape[0] # Dimensionality
  orig_data = np.zeros((T, D), dtype=np.float32)
  orig_data[:, dim_to_use] = data
  
  return orig_data


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


def plot_3d_graph(pos, ax, color = 'k'):
    
    pos = np.array(pos)    
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=color, s=10, edgecolors='k', alpha=0.7)
           
    for _, j in enumerate(G.edges()): 
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
                   
        ax.plot(x, y, z, c=color, alpha=0.3, linewidth = 1)
        
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
nodes = list(range(38))
edges = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),(10,11),(11,12),(12,13),(13,14),(16,17),(17,18),(19,20),
         (20,21),(21,22),(22,23),(24,25),(25,26),(26,27),(27,28),(29,30),(30,31),(31,32),(32,33),(35,36),(36,37)]

edges = [e for e in edges if e[0]<15 and e[1]<15]
nodes = list(range(15))


#build graph
G=nx.Graph()
G.add_edges_from(edges)
#G.add_nodes_from(nodes)

#load data and prediction
data_dir = '/Users/adamgosztolai/Dropbox/'

xyz = torch.load(data_dir +'train_3d.pth.tar')
xy = torch.load(data_dir +'train_2d.pth.tar')
z = torch.load(data_dir +'train_z.pth.tar')
xyz_mean = torch.load(data_dir +'stat_3d.pth.tar')['mean']
xy_mean = torch.load(data_dir +'stat_2d.pth.tar')['mean']
z_mean = torch.load(data_dir +'stat_z.pth.tar')['mean']
xyz_std = torch.load(data_dir +'stat_3d.pth.tar')['std']
xy_std = torch.load(data_dir +'stat_2d.pth.tar')['std']
z_std = torch.load(data_dir +'stat_z.pth.tar')['std']
targets = torch.load(data_dir +'stat_z.pth.tar')['target_sets']
anchors = torch.load(data_dir +'stat_z.pth.tar')['anchors']
targets_xyz = get_coords_in_dim(targets, 3)
targets_xy = get_coords_in_dim(targets, 2)
targets_z = get_coords_in_dim(targets, 1)

xyz = xyz[(6, 'MDN_CsCh', 'pose_result__data_paper_180919_MDN_CsCh_Fly6_005_SG1_behData_images.pkl')]
xy = xy[(6, 'MDN_CsCh', 'pose_result__data_paper_180919_MDN_CsCh_Fly6_005_SG1_behData_images.cam_1')]
z = z[(6, 'MDN_CsCh', 'pose_result__data_paper_180919_MDN_CsCh_Fly6_005_SG1_behData_images.cam_1')]

#unnormalise
xyz_0 = np.zeros_like(xyz, dtype=np.float32)
xyz_0[:, targets_xyz] = xyz[:,targets_xyz]
xyz = xyz_0
xy = unNormalizeData(xy, xy_mean, xy_std, targets_xy)
z = unNormalizeData(z, z_mean, z_std, targets_z)

#xyz = de_anchor(, 3)
#xy = de_anchor(, 2)
#z = de_anchor(, 1)
#old_poses, new_poses, 
#           anchors = [0, 5, 10], 
#           target_sets = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]], 
#           dim=3)


#plot
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection = '3d')

metadata = dict(title='LiftFly3D prediction', artist='Nely',comment='Watch this!')
writer = FFMpegWriter(fps=10, metadata=metadata)
with writer.saving(fig, "LiftFly3d_prediction_test.mp4", 100):
    for t in range(100):
        pos = []
        pos2 = []
        ax.cla()
        for j in range(len(nodes)):
            pos.append((xyz[t, 3*j], xyz[t, 3*j+1], xyz[t, 3*j+2]))
            pos2.append((xy[t, 2*j], xy[t, 2*j+1], z[t, j]))
            
        ax = plot_3d_graph(pos, ax, color = 'r')
        ax = plot_3d_graph(pos2, ax, color = 'b')
        writer.grab_frame()