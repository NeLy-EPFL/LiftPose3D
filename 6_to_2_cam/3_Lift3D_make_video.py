import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
from matplotlib.legend_handler import HandlerTuple
from matplotlib.animation import FFMpegWriter
matplotlib.use('Agg')
import src.utils as utils
from skeleton import skeleton
#import pickle
from tqdm import tqdm

def average_cameras(tar):
    #average over cameras
    
    tar = np.stack( tar, axis=2 )
    tar[tar == 0] = np.nan
    tar_avg = np.nanmean(tar, axis=2)  
    tar_avg[np.isnan(tar_avg)] = 0
    
    return tar_avg


#cameras = [0,4]
cameras = [1,5]
#cameras = [2,6] #keep order, they come in L-R pairs!

root_dir = '/data/LiftFly3D/DF3D/cam_angles/cam'
#root_dir = '/data/LiftFly3D/DF3D/lift_vs_tri/cam'

#import
G, color_edge = skeleton() #skeleton

print('making video using cameras' + str(cameras))

#load data / statistics for cameras
out, tar = [], []
for cam in cameras:
    data_dir = root_dir + str(cam)
    
    #load stats
    tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
    tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
    targets_3d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_3d']
    cam_par = torch.load(data_dir + '/stat_3d.pth.tar')['rcams']
    cam_par = [vv for k,v in cam_par.items() for vv in v]
    offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
    offset = np.concatenate([v for k,v in offset.items()], 0)
    
    #load predictions
    data = torch.load(data_dir + '/test_results.pth.tar')
    tar_ = data['target']
    out_ = data['output']
    
    tar_ = utils.filter_data(tar_, window=5, order=2)
    out_ = utils.filter_data(out_, window=5, order=2)
    
    #expand
    tar_ = utils.expand(tar_, targets_3d, len(tar_mean))
    out_ = utils.expand(out_, targets_3d, len(tar_mean))
    
    #unnormalise
    tar_ = utils.unNormalizeData(tar_, tar_mean, tar_std) 
    out_ = utils.unNormalizeData(out_, tar_mean, tar_std)
    
    #translate legs back to their original places
    tar_ += offset[0,:]
    out_ += offset[0,:]
    
    #transform back to world
    tar_ = utils.camera_to_world(tar_, cam_par)
    out_ = utils.camera_to_world(out_, cam_par)
    
    #take only visible coordinates
    if cam==1:
        tar_[:,tar_.shape[1]//2:]=0
        out_[:,tar_.shape[1]//2:]=0
        
    if cam==5:
        tar_[:,:tar_.shape[1]//2]=0
        out_[:,:tar_.shape[1]//2]=0

    tar.append(tar_)
    out.append(out_)
    
#combine cameras
tar = average_cameras(tar)
out = average_cameras(out)


# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=20, azim=10)

writer = FFMpegWriter(fps=25)
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(900)):
        pos_pred, pos_tar = [], []
        
        ax.cla()
        
        for j in range(tar.shape[1]//3):
            pos_tar.append((tar[t, 3*j], tar[t, 3*j+1], tar[t, 3*j+2]))
            pos_pred.append((out[t, 3*j], out[t, 3*j+1], out[t, 3*j+2]))
            
        utils.plot_3d_graph(G,  pos_tar, ax, color_edge = color_edge)
        utils.plot_3d_graph(G, pos_pred, ax, color_edge = color_edge, style='--')
            
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p1, = ax.plot(pts, pts, pts, 'r-')
        p2, = ax.plot(pts, pts, pts, 'b-')
        p3, = ax.plot(pts, pts, pts, 'r--', dashes=(2, 2))
        p4, = ax.plot(pts, pts, pts, 'b--', dashes=(2, 2))
        ax.legend([(p1, p2), (p3, p4)], 
            ['Triangulated 3D pose', 'Lift3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.9),
            frameon=False)    
        p1.remove()
        p2.remove()
        p3.remove()
        p4.remove()
        ####    
           
        ax.set_xlim([-3,3])
        ax.set_ylim([-1,4])
        ax.set_zlim([0.5,2.5])
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True)
        
        writer.grab_frame()