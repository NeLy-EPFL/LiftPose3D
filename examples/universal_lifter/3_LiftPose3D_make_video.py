import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
from matplotlib.animation import FFMpegWriter
from matplotlib.legend_handler import HandlerTuple
matplotlib.use('Agg')
import src.utils as utils
import src.transform as transform
import src.stat as stat
import src.plotting as plotting
from skeleton import skeleton
from tqdm import tqdm
import yaml
import sys

def average_cameras(tar):
    #average over cameras
    
    tar = np.stack( tar, axis=2 )
    tar[tar == 0] = np.nan
    tar_avg = np.nanmean(tar, axis=2)  
    tar_avg[np.isnan(tar_avg)] = 0
    
    return tar_avg


#specify folder
usr_input = sys.argv[-1]

#load global parameters
par = yaml.full_load(open(usr_input, "rb"))

cameras = [1,5]

#import
G, color_edge = skeleton() #skeleton
legtips = [4, 9, 14, 19, 24, 29]

print('making video using cameras ' + str(cameras))

#load data / statistics for cameras

# data_dir = root_dir #+ str(cam)
data_dir = par['out_dir']
    
#load stats
tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
targets_3d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_3d']
cam_par = torch.load(data_dir + '/stat_3d.pth.tar')['rcams']
cam_par = list(cam_par.values())
offset = torch.load(data_dir + '/stat_3d.pth.tar')['offset']
offset = np.concatenate([v for k,v in offset.items()], 0)
    
#load predictions
data = torch.load(data_dir + '/test_results.pth.tar')
tar_ = data['target']
out_ = data['output']
    
#expand
tar_ = utils.add_roots(tar_, targets_3d, len(tar_mean))
out_ = utils.add_roots(out_, targets_3d, len(tar_mean))    
    
#unnormalise
tar_ = stat.unNormalize(tar_, tar_mean, tar_std) 
out_ = stat.unNormalize(out_, tar_mean, tar_std) 
    
#translate legs back to their original places
tar_ += offset#[0,:]
out_ += offset#[0,:]

cam_number = []
for i in range(tar_.shape[0]):
    cam = int(data['keys'][i][2].split('.')[1][-1])
    cam_number.append(cam)
    
#transform back to worldcamera_to_world( poses_cam, cam_par, cam )
for cam in set(cam_number):
    ind = np.where(np.array(cam_number)==cam)[0]
    tar_[ind,:] = transform.camera_to_world(tar_[ind,:], cam_par[0][cam])
    out_[ind,:] = transform.camera_to_world(out_[ind,:], cam_par[0][cam])
    
#    tar_ = utils.filter_data(tar_, window=5, order=2)
#    out_ = utils.filter_data(out_, window=5, order=2)
    
out, tar = [], []
for cam in cameras:
    ind = np.where(np.array(cam_number)==cam)[0]
    tar.append(tar_[ind,:])
    out.append(out_[ind,:])
    
tar = np.concatenate(tar,axis=1)
out = np.concatenate(out,axis=1)

#combine cameras
#tar = average_cameras(tar)
#out = average_cameras(out)

# pts3d = pts3d - np.mean(pts3d, axis=0, keepdims=True)
from scipy.spatial.transform import Rotation as Rot
R = Rot.from_euler('xyz', [-np.pi/2, 0, -3*np.pi/4], degrees=False).as_matrix()
tar = transform.world_to_camera(tar, {'R': R,'tvec':[0,0,0]})
out = transform.world_to_camera(out, {'R': R,'tvec':[0,0,0]})

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=17, azim=123)

writer = FFMpegWriter(fps=10)
with writer.saving(fig, "LiftPose3D_prediction.mp4", 100):
    for t in tqdm(range(900)):
        pos_pred, pos_tar = [], []
        
        ax.cla()
        
        thist = 7
        pos_pred, pos_tar = [], []
        for j in range(out.shape[1]//3):
            tmin = max(0,t-thist+1)
            pos_pred.append((out[tmin:(t+1), 3*j], out[tmin:(t+1), 3*j+1], out[tmin:(t+1), 3*j+2]))
            pos_tar.append((tar[tmin:(t+1), 3*j], tar[tmin:(t+1), 3*j+1], tar[tmin:(t+1), 3*j+2]))
                
        pos_pred, pos_tar = np.array(pos_pred), np.array(pos_tar)
        
        plotting.plot_trailing_points(pos_pred[legtips,:,:],min(thist,t+1),ax)
        
        #plot skeleton
        plotting.plot_3d_graph(G, pos_tar[:,:,-1], ax, color_edge=color_edge)    
        plotting.plot_3d_graph(G, pos_pred[:,:,-1], ax, color_edge=color_edge, style='--')
            
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p1, = ax.plot(pts, pts, pts, 'r-')
        p2, = ax.plot(pts, pts, pts, 'b-')
        p3, = ax.plot(pts, pts, pts, 'r--', dashes=(2, 2))
        p4, = ax.plot(pts, pts, pts, 'b--', dashes=(2, 2))
        ax.legend([(p1, p2), (p3, p4)], 
            ['Triangulated 3D pose using 3 cameras per keypoint', \
             'LiftPose3D prediction using 1 camera per keypoint'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.9),
            frameon=False)    
        p1.remove()
        p2.remove()
        p3.remove()
        p4.remove()
        ####    
           
        ax.set_xlim([-2,2])
        ax.set_ylim([-4,4])
        ax.set_zlim([0,2.5])
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.grid(True)
        
        writer.grab_frame()