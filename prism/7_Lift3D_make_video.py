import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.legend_handler import HandlerTuple
import matplotlib
matplotlib.use('Agg')
from skeleton import skeleton
from tqdm import tqdm
import src.utils as utils

print('making video')

#specify folder
data_dir = '/data/LiftFly3D/prism/data_oriented/test_data/'

#load
G, color_edge = skeleton()
legtips = [4, 9, 14, 19, 24, 29]
data = torch.load(data_dir + '/test_results.pth.tar')

tar_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
tar_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
targets_1d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_1d']
tar_offset = np.vstack(torch.load(data_dir + '/stat_3d.pth.tar')['offset'].values())

inp_mean = torch.load(data_dir + '/stat_2d.pth.tar')['mean']
inp_std = torch.load(data_dir + '/stat_2d.pth.tar')['std']
targets_2d = torch.load(data_dir + '/stat_2d.pth.tar')['targets_2d']
inp_offset = np.vstack(torch.load(data_dir + '/stat_2d.pth.tar')['offset'].values())[0,:]

good_keypts = utils.expand(data['good_keypts'],targets_1d,len(tar_mean))
if np.sum(good_keypts[0,:15])>10:
    tar_offset = np.hstack((tar_offset[0,:15],tar_offset[0,:15]))
else:
    tar_offset = np.hstack((tar_offset[0,15:],tar_offset[0,15:]))

#unnormalize
tar = utils.unNormalizeData(data['target'], tar_mean[targets_1d], tar_std[targets_1d])
tar = utils.expand(tar,targets_1d,len(tar_mean))
tar += tar_offset
out = utils.unNormalizeData(data['output'], tar_mean[targets_1d], tar_std[targets_1d])
out = utils.expand(out,targets_1d,len(tar_mean))
out += tar_offset
inp = utils.unNormalizeData(data['input'], inp_mean[targets_2d], inp_std[targets_2d])
inp = utils.expand(inp,targets_2d,len(inp_mean))
inp += inp_offset

# Set up a figure
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40, azim=140)

writer = FFMpegWriter(fps=10)
xlim, ylim, zlim = None,None,None
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(600,1000)):
        
        plt.cla()
        
        thist = 5
        pos_pred, pos_tar = [], []
        for j in range(out.shape[1]):
            tmin = max(0,t-thist+1)
            pos_pred.append((inp[tmin:(t+1), 2*j], inp[tmin:(t+1), 2*j+1], out[tmin:(t+1), j]))
            pos_tar.append((inp[tmin:(t+1), 2*j], inp[tmin:(t+1), 2*j+1], tar[tmin:(t+1), j]))
                
        pos_pred, pos_tar = np.array(pos_pred), np.array(pos_tar)
                    
        #plot skeleton
        utils.plot_3d_graph(G, pos_tar[:,:,-1], ax, color_edge=color_edge, good_keypts=good_keypts[t,:])    
        utils.plot_3d_graph(G, pos_pred[:,:,-1], ax, color_edge=color_edge, style='--') 
            
        #plot trailing dots
        utils.plot_trailing_points(pos_pred[legtips,:,:],thist,ax)

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
            ['Triangulated 3D pose', 'Lift3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.9),
            frameon=False)    
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