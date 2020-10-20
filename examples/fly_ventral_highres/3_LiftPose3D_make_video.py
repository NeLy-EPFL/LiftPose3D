import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
from matplotlib.animation import FFMpegWriter
from matplotlib.legend_handler import HandlerTuple
matplotlib.use('Agg')
import src.plotting as plotting
import src.utils as utils
import src.stat as stat
from skeleton import skeleton
from tqdm import tqdm
import yaml
import sys

print('making video')

#specify folder
usr_input = sys.argv[-1]

#load global parameters
par = yaml.full_load(open(usr_input, "rb"))

#load
G, color_edge = skeleton()
legtips = [4, 9, 14, 19, 24, 29]
data = torch.load(par['data_dir'] + '/test_results.pth.tar')

targets_1d = torch.load(par['data_dir'] + '/stat_3d.pth.tar')['targets_1d']
out_mean = torch.load(par['data_dir'] + '/stat_3d.pth.tar')['mean']
out_std = torch.load(par['data_dir'] + '/stat_3d.pth.tar')['std']
out = stat.unNormalize(data['output'], out_mean[targets_1d], out_std[targets_1d])
out_offset = np.vstack(torch.load(par['data_dir']  + '/stat_3d.pth.tar')['offset'].values())
good_keypts = np.vstack(torch.load(par['data_dir']  + '/stat_3d.pth.tar')['LR_test'].values())

good_keypts = utils.add_roots(good_keypts,targets_1d,len(out_mean))
if np.sum(good_keypts[0,:15])>10:
    out_offset = np.hstack((out_offset[0,:15],out_offset[0,:15]))
else:
    out_offset = np.hstack((out_offset[0,15:],out_offset[0,15:]))

#inputs
targets_2d = torch.load(par['data_dir'] + '/stat_2d.pth.tar')['targets_2d']    
inp_mean = torch.load(par['data_dir'] + '/stat_2d.pth.tar')['mean']
inp_std = torch.load(par['data_dir'] + '/stat_2d.pth.tar')['std']
inp = stat.unNormalize(data['input'], inp_mean[targets_2d], inp_std[targets_2d])
inp_offset = np.vstack(torch.load(par['template_dir'] + '/stat_2d.pth.tar')['offset'].values())[0,:]

targets_1d = torch.load(par['template_dir'] + '/stat_3d.pth.tar')['targets_1d'] 
targets_2d = torch.load(par['template_dir'] + '/stat_2d.pth.tar')['targets_2d'] 
out = utils.add_roots(out,targets_1d,len(out_mean))
inp = utils.add_roots(inp,targets_2d,len(inp_mean))

out += out_offset
inp += inp_offset

out = utils.filter_data(out, window=9, order=2)
inp = utils.filter_data(inp, window=9, order=2)

# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=0.97)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40, azim=140)

writer = FFMpegWriter(fps=10)
xlim, ylim, zlim = None, None, None
thist = 20
with writer.saving(fig, "LiftPose3D_prediction.mp4", 100):
    for t in tqdm(range(100)):
              
        plt.cla()
        
        pos_pred = []
        for j in range(out.shape[1]):
            tmin = max(0,t-thist+1)
            pos_pred.append((inp[tmin:(t+1), 2*j], inp[tmin:(t+1), 2*j+1], out[tmin:(t+1), j]))
                
        pos_pred = np.array(pos_pred)
        
        #plot skeleton
        plotting.plot_3d_graph(G, pos_pred[:,:,-1], ax, color_edge=color_edge) 
            
        #plot trailing points        
        plotting.plot_trailing_points(pos_pred[legtips,:,:],min(thist,t+1),ax)
        
        if xlim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
              
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p3, = ax.plot(pts, pts, pts, 'r-')
        p4, = ax.plot(pts, pts, pts, 'b-')
        ax.legend([(p3, p4)], 
            ['LiftPose3D prediction'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.95),
            frameon=False, 
            fontsize=15)    
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