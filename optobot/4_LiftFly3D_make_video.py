import torch
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.legend_handler import HandlerTuple
import matplotlib
matplotlib.use('Agg')
from skeleton import skeleton
import pickle
from tqdm import tqdm
import src.utils as utils

G, color_edge = skeleton() #skeleton of fly
legtips = [4, 9, 14, 19, 24, 29]
print('making video')

data_dir = '/data/LiftFly3D/optobot/102906_s1a5_p6-0' #data directory
template_dir = '/data/LiftFly3D/optobot/network/'
        
#predictions
data = torch.load(data_dir + '/test_results.pth.tar')
out_offset, inp_offset = pickle.load(open('joint_locations.pkl','rb'))

#output
targets_1d = torch.load(data_dir + '/stat_3d.pth.tar')['targets_1d']
out_mean = torch.load(data_dir + '/stat_3d.pth.tar')['mean']
out_std = torch.load(data_dir + '/stat_3d.pth.tar')['std']
out = utils.unNormalizeData(data['output'], out_mean[targets_1d], out_std[targets_1d])

#inputs
targets_2d = torch.load(template_dir + '/stat_2d.pth.tar')['targets_2d']    
inp_mean = torch.load(template_dir + 'stat_2d.pth.tar')['mean']
inp_std = torch.load(template_dir + 'stat_2d.pth.tar')['std']
inp = utils.unNormalizeData(data['input'], inp_mean[targets_2d], inp_std[targets_2d])

targets_1d = torch.load(template_dir + '/stat_3d.pth.tar')['targets_1d'] 
targets_2d = torch.load(template_dir + '/stat_2d.pth.tar')['targets_2d'] 
out = utils.expand(out,targets_1d,len(out_mean))
inp = utils.expand(inp,targets_2d,len(inp_mean))

out *= 6

out += out_offset
inp += inp_offset


# Set up a figure twice as tall as it is wide
fig = plt.figure(figsize=plt.figaspect(1))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=40, azim=140)

writer = FFMpegWriter(fps=10)
xlim, ylim, zlim = None,None,None
l, trail = {}, {}
thist = 5
with writer.saving(fig, "prediction_cams.mp4", 100):
    for t in tqdm(range(2399)):
              
        pos_pred = []
        for j in range(out.shape[1]):
            tmin = max(0,t-thist)
            pos_pred.append((inp[tmin:(t+1), 2*j], inp[tmin:(t+1), 2*j+1], -out[tmin:(t+1), j]))
                
        pos_pred = np.array(pos_pred)
        
        l = utils.plot_3d_graph(G, pos_pred[:,:,-1], ax, l, color_edge=color_edge) 
            
        for leg in legtips:
            if leg not in trail.keys():
                trail[leg], = ax.plot(pos_pred[leg,0,:], pos_pred[leg,1,:], pos_pred[leg,2,:], c='y', alpha=0.9)
            else:
                trail[leg].set_xdata(pos_pred[leg,0,:])
                trail[leg].set_ydata(pos_pred[leg,1,:])
                trail[leg].set_3d_properties(pos_pred[leg,2,:], zdir='z')     
        
        if xlim is None:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            zlim = ax.get_zlim()
              
        #### this bit is just to make special legend 
        pts = np.array([1,1])
        p3, = ax.plot(pts, pts, pts, 'r--', dashes=(2, 2))
        p4, = ax.plot(pts, pts, pts, 'b--', dashes=(2, 2))
        ax.legend([(p3, p4)], 
            ['LiftFly3D prediction \n(x0.25 real time)'], 
            numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)},
            loc=(0.1,0.85),
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
        
#        ax.set_axis_off()
#        plt.savefig('test.svg')
#        import sys
#        sys.exit()
    
        writer.grab_frame()