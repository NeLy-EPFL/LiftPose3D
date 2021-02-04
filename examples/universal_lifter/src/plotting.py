import numpy as np
import matplotlib.pyplot as plt


def plot_3d_graph(G, pos, ax, color_edge=None, style=None, good_keypts=None):
    
    for i, j in enumerate(reversed(list(G.edges()))):
            
        if good_keypts is not None:
            if (good_keypts[j[0]]==0) | (good_keypts[j[1]]==0):
                continue
            
        #coordinates
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))
        
        #edge color
        if color_edge is not None:
            c = color_edge[j[0]]
        else:
            c = 'k'
            
        #edge style
        if style is None:
            style = '-'

        #plot           
        ax.plot(x, y, z, style, c=c, alpha=1.0, linewidth=2) 
        
        
def plot_skeleton(G, x, y, color_edge=None,  ax=None, good_keypts=None):
           
    for i, j in enumerate(G.edges()): 
        if good_keypts is not None:
            if (good_keypts[j[0]]==0) | (good_keypts[j[1]]==0):
                continue   
       
        u = np.array((x[j[0]], x[j[1]]))
        v = np.array((y[j[0]], y[j[1]]))
        
        #edge color
        if color_edge is not None:
            c = color_edge[j[0]]
        else:
            c = 'k'
            
        if ax is not None:
            ax.plot(u, v, c=c, alpha=1.0, linewidth = 2)
        else:
            plt.plot(u, v, c=c, alpha=1.0, linewidth = 2)  
        
        
def plot_trailing_points(pos,thist,ax):
    alphas = np.linspace(0.1, 1, thist)
    rgba_colors = np.zeros((thist,4))
    rgba_colors[:,[0,1,2]] = 0.8
    rgba_colors[:, 3] = alphas
    for j in range(pos.shape[0]):
        ax.scatter(pos[j,0,:], pos[j,1,:], pos[j,2,:], '-o', color=rgba_colors)
        for i in range(thist-1):
            if i<thist:
                ax.plot(pos[j,0,i:i+2], pos[j,1,i:i+2], pos[j,2,i:i+2], '-o', c=rgba_colors[i,:])