import os
import numpy as np

import glob
import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

DF_LEFT_I  = np.array([0,1,2,3, 5,6,7,8, 10,11,12,13]) # start points
DF_LEFT_J  = np.array([1,2,3,4, 6,7,8,9, 11,12,13,14]) # end points
DF_RIGHT_I  = np.array([15,16,17,18, 20,21,22,23, 25,26,27,28]) # start points
DF_RIGHT_J  = np.array([16,17,18,19, 21,22,23,24, 26,27,28,29]) # end points

SAVE_VIDEO = False

def get_3d_pose(channels, ax):
    colors = [["#4D4DFF", "#0000FF", "#0000CC"],
              ["#FF4D4D", "#FF0000", "#CC0000"],
              ["#4DFF4D", "#00FF00", "#00CC00"]]
    cidx = 0
    for ch in channels:
        leg = 0
        for i in np.arange(len(DF_LEFT_I)):
            x, z, y = [np.array( [ch[DF_LEFT_I[i], j], ch[DF_LEFT_J[i], j]] ) for j in range(3)]
            y *= -1
            z *= -1
            ax.plot(x, y, z, lw=2, c=colors[cidx][leg])
            if (i+1)%4 == 0 : leg += 1
        leg = 0
        for i in np.arange(len(DF_RIGHT_I)):
            x, z, y = [np.array( [ch[DF_RIGHT_I[i], j], ch[DF_RIGHT_J[i], j]] ) for j in range(3)]
            y *= -1
            z *= -1
            ax.plot(x, y, z, lw=2, c=colors[cidx][leg])
            if (i+1)%4 == 0 : leg += 1
        cidx += 1

    # Get rid of the ticks and tick labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().set_ticklabels([])
    ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)

def get_azim(frame_num):
    frame_num /= 4
    div = int(frame_num / 360)+1
    return frame_num - 180 * div

def update_predic_viz(num, df_test_data, df_predic, ax3d):
    ax3d.cla()
    ax3d.set_title("prediction in RED {0}".format(num))
    ax3d.set_xlim([-2.5, 1.5])
    ax3d.set_ylim([-2, 2])
    ax3d.set_zlim([-1.5, 0.5])
    if SAVE_VIDEO:
        ax3d.view_initv(elev=30., azim=get_azim(num))
    channels = [ df_test_data[num], df_predic[num] ]
    get_3d_pose(channels, ax3d)

def visualize_df_predic(df_test_data, df_predic):
    """ Visualize animation of the test and predicted data, on the same plot
        Args:
          df_test_data: DeepFly3D data used fot testing
          df_predic: DeepFly3D predicted data
    """
    start = 0
    end = df_test_data.shape[0]
    
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_title("prediction in RED 0")
    channels = [df_test_data[start], df_predic[start]]
    get_3d_pose(channels, ax3d)
    ani = animation.FuncAnimation(fig, update_predic_viz, end-start,
      fargs=(df_test_data, df_predic, ax3d), interval=1, blit=False)
    if SAVE_VIDEO:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=200)
        ani.save("videos/df_predictions.mp4", writer=writer)
    plt.show()

if __name__ == '__main__':
    df_predic = np.load("saved_structures/df_predic.npy")
    df_test_data = np.load("saved_structures/df_test_data.npy")

    visualize_df_predic(df_test_data, df_predic)
