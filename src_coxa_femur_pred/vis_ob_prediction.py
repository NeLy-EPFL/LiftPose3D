import os
import numpy as np

import glob
import copy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import data_utils

OB_LEFT_I  = np.array([0,1,2,4,5,6,8,9,10]) # start points
OB_LEFT_J  = np.array([1,2,3,5,6,7,9,10,11]) # end points
OB_RIGHT_I  = np.array([12,13,14,16,17,18,20,21,22]) # start points
OB_RIGHT_J  = np.array([13,14,15,17,18,19,21,22,23]) # end points

SAVE_VIDEO = False

def get_3d_pose(channels, ax):
    colors = [["#FF4D4D", "#FF0000", "#CC0000"],
              ["#4D4DFF", "#0000FF", "#0000CC"],
              ["#4DFF4D", "#00FF00", "#00CC00"]]
    cidx = 0
    for ch in channels:
        leg = 0
        for i in np.arange(len(OB_LEFT_I)):
            x, z, y = [np.array( [ch[OB_LEFT_I[i], j], ch[OB_LEFT_J[i], j]] ) for j in range(3)]
            y *= -1
            z *= -1
            ax.plot(x, y, z, lw=2, c=colors[cidx][leg])
            if (i+1)%3 == 0 : leg += 1
        leg = 0
        for i in np.arange(len(OB_RIGHT_I)):
            x, z, y = [np.array( [ch[OB_RIGHT_I[i], j], ch[OB_RIGHT_J[i], j]] ) for j in range(3)]
            y *= -1
            z *= -1
            ax.plot(x, y, z, lw=2, c=colors[cidx][leg])
            if (i+1)%3 == 0 : leg += 1
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

def update_ob_predic_viz(num, ob_predic, t, ax3d):
    ax3d.cla()
    ax3d.set_title(t + "\n" + str(num))
    ax3d.set_xlim([-2.5, 1.5])
    ax3d.set_ylim([-2, 2])
    ax3d.set_zlim([-1.5, 0.5])
    if SAVE_VIDEO:
        ax3d.view_initv(elev=30., azim=get_azim(num))
    channels = [ob_predic[num]]
    get_3d_pose(channels, ax3d)

def visualize_ob_predic(ob_predic):
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    start = 0 
    end = int(ob_predic.shape[0]/5)

    channels = [ob_predic[start]]
    get_3d_pose(channels, ax3d)
    ani = animation.FuncAnimation(fig, update_ob_predic_viz, end-start,
        fargs=(ob_predic, "OptoBot 3D prediction", ax3d), interval=80, blit=False)
    if SAVE_VIDEO:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=200)
        ani.save("videos/optobot_prediction.mp4", writer=writer)
    plt.show()

def compare_limbs_length(df_data, ob_data):
    names = ["Femur", "Tibia", "Tarsus"]
    for idx in range(data_utils.OB_JOINTS_PER_LEG-1):
        print(names[idx])
        print("\t    Limb1\tLimb2\t   Limb3\tLimb4\t    Limb5\tLimb6")
        df_dist_x = (df_data[:, data_utils.OB_BODY_COXA+idx, 0] -\
                df_data[:, data_utils.OB_BODY_COXA+idx+1, 0])**2
        df_dist_y = (df_data[:, data_utils.OB_BODY_COXA+idx, 2] -\
                df_data[:, data_utils.OB_BODY_COXA+idx+1, 2])**2
        df_dist_z = (df_data[:, data_utils.OB_BODY_COXA+idx, 1] -\
                df_data[:, data_utils.OB_BODY_COXA+idx+1, 1])**2
        df_dist = np.sqrt(df_dist_x + df_dist_y + df_dist_z)
        df_dist = np.mean(df_dist, axis=0) #df_dist = np.mean(df_dist)
        print("DeepFly: ", df_dist)

        ob_dist_x = (ob_data[:, data_utils.OB_BODY_COXA+idx, 0] -\
                ob_data[:, data_utils.OB_BODY_COXA+idx+1, 0])**2
        ob_dist_y = (ob_data[:, data_utils.OB_BODY_COXA+idx, 2] -\
                ob_data[:, data_utils.OB_BODY_COXA+idx+1, 2])**2
        ob_dist_z = (ob_data[:, data_utils.OB_BODY_COXA+idx, 1] -\
                ob_data[:, data_utils.OB_BODY_COXA+idx+1, 1])**2
        ob_dist = np.sqrt(ob_dist_x + ob_dist_y + ob_dist_z)
        ob_dist = np.mean(ob_dist, axis=0) #ob_dist = np.mean(ob_dist)
        print("OptoBot: ", ob_dist)
        print("Error: %.2f\n"%np.mean(np.abs(df_dist - ob_dist)))

if __name__ == '__main__':
    ob_predic = np.load("saved_structures/ob_predic.npy")
    df_data = np.load("saved_structures/df_data.npy")
    df_data = df_data[:, data_utils.DF_NON_COXA_FEMUR]

    compare_limbs_length(df_data, ob_predic)

    visualize_ob_predic(ob_predic)
