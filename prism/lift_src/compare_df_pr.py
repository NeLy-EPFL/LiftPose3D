import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

import data_utils
from procrustes import procrustes_separate

LEFT_I  = np.array([0,1,2,3, 5,6,7,8, 10,11,12,13]) # start points
LEFT_J  = np.array([1,2,3,4, 6,7,8,9, 11,12,13,14]) # end points
RIGHT_I  = np.array([15,16,17,18, 20,21,22,23, 25,26,27,28]) # start points
RIGHT_J  = np.array([16,17,18,19, 21,22,23,24, 26,27,28,29]) # end points

SAVE_VIDEO = False
PROJ_GROUND = True

def get_3d_pose(channels, ax):
    colors = [["#4DFF4D", "#00FF00", "#00CC00"],
              ["#4D4DFF", "#0000FF", "#0000CC"],
              ["#FF4D4D", "#FF0000", "#CC0000"]]
    cidx = 0
    for ch in channels:
        leg = 0
        for i in np.arange(len(LEFT_I)):
            x, z, y = [np.array( [ch[LEFT_I[i], j], ch[LEFT_J[i], j]] ) for j in range(3)]
            y *= -1
            z *= -1
            ax.plot(x, y, z, lw=2, c=colors[cidx][leg])
            if (i+1)%4 == 0 : leg += 1
        leg = 0
        for i in np.arange(len(RIGHT_I)):
            x, z, y = [np.array( [ch[RIGHT_I[i], j], ch[RIGHT_J[i], j]] ) for j in range(3)]
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

def update_files_viz(num, list_data3d, f, ax3d):
    ax3d.cla()
    ax3d.set_title(f + "\n" + str(num))
    ax3d.set_xlim([-2.5, 2.5])
    ax3d.set_ylim([-2.5, 2.5])
    channels = []
    for d3d in list_data3d:
        channels.append(d3d[num])
    get_3d_pose(channels, ax3d)

def visualize_files_atg(list_data3d):
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    start = 0
    shapes = [x.shape[0] for x in list_data3d]
    end = min(shapes)

    channels = []
    for d3d in list_data3d:
        channels.append(d3d[0])
    get_3d_pose(channels, ax3d)
    ani = animation.FuncAnimation(fig, update_files_viz, end-start,
        fargs=(list_data3d, "Prism in green. Deepfly in blue", ax3d), interval=1, blit=False)
    if SAVE_VIDEO:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=200)
        ani.save("videos/prism_deepfly_procrustes.mp4", writer=writer)
    plt.show()

def compare_limbs_length(df_data, pr_data):
    names = ["Coxa", "Femur", "Tibia", "Tarsus"]
    for idx in range(data_utils.JOINTS_PER_LEG):
        print(names[idx])
        print("\t    Limb1\tLimb2\t   Limb3\tLimb4\t    Limb5\tLimb6")
        df_dist_x = (df_data[:, data_utils.BODY_COXA+idx, 0] -\
                df_data[:, data_utils.BODY_COXA+idx+1, 0])**2
        df_dist_y = (df_data[:, data_utils.BODY_COXA+idx, 2] -\
                df_data[:, data_utils.BODY_COXA+idx+1, 2])**2
        df_dist = np.sqrt(df_dist_x + df_dist_y)
        df_dist = np.mean(df_dist, axis=0) #df_dist = np.mean(df_dist)
        print("DeepFly: ", df_dist)

        pr_dist_x = (pr_data[:, data_utils.BODY_COXA+idx, 0] -\
                pr_data[:, data_utils.BODY_COXA+idx+1, 0])**2
        pr_dist_y = (ob_data[:, data_utils.BODY_COXA+idx, 2] -\
                ob_data[:, data_utils.BODY_COXA+idx+1, 2])**2
        pr_dist = np.sqrt(pr_dist_x + pr_dist_y)
        pr_dist = np.mean(pr_dist, axis=0) #ob_dist = np.mean(ob_dist)
        print("OptoBot: ", pr_dist)
        print("Error: %.2f\n"%np.mean(np.abs(df_dist - pr_dist)))

def compare_distances_bc(df_data, pr_data):
    names = ["Front limbs", "Middle limbs", "Back limbs"]
    for n, bcl, bcr in zip(names, data_utils.BODY_COXA[:3], data_utils.BODY_COXA[3:]):
        print(n)
        df_dist_x = np.abs(df_data[:, bcl, 0] - df_data[:, bcr, 0])
        df_dist_x = np.mean(df_dist_x)
        df_dist_y = np.abs(df_data[:, bcl, 2] - df_data[:, bcr, 2])
        df_dist_y = np.mean(df_dist_y)
        print("DeepFly: (%.2f, %.2f)"%(df_dist_x, df_dist_y))
        pr_dist_x = np.abs(pr_data[:, bcl, 0] - pr_data[:, bcr, 0])
        pr_dist_x = np.mean(pr_dist_x)
        pr_dist_y = np.abs(pr_data[:, bcl, 2] - pr_data[:, bcr, 2])
        pr_dist_y = np.mean(pr_dist_y)
        print("Prism: (%.2f, %.2f)"%(pr_dist_x, pr_dist_y))
        print()

if __name__ == '__main__':
    df_data = np.load("saved_structures/df_data.npy")
    ob_data = np.load("saved_structures/ob_data.npy")

    df_data_depth = np.copy(df_data[:,:,1])
    if PROJ_GROUND:
        df_data[:,:,1] = 0
        ob_data[:,:,1] = 0

    compare_limbs_length(df_data, pr_data)
    compare_distances_bc(df_data, pr_data)

    print(df_data.shape)
    print(pr_data.shape)

    visualize_files_atg([df_data, pr_data])
