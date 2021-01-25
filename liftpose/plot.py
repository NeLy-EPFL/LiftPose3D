import glob

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import networkx as nx
import numpy as np


def plot_pose_3d(ax, tar, bones, normalize=True, pred=None, limb_id=None, colors=None, good_keypts=None):
    tar = tar.copy()
    if normalize: # move points toward origin for easier visualization
        tar_m = np.nanmedian(tar, axis=0)
        tar -= tar_m
        if pred is not None:
            pred = pred.copy()
            pred -= tar_m

    G = nx.Graph()
    G.add_edges_from(bones)
    G.add_nodes_from(np.arange(tar.shape[0]))
    
    # if limb_id or colors are not provided, then paint everything in blue
    if limb_id is None or colors is None:
        edge_colors = [[0,0,1.0] for _ in limb_id]
    else:
        edge_colors = [[x / 255.0 for x in colors[i]] for i in limb_id]

    plot_3d_graph(G, tar, ax, color_edge=edge_colors, good_keypts=good_keypts)
    if pred is not None:
        plot_3d_graph(G, pred, ax, color_edge=edge_colors, style="--")

    #### this bit is just to make special legend
    pts = tar.mean(axis=0)
    (p1,) = ax.plot(pts[[0]], pts[[1]], pts[[2]], "r-")
    # (p2,) = ax.plot(pts, pts, pts, "b-")
    (p3,) = ax.plot(pts[[0]], pts[[1]], pts[[2]], "r--", dashes=(2, 2))
    # (p4,) = ax.plot(pts, pts, pts, "b--", dashes=(2, 2))
    ax.legend(
        [(p1), (p3)],
        ["Triangulated 3D pose", "LiftPose3D prediction"] if pred is not None else ["Triangulated 3D pose"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc=(0.1, 0.9),
        frameon=False,
    )
    p1.remove()
    p3.remove()
    ####


def plot_3d_graph(G, pos, ax, color_edge=None, style=None, good_keypts=None):
    for i, j in enumerate(reversed(list(G.edges()))):

        if good_keypts is not None:
            if (good_keypts[j[0]] == 0) | (good_keypts[j[1]] == 0):
                continue
            if np.any(np.isnan(pos[j[0]])) or np.any(np.isnan(pos[j[1]])):
                continue

        # coordinates
        x = np.array((pos[j[0]][0], pos[j[1]][0]))
        y = np.array((pos[j[0]][1], pos[j[1]][1]))
        z = np.array((pos[j[0]][2], pos[j[1]][2]))

        # edge color
        if color_edge is not None:
            c = color_edge[j[0]]
        else:
            c = "k"

        # edge style
        if style is None:
            style = "-"

        # plot
        ax.plot(x, y, z, style, c=c, alpha=1.0, linewidth=2)


def plot_skeleton(G, x, y, color_edge, ax=None, good_keypts=None):
    for i, j in enumerate(G.edges()):
        if good_keypts is not None:
            if (good_keypts[j[0]] == 0) | (good_keypts[j[1]] == 0):
                continue

        u = np.array((x[j[0]], x[j[1]]))
        v = np.array((y[j[0]], y[j[1]]))
        if ax is not None:
            ax.plot(u, v, c=color_edge[j[0]], alpha=1.0, linewidth=2)
        else:
            plt.plot(u, v, c=color_edge[j[0]], alpha=1.0, linewidth=2)


def plot_trailing_points(pos, thist, ax):
    alphas = np.linspace(0.1, 1, thist)
    rgba_colors = np.zeros((thist, 4))
    rgba_colors[:, [0, 1, 2]] = 0.8
    rgba_colors[:, 3] = alphas
    for j in range(pos.shape[0]):
        ax.scatter(pos[j, 0, :], pos[j, 1, :], pos[j, 2, :], "-o", color=rgba_colors)
        for i in range(thist - 1):
            if i < thist:
                ax.plot(
                    pos[j, 0, i : i + 2],
                    pos[j, 1, i : i + 2],
                    pos[j, 2, i : i + 2],
                    "-o",
                    c=rgba_colors[i, :],
                )


def plot_log_train(ax, loss_train, loss_test, epochs):
    ax.plot(epochs, loss_train, label="train")
    ax.plot(epochs, loss_test, label="test")
    # ax.xscale('log')
    # ax.xlim([0, 100])
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()
    # ax.savefig("training_error.svg")


def read_log_train(out_dir):
    file = glob.glob(out_dir + "/log_train.txt")[0]
    f = open(file, "r")
    contents = f.readlines()
    epoch, lr, loss_train, loss_test, err_test = [], [], [], [], []
    for i in range(1, len(contents)):
        line = contents[i][:-1].split("\t")
        epoch.append(float(line[0]))
        lr.append(float(line[1]))
        loss_train.append(float(line[2]))
        loss_test.append(float(line[3]))
        err_test.append(float(line[4]))

    return epoch, lr, loss_train, loss_test, err_test
