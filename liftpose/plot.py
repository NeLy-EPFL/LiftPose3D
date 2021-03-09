from matplotlib.legend_handler import HandlerTuple
import networkx as nx
import numpy as np

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import matplotlib.animation

from typing import Optional, List, Callable


def plot_video_3d(fig, ax, n: int, fps: int, draw_function: Callable, name: str):
    """
    draw_function should take matplotlib axis object and frame id 
    def f(ax3d, idx):
        plot_pose_3d(ax=ax3d, tar=test_3d_gt[idx],
                 pred=test_3d_pred_aligned[idx],
                 bones=par_data["vis"]["bones"], 
                 limb_id=par_data["vis"]["limb_id"], 
                 colors=par_data["vis"]["colors"], 
                 normalize=True)
    plot_video_3d(fig, ax3d, n=2, fps=1, draw_function=f, name='kek.mp4')
    """
    writer = matplotlib.animation.FFMpegWriter(fps=fps)
    with writer.saving(fig, name, dpi=100):
        for i in range(n):
            draw_function(ax, i)
            writer.grab_frame()
            ax.clear()


def plot_pose_3d(
    ax,
    tar,
    bones,
    normalize=True,
    pred=None,
    limb_id=None,
    colors=None,
    good_keypts=None,
    show_pred_always=False,
    show_gt_always=False,
    legend=False,
):
    """
    Plot 3D pose


    Parameters
    ----------
    ax : matplotlib axes object
    tar : n x 3 numpy array
        Positions for n joints.
    bones : list of lists of integers
        Bones in skeleton.
    normalize : True/False, optional
        Center pose by mean. The defauls is False.
    pred : n x 3 numpy array
        Positions for n joints. Useful when comparing target to prediction. The default is None.
    limb_id : list of integers, optional
        Numbers represent which leg the joint belongs to. The default is None.
    colors : list of triples, optional
        Color assigned to bones for each leg. The default is None.
    good_keypts_tar : n x 3 boolean array, optional
        Selectively plot keypoints where good_keypoints_tar is 1. The default is None.
    good_keypts_pred : n x 3 boolean array, optional
        Selectively plot keypoints where good_keypoints_pred is 1. The default is None.

    Returns
    -------
    None.

    """
    assert tar.ndim == 2

    tar = tar.copy()
    if normalize:  # move points toward origin for easier visualization
        tar_m = np.nanmedian(tar, axis=0, keepdims=True)
        tar -= tar_m
        if pred is not None:
            pred = pred.copy()
            pred -= tar_m

    G = nx.Graph()
    G.add_edges_from(bones)
    G.add_nodes_from(np.arange(tar.shape[0]))

    # if limb_id or colors are not provided, then paint everything in blue
    if limb_id is None or colors is None:
        edge_colors = [[0, 0, 1.0] for _ in limb_id]
    else:
        edge_colors = [[x / 255.0 for x in colors[i]] for i in limb_id]

    plot_3d_graph(
        G,
        tar,
        ax,
        color_edge=edge_colors,
        good_keypts=good_keypts if not show_gt_always else None,
    )
    if pred is not None:
        plot_3d_graph(
            G,
            pred,
            ax,
            color_edge=edge_colors,
            style="--",
            good_keypts=good_keypts if not show_pred_always else None,
        )

    #### this bit is just to make special legend
    pts = np.nanmean(tar, axis=0)
    (p1,) = ax.plot(pts[[0]], pts[[1]], pts[[2]], "r-")
    # (p2,) = ax.plot(pts, pts, pts, "b-")
    (p3,) = ax.plot(pts[[0]], pts[[1]], pts[[2]], "r--", dashes=(2, 2))
    # (p4,) = ax.plot(pts, pts, pts, "b--", dashes=(2, 2))
    if legend:
        ax.legend(
        [(p1), (p3)],
        ["Triangulated 3D pose", "LiftPose3D prediction"]
        if pred is not None
        else ["Triangulated 3D pose"],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc=(0.1, 0.9),
        frameon=False,
        )
        p1.remove()
        p3.remove()


def plot_pose_2d(
    ax, 
    tar, 
    bones,
    normalize=True, 
    limb_id=None, 
    colors=None, 
    good_keypts=None
):
    """
    Plot 2D pose

    Parameters
    ----------
    ax : matplotlib axes object
    tar : n x 2 numpy array
        Positions for n joints.
    bones : list of lists of integers
        Bones in skeleton.
    normalize : True/False, optional
        Center pose by mean. The defauls is False.
    limb_id : list of integers, optional
        Numbers represent which leg the joint belongs to. The default is None.
    colors : list of triples, optional
        Color assigned to bones for each leg. The default is None.
    good_keypts : n x 2 boolean array, optional
        Selectively plot keypoints where good_keypoints_tar is 1. The default is None.

    Returns
    -------
    None.

    """

    tar = tar.copy()
    if normalize:  # move points toward origin for easier visualization
        tar_m = np.nanmedian(tar, axis=0, keepdims=True)
        tar -= tar_m

    G = nx.Graph()
    G.add_edges_from(bones)
    G.add_nodes_from(np.arange(tar.shape[0]))

    # if limb_id or colors are not provided, then paint everything in blue
    if limb_id is None or colors is None:
        edge_colors = [[0, 0, 1.0] for _ in limb_id]
    else:
        edge_colors = [[x / 255.0 for x in colors[i]] for i in limb_id]

    plot_2d_graph(
        G, tar, ax, color_edge=edge_colors, good_keypts=good_keypts,
    )


def plot_3d_graph(G, pos, ax, color_edge=None, style=None, good_keypts=None):
    """
    Plot 3D graph, called by plot_pose_3d()

    Parameters
    ----------
    G : networkx graph object
        Graph with nodes being keypoints and edge being bones.
    pos : n x 3 numpy array
        Position of keypoints.
    ax : matplotlib axes object
    color_edge : list of triples, optional
        Color assigned to bones for each leg. The default is None.
    style : '.', '--' ,'._', optional
        Line style. The default is None.
    good_keypts : n x 3 boolean array, optional
        Selectively plot keypoints where good_keypoints_tar is 1. The default is None.

    Returns
    -------
    None.

    """

    for i, j in enumerate(reversed(list(G.edges()))):

        if good_keypts is not None:
            if (good_keypts[j[0]][0] == 0) | (good_keypts[j[1]][0] == 0):
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


def plot_2d_graph(G, pos, ax, color_edge=None, style=None, good_keypts=None):
    """
    Plot 2D graph, called by plot_pose_2d()

    Parameters
    ----------
    G : networkx graph object
        Graph with nodes being keypoints and edge being bones.
    pos : n x 2 numpy array
        Position of keypoints.
    ax : matplotlib axes object
    color_edge : list of triples, optional
        Color assigned to bones for each leg. The default is None.
    style : '.', '--' ,'._', optional
        Line style. The default is None.
    good_keypts : n x 2 boolean array, optional
        Selectively plot keypoints where good_keypoints_tar is 1. The default is None.

    Returns
    -------
    None.

    """

    for i, j in enumerate(G.edges()):
        if good_keypts is not None:
            if (good_keypts[j[0]] == 0) | (good_keypts[j[1]] == 0):
                continue

        u = np.array((pos[j[0], 0], pos[j[1], 0]))
        v = np.array((pos[j[0], 1], pos[j[1], 1]))

        # edge color
        if color_edge is not None:
            c = color_edge[j[0]]
        else:
            c = "k"

        # edge style
        if style is None:
            style = "-"

        ax.plot(u, v, c=c, alpha=1.0, linewidth=2)


def plot_trailing_points(pos, thist, ax):
    """
    Plot lagging, trailing points when moving legs

    Parameters
    ----------
    pos : n x 3 x t numpy array
        Positions of all n keypoints for t timesteps.
    thist : integer
        Number of points trailign behind.
    ax : matplotlib axis object

    Returns
    -------
    None.

    """
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
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()


def read_log_train(out_dir: str):
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


def get_violin_ylabel(body_length, units):
    if body_length is not None:
        return "Percentage of body length"
    if units is not None:
        return f"Error ({units})"
    return "Error (unitless)"


def pred_and_gt_to_pandas(
    test_3d_gt, test_3d_pred, test_keypoints, joints_name, body_length
):

    err_norm = np.mean(np.abs(test_3d_gt - test_3d_pred), axis=-1)

    err_norm_sp = err_norm.copy()
    for j in range(err_norm.shape[1]):
        q = np.quantile(err_norm[:, j], 0.95)
        err_norm_sp[err_norm_sp[:, j] > q, j] = q

    # TODO vectorize the loop
    e_list = list()
    n_list = list()
    for i in range(err_norm_sp.shape[0]):
        for j in range(err_norm_sp.shape[1]):
            if not np.isnan(err_norm_sp[i, j]):
                if test_keypoints[i, j, 0]:
                    e_list.append(
                        err_norm_sp[i, j]
                        if body_length is None
                        else err_norm_sp[i, j] / body_length * 100
                    )
                    n_list.append(joints_name[j])

    # remove outliers
    d = pd.DataFrame({"err": e_list, "joint": n_list})
    q = d.quantile(q=0.95)
    d = d.loc[d["err"] < q["err"]]

    return d


def violin_plot(
    ax,
    test_3d_gt: np.ndarray,
    test_3d_pred: np.ndarray,
    test_keypoints: np.ndarray,
    joints_name: List[str],
    body_length: int = None,
    units: str = None,
    ylim: List[int] = None,
    order: List[str] = None,
    hue_names: List[str] = None
):
    """ creates violin plot of error distribution for each joint """
    if isinstance(test_3d_pred, np.ndarray):
        test_3d_pred = [test_3d_pred]
    # fmt: off
    #assert test_3d_pred[0].shape[1] == len(joints_name), f"given 3d data has {test_3d_pred[0].shape[1]} joints however joints_name only has {len(joints_name)} names"
    #assert (test_3d_pred[0].shape == test_3d_pred.shape), "ground-truth and prediction shapes do not match"
    # fmt: on

    pandas_list = [
        pred_and_gt_to_pandas(
            test_3d_gt, test_3d_pred[i], test_keypoints, joints_name, body_length
        )
        for i in range(len(test_3d_pred))
    ]

    for i, p in enumerate(pandas_list):
        p["hue"] = hue_names[i] if hue_names is not None else str(i)

    import pandas as pd
    d = pd.concat(pandas_list)

    # draw the violin
    s = sns.violinplot(
        x="joint", y="err", hue="hue", data=d, color="gray", order=order, bw=0.4
    )

    # set the labels
    s.set_xticklabels(s.get_xticklabels(), rotation=30)
    ax.set_xlabel("")
    ax.set_ylabel(get_violin_ylabel(body_length, units))
    ax.set_ylim(ylim) if ylim is not None else None

