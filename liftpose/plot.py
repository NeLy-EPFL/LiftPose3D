from matplotlib.legend_handler import HandlerTuple
import networkx as nx
import numpy as np


def plot_pose_3d(
    ax,
    tar,
    bones,
    normalize=True,
    pred=None,
    limb_id=None,
    colors=None,
    good_keypts_tar=None,
    good_keypts_pred=None
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
        good_keypts=good_keypts_tar
    )
    if pred is not None:
        plot_3d_graph(
            G, 
            pred, 
            ax, 
            color_edge=edge_colors, 
            style="--", 
            good_keypts=good_keypts_pred
        )

    #### this bit is to make special legend
    pts = np.nanmean(tar,axis=0)
    (p1,) = ax.plot(pts[[0]], pts[[1]], pts[[2]], "r-")
    # (p2,) = ax.plot(pts, pts, pts, "b-")
    (p3,) = ax.plot(pts[[0]], pts[[1]], pts[[2]], "r--", dashes=(2, 2))
    # (p4,) = ax.plot(pts, pts, pts, "b--", dashes=(2, 2))
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
        G,
        tar,
        ax,
        color_edge=edge_colors,
        good_keypts=good_keypts,
    )


def plot_3d_graph(
        G, 
        pos, 
        ax, 
        color_edge=None, 
        style=None, 
        good_keypts=None
        ):
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


def plot_2d_graph(
        G, 
        pos, 
        ax, 
        color_edge=None, 
        style=None, 
        good_keypts=None
        ):
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

        u = np.array((pos[j[0],0], pos[j[1],0]))
        v = np.array((pos[j[0],1], pos[j[1],1]))
        
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
    # ax.xscale('log')
    # ax.xlim([0, 100])
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epochs")
    ax.legend()
    # ax.savefig("training_error.svg")


def read_log_train(out_dir):
    import glob
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
