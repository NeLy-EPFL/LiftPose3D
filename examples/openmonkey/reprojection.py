# taken from https://github.com/OpenMonkeyStudio/OMS_Data
import numpy as np
import cv2
import sys, os, getopt
from numpy.linalg import inv
from scipy.io import loadmat
from collections import defaultdict
from matplotlib import pyplot as plt


def get_cameras(btch):
    with open(
        "/data/LiftFly3D/openmonkey/OMS_Dataset/Batch{}/intrinsic.txt".format(btch)
    ) as f:
        lines = f.readlines()
        cameras = {}
        for i in range(0, len(lines), 5):
            cam_line = lines[i]
            K_lines = lines[i + 1 : i + 4]
            ds = lines[i + 4].rstrip("\n")
            d = ds.split(" ")
            d1 = float(d[0])
            d2 = float(d[1])
            cam = cam_line.strip().split(" ")[1]
            K = np.reshape(
                np.array(
                    [float(f) for K_line in K_lines for f in K_line.strip().split(" ")]
                ),
                [3, 3],
            )
            cameras[cam] = {"K": K, "d1": d1, "d2": d2}

    # Extrinsics
    with open(
        "/data/LiftFly3D/openmonkey/OMS_Dataset/Batch{}/camera.txt".format(btch)
    ) as f:
        lines = f.readlines()
        for i in range(3, len(lines), 5):
            cam_line = lines[i]
            C_line = lines[i + 1]
            R_lines = lines[i + 2 : i + 5]
            cam = cam_line.strip().split(" ")[1]
            C = np.array([float(f) for f in C_line.strip().split(" ")])
            R = np.reshape(
                np.array(
                    [float(f) for R_line in R_lines for f in R_line.strip().split(" ")]
                ),
                [3, 3],
            )
            P = cameras[cam]["K"] @ (
                R @ (np.concatenate((np.identity(3), -np.reshape(C, [3, 1])), axis=1))
            )
            cameras[cam]["R"] = R
            cameras[cam]["C"] = C
            cameras[cam]["P"] = P

    return cameras


leaves = [0, 4, 6, 9, 11, 12]
#          0  1  2  3  4  5  6  7   8  9  10 11 12
parents = [1, 2, 7, 2, 3, 2, 5, -1, 7, 8, 7, 10, 7]
# Annotations
edges = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (2, 5),
    (5, 6),
    (2, 7),
    (7, 8),
    (8, 9),
    (7, 10),
    (10, 11),
    (7, 12),
]

bone_length = np.array(
    [
        0.20820029,
        0.24672319,
        0.20071098,
        0.67488184,
        0.20363887,
        0.68685812,
        0.8618808,
        0.52081985,
        0.37218406,
        0.52878267,
        0.37562456,
        0.37562456,
    ]
)


def normalize_bone_length(pose3d, edges, bone_length, parents, leaves):
    pose3d_normalized = pose3d.copy()
    for leaf in leaves:
        curr = leaf
        parent = parents[curr]
        history = list()
        while parent != -1:
            try:
                idx = edges.index((curr, parent))
            except:
                idx = edges.index((parent, curr))
            vec = pose3d_normalized[curr] - pose3d_normalized[parent]
            curr_length = np.linalg.norm(vec)
            offset = (vec / curr_length) * (bone_length[idx] - curr_length)

            history.append((curr, parent))
            for c, p in history:
                pose3d_normalized[c] += offset

            curr = parent
            parent = parents[curr]

    return pose3d_normalized


image_id_list = [
    [7380, 8100],
    [8480, 9280],
    [9980, 10700],
    [13160, 14000],
    [14980, 15620],
    [16320, 16960],
    [11520, 12280],
    [12360, 12580],
]


def is_img_id_valid(img_id):
    return any([s < img_id < e for (s, e) in image_id_list])


def is_good_data(btch, frame_id):
    return not (btch == "9" and not is_img_id_valid(frame_id))


def distort_point(cameras, u_x, u_y, cam):
    K = cameras[cam]["K"]
    d1 = cameras[cam]["d1"]
    d2 = cameras[cam]["d2"]

    invK = inv(K)
    z = np.array([u_x, u_y, 1])
    nx = invK.dot(z)

    x_dn = nx[0] * (
        1
        + d1 * (nx[0] * nx[0] + nx[1] * nx[1])
        + d2 * (nx[0] * nx[0] + nx[1] * nx[1]) * (nx[0] * nx[0] + nx[1] * nx[1])
    )
    y_dn = nx[1] * (
        1
        + d1 * (nx[0] * nx[0] + nx[1] * nx[1])
        + d2 * (nx[0] * nx[0] + nx[1] * nx[1]) * (nx[0] * nx[0] + nx[1] * nx[1])
    )

    z2 = np.array([x_dn, y_dn, 1])
    x_d = K.dot(z2)

    return np.array([x_d[0], x_d[1]])


def get_projection(cameras, cam, coords_3d):
    P = cameras[cam]["P"]
    u = P @ np.append(coords_3d, [1])
    u = u[0:2] / u[2]
    # proj = distort_point(cameras, u[0], u[1], cam)
    return u


def rotate_point(cameras, cam, coords_3d):
    R, C = cameras[cam]["R"], cameras[cam]["C"]
    M = R @ (np.concatenate((np.identity(3), -np.reshape(C, [3, 1])), axis=1))
    u = M @ np.append(coords_3d, [1])
    return u


def get_btch(btch):
    cameras = get_cameras(btch)

    Data = defaultdict(dict)
    annot_path = f"/data/LiftFly3D/openmonkey/OMS_Dataset/Batch{btch}/coords_3D.mat"
    annotations = loadmat(annot_path)
    param_path = f"/data/LiftFly3D/openmonkey/OMS_Dataset/Batch{btch}/crop_para.mat"
    parameters = loadmat(param_path)
    # list of img-id, cam_id, h_crop, w_crop, h, w
    image_id_list = parameters["crop"].transpose()[0]
    image_id_unique = np.unique(image_id_list, axis=0)  # unique image-ids

    for idx, image_id in enumerate(image_id_unique):
        # indices where image_id is seen
        q = np.where(image_id_list == image_id)[0]
        for i in range(q.shape[0]):
            frame, cmr, crop_x, crop_y, h, w = parameters["crop"][q[i]]
            if not is_good_data(btch, frame):
                continue
            k = (btch, frame, str(cmr))
            Data[k] = {
                "points3d": np.zeros((13, 3)),
                "points2d": np.zeros((13, 2)),
                "points2d_distort": np.zeros((13, 2)),
                "crop": np.array([crop_x, crop_y]),
            }

            # set 3d points
            ii = idx * 13
            for jt in range(13):
                Data[k]["points3d"][jt] = annotations["coords"][ii + jt, 1:4]
                            
            for jt in range(13):
                if Data[k]["points3d"][jt] is not None:
                    x, y = get_projection(cameras, str(cmr), Data[k]["points3d"][jt])
                    proj = distort_point(cameras, x, y, str(cmr))
                    Data[k]["points2d_distort"][jt] = proj
                    #Data[k]["points2d"][jt] = [x, y]
                    Data[k]["points3d"][jt] = rotate_point(
                        cameras, str(cmr), Data[k]["points3d"][jt]
                    )
            
            # bone-length normalize
            pt3d = Data[k]["points3d"]
            Data[k]["points3d"] = normalize_bone_length(
                pt3d, edges, bone_length, parents, leaves
            )
    
    # set points2d
    # taken from img_label_visualizer.py from https://github.com/OpenMonkeyStudio/OMS_Data
    data = loadmat(f"/data/LiftFly3D/openmonkey/OMS_Dataset/Data.mat")
    name = data["T"][0][0]["name"]
    label = data["T"][0][0]["data"]        
            
    # fill points2d
    for i, n in enumerate(name):
        n = n[0][0]
        batch_id, frame_id, cam_id = parse_img_name(n)
        k = (batch_id, frame_id, str(cam_id))
        if btch == batch_id  and k in Data.keys():
            Data[k]["points2d"] = get_pts2d(i, label)

    return Data, cameras

def parse_img_name(name):
    sp = name.split("_")
    batch_id = sp[0].replace("batch", "")
    frame_id = int(sp[1])
    cam_id = int(sp[2].replace(".jpg", ""))
    return batch_id, frame_id, str(cam_id)

def get_pts2d(i, label):
    jt_loc = np.zeros((13, 2))
    for j in range(13):
        jt_loc[j] = [label[i][5 + 2 * j], label[i][4 + 2 * j]]

    return jt_loc

if __name__ == "__main__":
    get_btch("9")

