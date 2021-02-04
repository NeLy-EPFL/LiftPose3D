import numpy as np
import pickle
from scipy.io import loadmat
import os
import cv2

# data = loadmat(data_folder + "Data.mat")
# name = data["T"][0][0]["name"]
# label = data["T"][0][0]["data"]
# name2idx = {n[0][0]: idx for (idx, n) in enumerate(name)}


# cameras = {c: {} for c in batch_w_3d}


def load_data(data_folder, batches):
    data = loadmat(data_folder + "Data.mat")
    name = data["T"][0][0]["name"]
    label = data["T"][0][0]["data"]
    batch_w_3d = batches
    cameras, cameras_copy = load_cameras(data_folder, batch_w_3d)

    cam_list = np.unique(
        np.concatenate([list(cameras[btch].keys()) for btch in batch_w_3d])
    ).tolist()
    pose_result = load_points(data_folder, batch_w_3d, name, cameras, cam_list, label)    
    pose_result = {k: linear(v, k) for (k, v) in pose_result.items()}
    

    return cameras_copy, pose_result


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


#          0  1  2  3  4  5  6  7   8  9  10 11 12



def bone_length_normalize(template, d, leaves, parents):
    pts3d = np.array(template)
    bone_length = np.zeros((len(edges)))
    for idx, edge in enumerate(edges):
        bone_length[idx] = np.linalg.norm(
            pts3d[:, edge[0]] - pts3d[:, edge[1]], axis=1
        ).mean()

    for idx, pts3d in enumerate(d):
        d[idx] = normalize_bone_length(
            d[idx], edges, bone_length, parents, leaves
        )

    return d


def linear(d, btch):
    d_linear = {
        "points2d": list(),
        "points3d": [],
        "cam_id": [],
        "img_id": [],
        "batch_id": [],
    }

    for cam_id in range(d["points2d"].shape[0]):
        for img_id in range(d["points2d"].shape[1]):
            if np.any(d["points2d"][cam_id, img_id] != 0) and np.any(
                d["points3d"][img_id] != 0
            ):
                d_linear["points2d"].append(d["points2d"][cam_id, img_id])                
                d_linear["points3d"].append(d["points3d"][img_id])
                d_linear["cam_id"].append(cam_id)
                d_linear["img_id"].append(img_id)
                d_linear["batch_id"].append(btch)

    return d_linear


def get_projection(cameras, cam, coords_3d):
    '''transforms 3d coordinates into 2d coordinates, using camera parameters inside cameras. 
    '''
    P = cameras[cam]["P"]
    u = P @ np.append(coords_3d, [1])
    u = u[0:2] / u[2]
    proj = u
    # proj = distort_point(u[0], u[1], cameras, cam)
    return proj


def load_points(data_folder, batch_w_3d, name, cameras, cam_list, label):
    n_joints = 13
    pose_result = dict()

    btch_9b_continue = np.array(
        [
            158,
            1125,
            2092,
            3059,
            4026,
            4993,
            5960,
            6927,
            7894,
            8861,
            9828,
            10795,
            11762,
            12729,
            13696,
            14663,
            15630,
            16597,
            17564,
            18531,
            19498,
            20465,
            21432,
            22399,
            23366,
            24333,
            25300,
            26267,
            27234,
            28201,
            29168,
            30135,
            31102,
            32069,
            33036,
            34003,
            34970,
            35937,
            36904,
            37871,
            38838,
            39805,
            40772,
            41739,
            42706,
            43673,
            44640,
            45607,
            46574,
            47541,
            48508,
            49475,
            50442,
            51409,
            52376,
            53343,
            54310,
            55277,
            56244,
            57211,
            58178,
            59145,
            5720,
            6400,
            6400,
            6700,
            6700,
            6700,
            6700,
            9800,
            9800,
            9800,
            21400,
        ]
    )
    btch_10_continue = np.array([12280, 14660, 14660, 72340, 87820])
    btch_11_continue = np.array([12280, 14660, 14660, 72340, 75860])

    for btch in batch_w_3d:
        print(f"processing batch {btch}")
        annotations = loadmat(data_folder + "Batch{}/coords_3D.mat".format(btch))
        parameters = loadmat(data_folder + "Batch{}/crop_para.mat".format(btch))

        #print("annotations", annotations["coords"].shape)
        max_num_images = get_max_n_images(name, btch)

        points2d = np.zeros(((len(cam_list)), max_num_images, n_joints, 2))
        points3d = np.zeros((max_num_images, n_joints, 3))

        for i, n in enumerate(name):
            n = n[0][0]
            batch_id, frame_id, cam_id = parse_img_name(n)
            cam_id = str(cam_id)
            if batch_id == btch and cam_id in cameras[btch]:
                dist = [cameras[btch][cam_id]["d1"], cameras[btch][cam_id]["d2"], 0, 0]
                intr = cameras[btch][cam_id]["K"]
                src = get_pts2d(i, n, label)
                points2d[cam_list.index(str(cam_id)), frame_id] = src
                #cv2.undistortPoints(
                #    np.expand_dims(src, 0).astype(np.float32),
                #    distCoeffs=np.array(dist).astype(np.float32),
                #    cameraMatrix=np.array(intr).astype(np.float32),
                #)

        # fill pts3d
        for frm in range((annotations["coords"].shape[0]) // 13 - 1):
            i = 0  # camera_id
            pt = parameters["crop"].transpose()[0]
            u = np.unique(pt, axis=0)
            q = np.where(pt == u[frm])
            frame_id = parameters["crop"][q[0][2 * i]][0]

            if btch == "9b" and frame_id in btch_9b_continue:
                continue
            if btch == "10" and frame_id in btch_10_continue:
                continue
            if btch == "11" and frame_id in btch_11_continue:
                continue

            ii = frm * 13
            for jt in range(n_joints):
                points3d[frame_id, jt] = annotations["coords"][ii + jt, 1:4]

            for cam_id in range(len(cam_list)):
                if (
                    str(cam_list[cam_id]) in cameras[btch]
                ):  # and (np.any(points2d[cam_id, frame_id] != 0)): #:
                    for jt in range(n_joints):
                        if btch == "9" or np.any(points2d[cam_id, frame_id] != 0):
                            continue
                        points2d[cam_id, frame_id, jt] = get_projection(
                            cameras[btch], str(cam_list[cam_id]), points3d[frame_id, jt]
                        )
            pose_result[btch] = {"points3d": points3d, "points2d": points2d}

    return pose_result


def load_cameras(data_folder, batch_w_3d):
    # load camera parameters for each batch with 3d data
    cameras = {c: {} for c in batch_w_3d}
    for btch in batch_w_3d:
        with open(data_folder + "Batch{}/intrinsic.txt".format(btch)) as f:
            lines = f.readlines()
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
                        [
                            float(f)
                            for K_line in K_lines
                            for f in K_line.strip().split(" ")
                        ]
                    ),
                    [3, 3],
                )
                cameras[btch][cam] = {"K": K, "d1": d1, "d2": d2}

        # Load Extrinsics
        with open(data_folder + "Batch{}/camera.txt".format(btch)) as f:
            lines = f.readlines()
            for i in range(3, len(lines), 5):
                cam_line = lines[i]
                C_line = lines[i + 1]
                R_lines = lines[i + 2 : i + 5]
                cam = cam_line.strip().split(" ")[1]
                C = np.array([float(f) for f in C_line.strip().split(" ")])
                R = np.reshape(
                    np.array(
                        [
                            float(f)
                            for R_line in R_lines
                            for f in R_line.strip().split(" ")
                        ]
                    ),
                    [3, 3],
                )
                P = cameras[btch][cam]["K"] @ (
                    R
                    @ (np.concatenate((np.identity(3), -np.reshape(C, [3, 1])), axis=1))
                )
                cameras[btch][cam]["R"] = R
                cameras[btch][cam]["C"] = C
                cameras[btch][cam]["P"] = P

    cam_list = np.unique(
        np.concatenate([list(cameras[btch].keys()) for btch in batch_w_3d])
    ).tolist()
    cameras_copy = {c: {} for c in batch_w_3d}
    for btch in batch_w_3d:
        for k, v in cameras[btch].items():
            cameras_copy[btch][cam_list.index(k)] = [
                v["R"],
                -1 * np.dot(v["R"], v["C"]),  # T = -R * C
                v["K"],
                (v["d1"], v["d2"]),
                np.arange(13),
            ]

    return cameras, cameras_copy


def get_pts2d(i, n, label):
    jt_loc = np.zeros((13, 2))
    for j in range(13):
        jt_loc[j] = [label[i][5 + 2 * j], label[i][4 + 2 * j]]

    return jt_loc


def parse_img_name(name):
    sp = name.split("_")
    batch_id = sp[0].replace("batch", "")
    frame_id = int(sp[1])
    cam_id = int(sp[2].replace(".jpg", ""))
    return batch_id, frame_id, cam_id


def constr_img_name(btch, frame, cmr):
    return "batch" + str(btch) + "_" + str(frame).zfill(9) + "_" + str(cmr) + ".jpg"


def get_max_n_images(name, btch):
    l = list()
    for n in name:
        n = n[0][0]
        batch_id, frame_id, cam_id = parse_img_name(n)
        if batch_id == btch:
            l.append(frame_id)

    return max(l) + 1


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
