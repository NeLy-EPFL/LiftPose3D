import cv2
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def get_pts2d(i, n, label):
    jt_loc = np.zeros((13,2))
    for j in range(13):
        jt_loc[j] = [label[i][5+2*j],label[i][4+2*j]]
        
    return jt_loc

def get_img_overlay(data_folder, i, name, label, joint_pairs, colors, points2d=None, cam_list=None, image=None):
    label = label[i]
    n = name[i][0][0]
    if image is None:
        img_name = data_folder + 'Images/{}'.format(n)
        image = cv2.imread(img_name)
    jt_loc = {}
    h = label[2]
    w = label[3]
    for j in range(13):
        _, frame_id, cam_id = parse_img_name(n)

        if points2d is None:
            jt_loc[j] = (label[5+2*j],label[4+2*j])
        else:
            jt_loc[j] = (int(points2d[cam_list.index(str(cam_id)), frame_id, j,0]), int(points2d[cam_list.index(str(cam_id)), frame_id, j, 1]))

    for j, (jt1, jt2) in enumerate(joint_pairs):
        pt1 = jt_loc[jt1]
        pt2 = jt_loc[jt2]
        cv2.line(image, pt1, pt2, colors[j], 3)
    return image, n



def get_img_overlay_matplotlib(ax, data_folder, i, name, label, joint_pairs, colors, points2d=None, cam_list=None, image=None):
    label = label[i]
    n = name[i][0][0]
    if image is None:
        img_name = data_folder + 'Images/{}'.format(n)
        image = cv2.imread(img_name)
    jt_loc = {}
    h = label[2]
    w = label[3]
    for j in range(13):
        _, frame_id, cam_id = parse_img_name(n)
        #print(n, frame_id, cam_id)
        if points2d is None:
            jt_loc[j] = (label[5+2*j],label[4+2*j])
        else:
            jt_loc[j] = (int(points2d[cam_list.index(str(cam_id)), frame_id, j,0]), int(points2d[cam_list.index(str(cam_id)), frame_id, j, 1]))
    
    
    ax.imshow(image)
    for j, (jt1, jt2) in enumerate(joint_pairs):
        pt1 = jt_loc[jt1]
        pt2 = jt_loc[jt2]
        #if pt1[0] < 1 or pt1[1] < 1 or pt2[0] < 1 or pt2[1] < 1 or pt1[0] > w or pt1[1] > h or pt2[0] > w or pt2[1] > h:
        #    continue
        #else:
        #cv2.line(image, pt1, pt2, colors[j], 3)
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=[c_/255 for c_ in colors[j]])
    return image, n



# batch(batch#)_(frame#)_(cameraID).jpg , Eg. batch7_000003120_18064162.jpg
def parse_img_name(name):
    sp = name.split('_')
    batch_id = sp[0].replace('batch', '')
    frame_id = int(sp[1])
    cam_id = int(sp[2].replace('.jpg', ''))
    return batch_id, frame_id, cam_id

def constr_img_name(btch, frame, cmr):
     return 'batch' + str(btch) + '_' + str(frame).zfill(9) + '_' + str(cmr) + '.jpg'

def get_max_n_images(name, btch):
    l = list()
    for n in name:
        n = n[0][0]
        batch_id, frame_id, cam_id = parse_img_name(n)
        if batch_id == btch:
            l.append(frame_id)

    return max(l) + 1


def distort_point(u_x, u_y, cameras, cam):
    K = cameras[cam]['K']
    d1 = cameras[cam]['d1']
    d2 = cameras[cam]['d2']

    invK = inv(K)
    z = np.array([u_x, u_y, 1])
    nx = invK.dot(z)

    x_dn = nx[0] * (1 + d1 * (nx[0] * nx[0] + nx[1] * nx[1]) + d2 * (nx[0] * nx[0] + nx[1] * nx[1]) * (
             nx[0] * nx[0] + nx[1] * nx[1]))
    y_dn = nx[1] * (1 + d1 * (nx[0] * nx[0] + nx[1] * nx[1]) + d2 * (nx[0] * nx[0] + nx[1] * nx[1]) * (
             nx[0] * nx[0] + nx[1] * nx[1]))

    z2 = np.array([x_dn, y_dn, 1])
    x_d = K.dot(z2)

    return np.array([x_d[0], x_d[1]])

def get_projection(cameras, cam, coords_3d):
    P = cameras[cam]['P']
    u = P @ np.append(coords_3d, [1])
    u = u[0:2] / u[2]
    proj = u
    #proj = distort_point(u[0], u[1], cameras, cam)
    return proj

def display_plot(I):
    fig = plt.figure(dpi=300)
    
    for i in range(8):
        sub1 = plt.subplot(2, 4, i+1)
        sub1.set_xticks(())
        sub1.set_yticks(())
        sub1.imshow(I[i])

    fig.tight_layout()
    
    
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