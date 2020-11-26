import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation
import pandas as pd
import cv2
import matplotlib.image as mpimg


idx_LF_body_coxa = 0
idx_LF_coxa_femur = 1
idx_LF_femur_tibia = 2
idx_LF_tibia_tarsus = 3
idx_LF_tarsal_claw = 4
idx_LM_body_coxa = 5
idx_LM_coxa_femur = 6
idx_LM_femur_tibia = 7
idx_LM_tibia_tarsus = 8
idx_LM_tarsal_claw = 9
idx_LH_body_coxa = 10
idx_LH_coxa_femur = 11
idx_LH_femur_tibia = 12
idx_LH_tibia_tarsus = 13
idx_LH_tarsal_claw = 14
idx_RF_body_coxa = 15
idx_RF_coxa_femur = 16
idx_RF_femur_tibia = 17
idx_RF_tibia_tarsus = 18
idx_RF_tarsal_claw = 19
idx_RM_body_coxa = 20
idx_RM_coxa_femur = 21
idx_RM_femur_tibia = 22
idx_RM_tibia_tarsus = 23
idx_RM_tarsal_claw = 24
idx_RH_body_coxa = 25
idx_RH_coxa_femur = 26
idx_RH_femur_tibia = 27
idx_RH_tibia_tarsus = 28
idx_RH_tarsal_claw = 29
idx_L_antenna = 30
idx_R_antenna = 31
idx_L_eye = 32
idx_R_eye = 33
idx_L_haltere = 34
idx_R_haltere = 35
idx_L_wing = 36
idx_R_wing = 37
idx_proboscis = 38
idx_neck = 39
idx_genitalia = 40

home_dir = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/VV'
points3d = np.load('{}/points3d.npy'.format(home_dir))
points3d_names_id = np.load('{}/points3d_names_id.npy'.format(home_dir))


x = points3d[:,:,0]
y= points3d[:,:,1]
z= points3d[:,:,2]

# a = np.random.rand(2000, 3)*10
# t = np.array([np.ones(100)*i for i in range(20)]).flatten()
# all_data = pd.DataFrame({"x" : x, "y" : y, "z" : z})

def update_graph(num):
    graph_FL.set_data (x[num,:5], y[num,:5])
    graph_FL.set_3d_properties(z[num,:5])

    graph_ML.set_data (x[num,5:10], y[num,5:10])
    graph_ML.set_3d_properties(z[num,5:10])
    
    graph_HL.set_data (x[num,10:15], y[num,10:15])
    graph_HL.set_3d_properties(z[num,10:15])

    graph_FR.set_data(x[num, 15:20], y[num, 15:20])
    graph_FR.set_3d_properties(z[num, 15:20])

    graph_MR.set_data(x[num, 20:25], y[num, 20:25])
    graph_MR.set_3d_properties(z[num, 20:25])

    graph_HR.set_data(x[num, 25:30], y[num, 25:30])
    graph_HR.set_3d_properties(z[num, 25:30])

    graph_Lhead.set_data(np.hstack((x[num,38:40],x[num,32],x[num,30])), np.hstack((y[num,38:40],y[num,32],y[num,30])))
    graph_Lhead.set_3d_properties(np.hstack((z[num,38:40],z[num,32],z[num,30])))

    graph_Rhead.set_data(np.hstack((x[num,38:40],x[num,33],x[num,31])), np.hstack((y[num,38:40],y[num,33],y[num,31])))
    graph_Rhead.set_3d_properties(np.hstack((z[num,38:40],z[num,33],z[num,31])))

    graph_back.set_data(np.hstack((x[num,idx_L_wing],x[num,idx_L_haltere],x[num,idx_R_haltere],x[num,idx_R_wing])), np.hstack((y[num,idx_L_wing],y[num,idx_L_haltere],y[num,idx_R_haltere],y[num,idx_R_wing])))
    graph_back.set_3d_properties(np.hstack((z[num,idx_L_wing],z[num,idx_L_haltere],z[num,idx_R_haltere],z[num,idx_R_wing])))

    title.set_text('points3d, frame={}'.format(num+1))

    try:
        img_name_LV = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/LV/3_PG_4_LV_{:05d}.tiff'.format(points3d_names_id[num])
        title_LV.set_text('LV, frame={}'.format(num + 1))
        img_LV = mpimg.imread(img_name_LV)
        im_LV.set_array(img_LV)

        img_name_RV = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/RV/3_PG_4_RV_{:05d}.tiff'.format(points3d_names_id[num])
        title_RV.set_text('RV, frame={}'.format(num + 1))
        img_RV = mpimg.imread(img_name_RV)
        im_RV.set_array(img_RV)

        img_name_VV = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/VV/3_PG_4_VV_{:05d}.tiff'.format(points3d_names_id[num])
        title_VV.set_text('VV, frame={}'.format(num + 1))
        img_VV = mpimg.imread(img_name_VV)
        im_VV.set_array(img_VV)
    except:
        pass

    return title, graph_FL, graph_ML, graph_HL, graph_FR, graph_MR, graph_HR, graph_Lhead, im_LV, title_LV, im_RV, title_RV, im_VV, title_VV,

fig = plt.figure(figsize=(16, 12))

from matplotlib import gridspec
gs = gridspec.GridSpec(3, 4)

ax = fig.add_subplot(gs[:,0:3], projection='3d')
title = ax.set_title('points3d')

graph_FL, = ax.plot(x[0,:5], y[0,:5], z[0,:5], linestyle="-", marker="o", color='k')
graph_ML, = ax.plot(x[0,5:10], y[0,5:10], z[0,5:10], linestyle="-", marker="o", color='k')
graph_HL, = ax.plot(x[0,10:15], y[0,10:15], z[0,10:15], linestyle="-", marker="o", color='k')

graph_FR, = ax.plot(x[0,:5], y[0,:5], z[0,:5], linestyle="-", marker="o", color='k')
graph_MR, = ax.plot(x[0,5:10], y[0,5:10], z[0,5:10], linestyle="-", marker="o", color='k')
graph_HR, = ax.plot(x[0,10:15], y[0,10:15], z[0,10:15], linestyle="-", marker="o", color='k')

graph_Lhead, = ax.plot(np.hstack((x[0,38:40],x[0,32],x[0,30])), np.hstack((y[0,38:40],y[0,32],y[0,30])), np.hstack((z[0,38:40],z[0,32],z[0,30])), linestyle="-", marker="o", color='g')
graph_Rhead, = ax.plot(np.hstack((x[0,38:40],x[0,33],x[0,31])), np.hstack((y[0,38:40],y[0,33],y[0,31])), np.hstack((z[0,38:40],z[0,33],z[0,31])), linestyle="-", marker="o", color='g')

graph_back, = ax.plot(np.hstack((x[0,idx_L_wing],x[0,idx_L_haltere],x[0,idx_R_haltere],x[0,idx_R_wing])), np.hstack((y[0,idx_L_wing],y[0,idx_L_haltere],y[0,idx_R_haltere],y[0,idx_R_wing])), np.hstack((z[0,idx_L_wing],z[0,idx_L_haltere],z[0,idx_R_haltere],z[0,idx_R_wing])), linestyle="-", marker="o", color='m')

# graph_Rhead, = ax.plot(np.hstack((x[0,38:40],[x[0,33]],[x[0,31]])), np.hstack((y[0,38:40],[y[0,33]],[y[0,31]])), np.hstack((z[0,38:40],[z[0,33]],[z[0,31]])), linestyle="-", marker="o", color='g')

# graph_Rhead, = ax.plot(np.hstack((x[0,38:40],[x[0,32]])), np.hstack((y[0,38:40],[y[0,32]])), np.hstack((z[0,38:40],[z[0,32]])), linestyle="-", marker="o", color='g')



# ax_LV = fig.add_subplot(212)
img_name_RV = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/RV/3_PG_4_RV_{:05d}.tiff'.format(points3d_names_id[0])
img_RV = mpimg.imread(img_name_RV)
plt.subplot(gs[0,3:4])
plt.imshow(img_RV, cmap='gray')
title_RV = plt.title('RV')
im_RV = plt.imshow(img_RV, animated=True)

# ax_VV = fig.add_subplot(212)
img_name_VV = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/VV/3_PG_4_VV_{:05d}.tiff'.format(points3d_names_id[0])
img_VV = mpimg.imread(img_name_VV)
plt.subplot(gs[1,3:4])
plt.imshow(img_VV, cmap='gray')
title_VV = plt.title('VV')
im_VV = plt.imshow(img_VV, animated=True)

# ax_LV = fig.add_subplot(212)
img_name_LV = '/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/LV/3_PG_4_LV_{:05d}.tiff'.format(points3d_names_id[0])
img_LV = mpimg.imread(img_name_LV)
plt.subplot(gs[2,3:4])
plt.imshow(img_LV, cmap='gray')
title_LV = plt.title('LV')
im_LV = plt.imshow(img_LV, animated=True)

ani = matplotlib.animation.FuncAnimation(fig, update_graph, x.shape[0]-1,
                               interval=1, blit=False)
ax.set_xlim(x.min(),x.max())
ax.set_ylim(y.min(),y.max())
ax.set_zlim(z.min(),z.max())


plt.show()

# Set up formatting for the movie files
Writer = matplotlib.animation.writers['ffmpeg']
writer = Writer(fps=2, metadata=dict(artist='Me'))
ani.save('{}/points3d_animation.mp4'.format(home_dir), writer=writer)
