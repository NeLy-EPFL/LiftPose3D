import numpy as np
import matplotlib.pyplot as plt

points3d = np.load('/media/mahdi/LaCie/Mahdi/data/clipped_NEW/fly_3_clipped/PG/4/VV/points3d.npy')

x = points3d[0,:,0]
y= points3d[0,:,1]
z= points3d[0,:,2]



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ax = plt.gca(projection="3d")
ax.scatter(x,y,z, c='r',s=30)

def connectpoints(x,y,z, p1,p2, color='k'):
    x1, x2 = x[p1], x[p2]
    y1, y2 = y[p1], y[p2]
    z1, z2 = z[p1], z[p2]
    plt.plot([x1,x2],[y1,y2], [z1,z2], '{}-'.format(color))

connectpoints(x,y,z, 0,1)
connectpoints(x,y,z, 1,2)
connectpoints(x,y,z, 2,3)
connectpoints(x,y,z, 3,4)

connectpoints(x,y,z, 5,6)
connectpoints(x,y,z, 6,7)
connectpoints(x,y,z, 7,8)
connectpoints(x,y,z, 8,9)

connectpoints(x,y,z, 10,11)
connectpoints(x,y,z, 11,12)
connectpoints(x,y,z, 12,13)
connectpoints(x,y,z, 13,14)

connectpoints(x,y,z, 15,16)
connectpoints(x,y,z, 16,17)
connectpoints(x,y,z, 17,18)
connectpoints(x,y,z, 18,19)

connectpoints(x,y,z, 20,21)
connectpoints(x,y,z, 21,22)
connectpoints(x,y,z, 22,23)
connectpoints(x,y,z, 23,24)

connectpoints(x,y,z, 25,26)
connectpoints(x,y,z, 26,27)
connectpoints(x,y,z, 27,28)
connectpoints(x,y,z, 28,29)

connectpoints(x,y,z, 39,30, color='b')
connectpoints(x,y,z, 30,32, color='b')

connectpoints(x,y,z, 39,31, color='b')
connectpoints(x,y,z, 31,33, color='b')

connectpoints(x,y,z, 39,38, color='g')

# connectpoints(x,y,z, 32,38, color='g')
#
# connectpoints(x,y,z, 33,38, color='g')
#
# connectpoints(x,y,z, 39,40)
# connectpoints(x,y,z, 40,36)
#
# connectpoints(x,y,z, 39,40)
# connectpoints(x,y,z, 40,37)

# ax.plot(x,y,z, color='r')

plt.show()




# plt.plot(x,y,z, 'ro')
#

#
# plt.axis('equal')
# plt.show()