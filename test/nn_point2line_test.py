import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import time_delta
from time import time

np.random.seed(5)
X = np.random.randn(10000,3)
P = np.random.randn(10000,3)
Xi = np.arange(X.shape[0]).reshape(-1,2)
X[Xi[:,1]] *= 0.5
X[Xi[:,1]] += X[Xi[:,0]]

print("Brute force test of nn_point2line")
delta = time_delta(time())
dist, mp, nn = nn_point2line(X, Xi, P)
print("Time", next(delta)) 
print("Mean loss:", dist.mean())
mp -= P

fig = plt.figure()
ax = fig.add_subplot((111), projection='3d')
seg = np.hstack((X[Xi[:,0]], X[Xi[:,1]]-X[Xi[:,0]]))
x, y, z, u, v, w = zip(*seg)

ax.quiver(x, y, z, u, v, w)
ax.scatter(P[:,0],P[:,1],P[:,2], color='r')
ax.quiver(P[:,0],P[:,1],P[:,2],mp[:,0],mp[:,1],mp[:,2], color='g')

ax.set_xlim3d(-3, 3)
ax.set_ylim3d(-3, 3)
ax.set_zlim3d(-3, 3)
plt.show()