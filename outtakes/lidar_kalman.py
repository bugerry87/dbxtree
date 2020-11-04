
from pykitti.utils import yield_velo_scans
import numpy as np
import mhdm.lidar as lidar
from mhdm.utils import ifile
from mhdm.spatial import magnitude

files = ifile('data/kitti_20110926_0005/*.bin')    
frame = yield_velo_scans(files)
X = next(frame)[:,:-1]
#print("X:", X.shape, X.mean(axis=0), np.abs(X).max(axis=0), "\n", X[:10])
#X = np.vstack((np.arange(100), 0.2*np.arange(100)**1.2, np.zeros(100))).T
#X[:,-1] = 0
#X += np.random.randn(100,3)*0.03
Y, p = lidar.kalman_filter(X, order=2, dt=0.01)
x = np.arange(len(Y))

import mhdm.viz as viz
fig = viz.create_figure(size=(800, 600))
viz.lines(Y[:,:3], Y[:,2], fig, colormap='Blues')
viz.lines(p[:,:3], p[:,2], fig, colormap='Reds', line_width=1.0)
viz.vertices(X, X[:,0], fig, colormap='Greens')
viz.show_figure()


"""
O = magnitude(X - Y, True)

print("Y:", Y.shape, Y.mean(axis=0), np.abs(Y).max(axis=0), "\n", Y[:10])
print("O:", O.shape, O.mean(), O.min(), O.max(), "\n", O[:10])
print('PSNR:', lidar.psnr(X, Y, 100))
print('Underate Sensor Noise:', 100 * float(np.sum(O<0.03)) / len(O))

import matplotlib.pyplot as plt
plt.scatter(x, O, 0.1, marker='.', label='offset')
plt.plot((x[0],x[-1]), (0.03, 0.03), 'r--', label='sensor noise')
plt.show()

import mhdm.viz as viz
fig = viz.create_figure(bgcolor=(1.,1.,1.), size=(800, 600))
viz.vertices(Y, O.flatten(), fig, colormap='binary')
viz.show_figure()
"""