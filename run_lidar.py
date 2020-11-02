
from pykitti.utils import yield_velo_scans
import numpy as np
import mhdm.lidar as lidar
from mhdm.utils import ifile
from mhdm.spatial import magnitude
import matplotlib.pyplot as plt
import mhdm.viz as viz

files = ifile('data/kitti_20110926_0005/*.bin')    
frame = yield_velo_scans(files)

## Detect Keypoints
X = next(frame)[:,:-1]
print("X:", X.shape, X.mean(axis=0), np.abs(X).max(axis=0), "\n", X[:10])
K, m = lidar.dot_keypoints(X, m=0.125)
x = np.arange(len(X))


## Fix Keypoints
Y = [np.interp(x, x[m], K[:,i]) for i in range(K.shape[-1])]
Y = np.vstack(Y).T
M = X - Y
O = magnitude(M, True)
m |= (O > 0.03).flatten()
K = X[m]
Y = [np.interp(x, x[m], K[:,i]) for i in range(K.shape[-1])]
Y = np.vstack(Y).T
M = X - Y

#'''
## Regress Offset
slices = np.nonzero(m)[0]
coefs=[]
for i, j in zip(slices[:-1], slices[1:]):
	if j-i > 2:
		coefs.append(np.hstack([np.polyfit(x[i:j], M[i:j,d], 2) for d in range(M.shape[-1])]))
	else:
		coefs.append(np.zeros(9))
coefs.append(np.zeros(9))
coefs = np.vstack(coefs).reshape(-1,3,3)

for i, (j, k) in enumerate(zip(slices[:-1], slices[1:])):
	Y[j:k,0] += np.poly1d(coefs[i,0])(x[j:k])
	Y[j:k,1] += np.poly1d(coefs[i,1])(x[j:k])
	Y[j:k,2] += np.poly1d(coefs[i,2])(x[j:k])

M = X - Y
#'''
O = magnitude(M, True)

print("K:", K.shape, K.mean(axis=0), np.abs(K).max(axis=0), "\n", K[:len(K)//10])
print("Y:", Y.shape, Y.mean(axis=0), np.abs(Y).max(axis=0), "\n", Y[:len(Y)//10])
print("O:", O.shape, O.mean(), O[m].min(), O.max(), "\n", O[::len(O)//10])
print('PSNR@100:', lidar.psnr(X, Y))
print('ACC@0.03:', 100 * float(np.sum(O<0.03)) / len(O))

plt.scatter(x[~m], O[~m], 0.1, marker='.', label='offset')
plt.plot((x[0],x[-1]), (0.03, 0.03), 'r--', label='sensor noise')
plt.show()

fig = viz.create_figure(bgcolor=(1.,1.,1.), size=(800, 600))
viz.vertices(Y, O.flatten(), fig, colormap='binary')
#viz.vertices(X, X[:,-1], fig, colormap='Blues')
#viz.vertices(Y, Y[:,-1], fig, colormap='Reds')
viz.show_figure()