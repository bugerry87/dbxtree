
import numpy as np
from scipy.spatial import Delaunay

import mhdm.lidar as lidar
import mhdm.spatial as spatial
#from mhdm.spatial import magnitude
#import matplotlib.pyplot as plt
from mayavi import mlab
import mhdm.viz as viz
from mhdm import aabbtree
import mhdm.bitops as bitops

## Detect Keypoints
X = np.fromfile('data/0000000000.bin', np.float32).reshape(-1, 4)[:,:-1]
print("X:", X.shape, X.mean(axis=0), np.abs(X).max(axis=0), "\n", X[:10])
t = 0.03
tau = 1.0

fig = viz.create_figure(bgcolor=(1.,1.,1.), size=(800, 600))
viz.vertices(X, X[:,-1], fig, colormap='Greys')
#input("Continue")

@mlab.animate(delay=10)
def animation():
	plot = None
	O = 1
	x = np.arange(len(X))
	m = np.zeros(len(X), dtype=bool)
	m[0] = True
	m[-1] = True
	K = X[m]
	A = np.roll(X, 1, axis=0) - X
	B = X - np.roll(X, -1, axis=0)
	count = 0
	
	while np.any(O > t**2):
		Y = [np.interp(x, x[m], K[:,i]) for i in range(K.shape[-1])]
		Y = np.vstack(Y).T
		M = X - Y
		O = spatial.magnitude(M)
		D = spatial.dot(A, B)[:,None]
		i = np.nonzero(m)[0]
		for a,b in zip(i[:-1], i[1:]):
			if a+1 >= b or np.all(O[a+1:b] <= t**2): continue
			am = np.argmax(O[a+1:b] * (1 - tau) - D[a+1:b] * tau)
			m[a+1+am] = True
		K = X[m]
		plot = viz.lines(K, K[:,-1], fig, plot)
		fig.render()
		count += 1
		yield
	
	K.tofile('data/run_lidar.bin')
	print("Done in steps:", count)
	print("Evaluating...")
	bits_per_dim = [14,14,10]
	Q, offset, scale = bitops.serialize(K, bits_per_dim, qtype=np.uint64, offset=None, scale=None)
	Q = bitops.realize(Q, bits_per_dim, offset, scale, xtype=np.float32)

	P = lidar.xyz2uvd(Q[:,(1,0,2)])
	P[:,(0,1)] *= (100, 200)
	Ti = Delaunay(P[:,(0,1)]).simplices
	#n = spatial.face_normals(P[Ti], True)
	#print(n, n.max(axis=0), n.min(axis=0))
	#Ti = Ti[np.abs(n[:,0]) > 0.1]
	O, Y, nn = aabbtree.query(Q, Ti, X)
	O = np.sqrt(O)

	print("K:", K.shape)
	print("Y:", Y.shape)
	print("O:", O.shape, O.mean(), O.min(), O.max())
	print('PSNR@1.0:', lidar.psnr(X, Y), 'dB')
	print('ACC@0.03:', 100 * float(np.sum(O<0.03)) / len(O), '%')
	print('CD:', O.mean(), 'm')

animator = animation()
viz.show_figure()
