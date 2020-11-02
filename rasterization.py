import numpy as np
from pykitti.utils import yield_velo_scans
import mhdm.spatial as spatial
import mhdm.lidar as lidar
import mhdm.aabbtree as aabbtree
from mhdm.utils import ifile
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import mhdm.viz as viz

if __name__ == '__main__':
	files = ifile('data/kitti_20110926_0005/*.bin')    
	frame = yield_velo_scans(files)
	X = next(frame)[:,:-1]
	U = lidar.xyz2uvd(X, z_off=-0.13, d_off=0.03, mode='cone')
	Ti = Delaunay(U[:,:2]).simplices
	fN = spatial.face_normals(X[Ti], True)
	vN = spatial.vec_normals(fN, Ti, True)
	m = spatial.mask_planar(vN, fN, Ti, 0.984375)
	Ti = Delaunay(U[m,:2]).simplices
	M = X[m]

	print('Query model size:', len(M))
	L, Y, nn = aabbtree.query(M, Ti, X, jobs=8)
	L = np.sqrt(L)
	m |= (L > 0.03).flatten()
	Y[m] = X[m]
	L = spatial.magnitude(X - Y, True)
	
	print("Points:", m.sum())
	print("PSNR@100m:", lidar.psnr(X, Y))
	print("ACC@0.03m:", 100 * float(np.sum(L<0.03)) / len(L))
	
	x = np.arange(len(L))
	plt.scatter(x, L, 0.1, marker='.', label='offset')
	plt.plot((x[0],x[-1]), (0.03, 0.03), 'r--', label='sensor noise')
	plt.show()

	fig = viz.create_figure(bgcolor=(1.,1.,1.), size=(800, 600))
	viz.vertices(Y, L.flatten(), fig, colormap='binary')
	#viz.vertices(Y[m], Y[m,-1], fig, colormap='Reds')
	viz.show_figure()