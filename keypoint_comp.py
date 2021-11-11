import os.path as path

import numpy as np
from scipy.spatial import Delaunay
import py7zr

import mhdm.lidar as lidar
import mhdm.spatial as spatial
from mhdm.utils import ifile
import mhdm.bitops as bitops
import mhdm.nbittree as nbittree

## Detect Keypoints

t = 0.03
tau = 1.0
tmp_file = 'data/keypoint_comp.bin'
arc_file = 'data/keypoint_comp.7z'
bpp_sum = 0
bpp_min = 1<<31
bpp_max = 0
count = 0
buffer = bitops.BitBuffer(tmp_file, 'wb')
bits_per_dim = [14,14,10]
word_length = sum(bits_per_dim)
dims = [3]

for file in ifile('./data/kitti_20110926_0005/*.bin'):
	count += 1
	X = np.fromfile(file, np.float32).reshape(-1, 4)[:,:-1]
	O = 1
	x = np.arange(len(X))
	m = np.zeros(len(X), dtype=bool)
	m[0] = True
	m[-1] = True
	K = X[m]
	A = np.roll(X, 1, axis=0) - X
	B = X - np.roll(X, -1, axis=0)
	
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
	
	
	Q, offset, scale = bitops.serialize(K, bits_per_dim, qtype=np.uint64, offset=None, scale=None)
	Q, permute, pattern = bitops.sort(Q, word_length, False, True)
	buffer.open(tmp_file, 'wb')
	for tree, payload in nbittree.encode(Q, dims, word_length, tree=buffer, yielding=True):
		pass
	buffer.close()
	
	with py7zr.SevenZipFile(arc_file, 'w') as z:
		z.write(tmp_file, path.basename(tmp_file))
	bpp = path.getsize(arc_file) * 8 / len(X)
	bpp_min = min(bpp_min, bpp)
	bpp_max = max(bpp_max, bpp)
	bpp_sum += bpp
	print("bpp mean:", bpp_sum / count, "\tmin:", bpp_min, "\tmax:", bpp_max)
print("Done")