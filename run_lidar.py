
from pykitti.utils import yield_velo_scans
import numpy as np
import mhdm.lidar as lidar
from mhdm.utils import ifile

files = ifile('data/kitti_20110926_0005/*.bin')    
frame = yield_velo_scans(files)

X = next(frame)
print("X:", X.shape, X.mean(axis=0), np.abs(X).max(axis=0), "\n", X[:10])
O, m = lidar.mask_keypoints_xyz(X[:,:3], m=1)
K = O[m]
O = O[~m]
print("K:", K.shape, K.mean(axis=0), np.abs(K).max(axis=0), "\n", K[:10])
print("O:", O.shape, O.mean(axis=0), np.abs(O).max(axis=0), "\n", O[:10])
