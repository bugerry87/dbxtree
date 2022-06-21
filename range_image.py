import numpy as np
import mhdm.lidar as lidar
from PIL import Image as im

X = np.fromfile('data/kitti_20110926_0005/0000000000.bin', np.float32).reshape(-1, 4)
edges = (X[:-1,1] < 0) & (X[1:,1] >= 0)
row = np.concatenate(([0], np.cumsum(edges).astype(int)))
counts = np.unique(row, return_counts = True)[-1]
uvd = lidar.xyz2uvd(X[:,:3])
u, v, d = uvd.T
n_cols = 1024 # int(counts.max() + 1)
azimuth = (n_cols - 1) * (u + np.pi) / (2 * np.pi)
col = np.round(azimuth).astype(int)

N = np.zeros((len(counts), n_cols, 2), float)
np.add.at(N[...,0], (row,col), 1)
D = np.round(((1<<8) - 1) * d / d.max())
np.add.at(N[...,1], (row,col), D)
N[...,1] = np.divide(N[...,1], N[...,0], where=N[...,0] != 0)

azimuth = azimuth - col
azimuth = np.round(127 * azimuth / azimuth.max()).astype(np.int8)
azimuth.tofile('/mnt/d/GANDM/data/azimuth.bin')

altitude = row.max() * (1 - (v - v.min()) / (v.max() - v.min()))
altitude = altitude - row
altitude = np.round(127 * altitude / altitude.max()).astype(np.int8)

altitude.tofile('/mnt/d/GANDM/data/altitude.bin')

d -= D * d.max() / ((1<<8) - 1)
print(np.abs(d).mean())
d = np.round(((1<<8) - 1) * d / d.max()).astype(np.int8)
d.tofile('/mnt/d/GANDM/data/depth.bin')

N = im.fromarray(N.astype(np.uint8))
N.save('/mnt/d/GANDM/data/range_image.png')