import numpy as np
import mhdm.lidar as lidar
from PIL import Image as im
#import mhdm.viz as viz

X = np.fromfile('/mnt/d/GANDM/data/kitti_20110926_0005/0000000000.bin', np.float32).reshape(-1, 4)
edges = (X[:-1,1] < 0) & (X[1:,1] >= 0)
row = np.concatenate(([0], np.cumsum(edges).astype(int)))
counts = np.unique(row, return_counts = True)[-1]
uvd = lidar.xyz2uvd(X[:,:3])
U = np.copy(uvd)
u, v, d = uvd.T
n_cols = 1024 # int(counts.max() + 1)
azimuth = (n_cols - 1) * (u + np.pi) / (2 * np.pi)
col = np.round(azimuth).astype(int)

N = np.zeros((len(counts), n_cols, 2), np.float32)
np.add.at(N[...,0], (row,col), 1)
d_max = d.max()
np.add.at(N[...,1], (row,col), d)
N[...,1] = 255 * np.divide(N[...,1], N[...,0], where=N[...,0] != 0) / d_max
N = np.round(N)

azimuth = azimuth - col
azi_max = np.abs(azimuth).max()
azimuth = np.round((2**15-1) * azimuth / azi_max).astype(np.int16)
azimuth.tofile('/mnt/d/GANDM/data/azimuth.bin')

v_min = v.min()
v_max = v.max()
altitude = row.max() * (1 - (v - v_min) / (v_max - v_min)) - row
alt_max = np.abs(altitude).max()
altitude = np.round((2**15-1) * altitude / alt_max).astype(np.int16)

altitude.tofile('/mnt/d/GANDM/data/altitude.bin')

D = N[...,1][(row,col)] * d_max / 255
d = D - d
delta_max = np.abs(d).max()
d = np.round((2**15-1) * d / delta_max).astype(np.int16)
d.tofile('/mnt/d/GANDM/data/depth.bin')

I = im.fromarray(N.astype(np.uint8))
I.save('/mnt/d/GANDM/data/range_image.png')

# re const

uvd[...,0] = (azimuth.astype(np.float32) * azi_max / (2**15-1) + col) * 2 * np.pi / (n_cols - 1) - np.pi
uvd[...,1] = (1 - (altitude.astype(np.float32) * alt_max / (2**15-1) + row) / row.max()) * (v_max - v_min) + v_min
uvd[...,2] = D - d.astype(np.float32) * delta_max / (2**15-1)

xyz = lidar.uvd2xyz(uvd).astype(np.float32)
xyz.tofile('/mnt/d/GANDM/data/range_image_xyz.bin')

x = lidar.uvd2xyz(lidar.xyz2uvd(X[...,:3])).astype(np.float32)
x.tofile('/mnt/d/GANDM/data/range_image_gt.bin')
print(np.sqrt(((x - xyz)**2).sum(-1)).mean())

#fig = viz.create_figure()
#viz.vertices(xyz, 'r', fig)
#viz.show()