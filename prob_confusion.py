
## Installed
import numpy as np
import matplotlib.pyplot as plt

## Local
import mhdm.bitops as bitops
from mhdm.utils import ifile


files = ifile('data/kitti_20110926_0005/*.bin')
cm = np.zeros([48,48], int)
for f in files:
	print(f)
	X = np.fromfile(f, dtype=np.float32).reshape(-1,4)[:,:3]
	X, offset, scale = bitops.serialize(X, [16,16,16], np.uint64, scale=[196,196,196])
	X, p, m = bitops.sort(X, 48, True, True)
	cm[np.arange(48), p] += 1

m = np.zeros(48, bool)
P = list()
for p in cm.copy():
	p[m] = -1
	i = p.argmax()
	P.append(i)
	m[i] = True
print(P)
plt.matshow(cm)
plt.show()
cm[np.arange(48), P] = cm.max()
plt.matshow(cm)
plt.show()