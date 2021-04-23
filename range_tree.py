
## Installed
import numpy as np

## Local
import mhdm.bitops as bitops
import mhdm.viz as viz

dim = 1
bits_per_dim = [16,16,16]
word_length = sum(bits_per_dim)
tree_depth = word_length // dim

tree = bitops.BitBuffer('data/NbitTree_range.flg.bin', 'wb')
X = np.fromfile('data/0000000000.bin', dtype=np.float32).reshape(-1,4)[:,:3]
X, offset, scale = bitops.serialize(X, bits_per_dim, scale=[196,196,196], qtype=np.uint64)
X, p, m = bitops.sort(X, word_length, True)
split = 24
for x in X[::len(X)//10]:
	print("{:0>16}".format(hex(x)[2:]))
layers = bitops.tokenize(X, dim, tree_depth+1)[:-split]
acc = 0
count = 0

print('encoding')
for i, (X0, X1) in enumerate(zip(layers[:-1], layers[1:])):
	uids, idx, counts = np.unique(X0, return_inverse=True, return_counts=True)
	flags, hist = bitops.encode(X1, idx, dim, ftype=np.uint8, htype=int)
	for val, count in zip(hist.T[1], counts):
		bits = max(int(count).bit_length(), 1)
		tree.write(val, bits)

print('finalize')
payload = X[...,None] >> np.arange(0, split*dim, 8).astype(X.dtype) & ((1<<8)-1)
payload.T.astype(np.uint8).tofile('data/NbitTree_range.pyl.bin')

print('decode')
tree.open('data/NbitTree_range.flg.bin', 'rb')
counts = [len(X)]
bits = [int(len(X)).bit_length()]
Y = np.zeros([1], dtype=X.dtype)
for layer in range(split):
	right = np.array([tree.read(b) for b in bits])
	hist = np.vstack([counts - right, right]).T
	counts = hist.flatten()
	counts = counts[counts>0]
	bits = [max(int(count).bit_length(), 1) for count in counts]
	i, y = np.where(hist)
	Y <<= dim
	Y = y.astype(Y.dtype) + Y[i]

print('finalize')
Y <<= split 
Y = (X[...,None] & ((1<<split)-1) + Y.repeat(counts)[...,None]).flatten()
print(Y)
for y in Y[::len(Y)//10]:
	print("{:0>16}".format(hex(y)[2:]))
Y = bitops.permute(Y, p)
Y = bitops.deserialize(Y, bits_per_dim, qtype=X.dtype)
Y = bitops.realization(Y, offset, scale, np.float32)

print('visualize')
fig = viz.create_figure()
viz.vertices(Y, Y[:,2], fig)
viz.show_figure()