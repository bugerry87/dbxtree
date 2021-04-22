
## Installed
import numpy as np

## Local
import mhdm.bitops as bitops
import mhdm.range_coder as range_coder

dim = 1
bits_per_dim = [16,16,16]
word_length = sum(bits_per_dim)
tree_depth = word_length // dim

tree = range_coder.RangeEncoder('data/2bitTree_range.flg.bin')
payload = bitops.BitBuffer('data/2bitTree_range.pyl.bin', 'wb')
X = np.fromfile('data/0000000000.bin', dtype=np.float32).reshape(-1,4)[:,:3]
X = bitops.serialize(X, bits_per_dim, scale=[196,196,196], qtype=np.uint64)[0]
X = bitops.sort(X, word_length, False, True)[0]
layers = bitops.tokenize(X, dim, tree_depth+1)
mask = np.ones(len(X), bool)
bits = np.zeros(len(X), int)
acc = 0
count = 0

print('encoding')
for i, (X0, X1) in enumerate(zip(layers[:-1], layers[1:])):
	uids, idx, counts = np.unique(X0[mask], return_inverse=True, return_counts=True)
	flags = bitops.encode(X1[mask], idx, dim, ftype=np.uint8, htype=int)[0]
	#mask[mask] = counts[idx] > 1
	#flags[counts==1] = 0
	#bits[mask] = max(word_length - (i+1)*dim, 0)
	layer = np.full(uids.shape, i/tree_depth)
	edge = (1<<i)-1
	pos = uids.astype(np.float64) - edge*0.5
	if edge:
		pos /= (1<<word_length)-1
	probs = np.vstack([
		#layer,
		0.5 - pos,
		0.5 + pos,
		1.0 - layer
	]).T
	cdf = range_coder.prob2cdf(probs, floor=0.001)
	tree.updates(flags-1, cdf)
	hit = probs[range(len(flags)),flags-1]
	print('Layer:', i, 'Acc:', hit.mean())
	acc += sum(hit)
	count += len(hit)
print('Final Acc', acc/count)

tree.finalize()
for x, b in zip(X, bits):
	payload.write(x, b)