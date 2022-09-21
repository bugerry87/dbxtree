
## Installed
import tensorflow as tf

## Local
if __name__ != '__main__':
	from . import bitops
else:
	import bitops

tokens = tf.range(8)
tokens = bitops.right_shift(tokens[...,None], tf.range(3))
tokens = bitops.bitwise_and(tokens, 1)
tokens = 0.5 - tf.cast(tokens, tf.float32)*1.0

def encode(X, nodes, inv, bbox, radius, means=0):
	n = tf.math.reduce_max(inv) + 1
	intervals = bbox / radius
	i = tf.argsort(intervals, axis=-1, direction='DESCENDING')
	bbox = tf.gather(bbox, i, batch_dims=1)
	means = tf.gather(means, i, batch_dims=1)
	i = tf.gather(i, inv)
	X = tf.gather(X, i, batch_dims=1)

	big = tf.cast(bbox > radius, X.dtype)
	means = tf.gather(means, inv)
	dims = tf.math.reduce_sum(big, axis=-1, keepdims=True)
	keep = tf.math.reduce_any(tf.math.abs(X) - means > radius, axis=-1, keepdims=True)
	keep = tf.cast(keep, X.dtype)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = tf.gather(keep, inv)

	sign = tf.cast(X >= means, X.dtype)
	bbox = tf.gather(bbox, inv)
	big = tf.gather(big, inv)
	bbox = (bbox * (1.0 - sign * 2.0) * big * mask + means) * 0.5
	X += bbox

	sign = tf.cast(sign, nodes.dtype)
	dims = tf.cast(dims, nodes.dtype)
	dims = tf.gather(dims, inv)
	bits = bitops.left_shift(sign, tf.range(3, dtype=tf.int64))
	bits = tf.math.reduce_sum(bits, axis=-1, keepdims=True)
	bits = bitops.bitwise_and(bits, bitops.left_shift(1,dims)-1)
	
	nodes = nodes[...,None]
	nodes = bitops.left_shift(nodes, dims)
	nodes = bitops.bitwise_or(nodes, bits)
	nodes = nodes[...,0]

	flags = tf.one_hot(bits[...,0], 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * tf.cast(keep, flags.dtype)
	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	i = tf.argsort(nodes)
	mask = tf.gather(mask[...,0], i)
	i = tf.gather(i, tf.where(mask)[...,0])
	nodes = tf.gather(nodes, i)
	bbox = tf.gather(bbox, i)
	X = tf.gather(X, i)
	inv = tf.unique(nodes)[-1]
	n = tf.math.reduce_max(inv) + 1
	
	bbox = tf.math.unsorted_segment_max(tf.abs(bbox), inv, n)
	means = tf.math.unsorted_segment_mean(X, inv, n)
	return X, nodes, bbox, flags, dims, means

def decode(flags, bbox, radius, X, keep, means=0, batch_dims=0):
	signs = bitops.right_shift(flags[...,None], tf.range(flags.shape[-1]))
	signs = bitops.bitwise_and(signs, 1)
	intervals = bbox / radius
	i = tf.argsort(intervals, axis=-1, direction='DESCENDING')
	X = tf.gather_nd(X, i, batch_dims+1)
	offsets = tf.cast(signs[...,None], tokens.dtype) * tokens[None,...] * (bbox + means) * tf.cast(bbox > radius, tokens.dtype)
	i = tf.where(flags == 0)
	keep = tf.concat([keep, tf.gather(X, i[...,0])], axis=-2)
	i = tf.where(signs)
	X = tf.gather_nd(X, i[...,-2], batch_dims) - tf.gather_nd(offsets, i)
	X = tf.concat([X, keep], axis=-2)
	return X, keep, bbox*0.5


if __name__ == '__main__':
	import numpy as np
	X = np.fromfile('../../data/0000000000.bin', dtype=np.float32).reshape(-1, 4)[:,:3]
	bbox0 = np.max(np.abs(X), axis=0)
	radius = np.zeros([1,3], dtype=np.float32) + 0.0015
	means = np.zeros([1,3], dtype=np.float32)
	nodes = np.zeros_like(X[:,0], dtype=np.int64)
	pos = np.zeros([1,3], dtype=np.float32)
	bbox = bbox0[None,...]
	X, nodes, pos, candidates, bbox, flags, uids, dims, means = encode(X, nodes, pos, bbox, radius, means)
	print(X, nodes, pos, candidates, bbox, flags, uids, dims, means)