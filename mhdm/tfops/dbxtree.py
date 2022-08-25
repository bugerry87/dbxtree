
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


def encode(X, nodes, pos, bbox, radius):
	uids, inv = tf.unique(nodes)
	n = tf.reduce_max(inv)+1

	big = tf.cast(bbox > radius, tf.int32)
	dims = tf.math.reduce_sum(big, axis=-1)
	keep = tf.math.reduce_any(tf.math.abs(X) > radius, axis=-1, keepdims=True)
	keep = tf.cast(keep, tf.int32)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = tf.gather(keep, inv)

	sign = tf.cast(X>=0, tf.int32)
	bits = bitops.left_shift(sign, tf.range(3))
	bits = tf.math.reduce_sum(bits, axis=-1, keepdims=True)
	bits = bitops.bitwise_and(bits, bitops.left_shift(1,dims)-1)
	
	nodes = nodes[...,None]
	nodes = bitops.left_shift(nodes, tf.cast(dims, nodes.dtype))
	nodes = bitops.bitwise_or(nodes, tf.cast(bits, nodes.dtype))
	
	big = tf.cast(big, tf.float32)
	pivots = pos - bbox * big * tokens[None,...]

	flags = tf.one_hot(bits[...,0], 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * keep

	i = tf.where(tf.reshape(flags, (-1,)))
	pos = tf.gather(tf.reshape(pivots,(-1,3)), i)

	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	bbox *= 1.0 - big * 0.5
	X += (1.0 - tf.cast(sign, tf.float32)*2.0) * bbox * tf.cast(mask, tf.float32)

	idx = tf.where(mask)[...,0]
	X = tf.gather(X, idx)
	nodes = tf.gather(nodes, idx)[...,0]

	i = tf.argsort(nodes)
	nodes = tf.gather(nodes, i)
	X = tf.gather(X, i)
	return X, nodes, pivots, pos, bbox, flags, uids, dims

def encode2(X, nodes, pos, bbox, radius, means=0):
	uids, inv = tf.unique(nodes)
	n = tf.math.reduce_max(inv)+1
	intervals = bbox / radius
	i = tf.argsort(intervals, axis=-1, direction='DESCENDING')
	bbox = tf.gather(bbox, i, batch_dims=1)
	radius = tf.gather(radius, i, batch_dims=1)
	pos = tf.gather(pos, i, batch_dims=1)
	means = tf.gather(means, i, batch_dims=1)
	i = tf.gather(i, inv)
	X = tf.gather(X, i, batch_dims=1)

	big = tf.cast(bbox > radius, tf.int32)
	bbox = bbox[...,None,:] * tokens[None,...] + means[...,None,:]
	bbox *= tf.cast(big[...,None,:], tf.float32)
	candidates = pos[...,None,:] - bbox

	big = tf.gather(big, inv)
	means = tf.gather(means, inv)
	dims = tf.math.reduce_sum(big, axis=-1, keepdims=True)
	keep = tf.math.reduce_any(tf.math.abs(X) - means, inv > radius, axis=-1, keepdims=True)
	keep = tf.cast(keep, tf.int32)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = tf.gather(keep, inv)

	sign = tf.cast(X >= means, tf.int32)
	bits = bitops.left_shift(sign, tf.range(3))
	bits = tf.math.reduce_sum(bits, axis=-1, keepdims=True)
	bits = bitops.bitwise_and(bits, bitops.left_shift(1,dims)-1)
	
	nodes = nodes[...,None]
	nodes = bitops.left_shift(nodes, tf.cast(dims, nodes.dtype))
	nodes = bitops.bitwise_or(nodes, tf.cast(bits, nodes.dtype))
	nodes = nodes[...,0]

	flags = tf.one_hot(bits[...,0], 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * keep

	i = tf.where(tf.reshape(flags, (-1,)))[...,0]
	pos = tf.gather(tf.reshape(candidates,(-1,3)), i)
	bbox = tf.gather(tf.reshape(bbox,(-1,3)), i)

	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	i = tf.where(mask[...,0])[...,0]
	X = tf.gather(X, i)
	X += bbox
	
	nodes = tf.gather(nodes, i)
	bbox = tf.gather(bbox, i)
	inv = tf.gather(inv, i)

	i = tf.argsort(nodes)
	X = tf.gather(X, i)
	nodes = tf.gather(nodes, i)
	bbox = tf.gather(bbox, i)
	
	inv = tf.gather(inv, i)
	means = tf.math.unsorted_segment_mean(X, inv, n)
	return X, nodes, pos, candidates, bbox, flags, uids, dims, means

def decode(flags, bbox, radius, X, keep):
	signs = bitops.right_shift(flags[...,None], tf.range(8))
	signs = bitops.bitwise_and(signs, 1)
	offsets = tf.cast(signs[...,None], tokens.dtype) * tokens[None,...] * bbox * tf.cast(bbox > radius, tokens.dtype)
	i = tf.where(flags == 0)
	keep = tf.concat([keep, tf.gather(X, i[...,0])], axis=-2)
	i = tf.where(signs)
	X = tf.gather(X, i[...,-2]) - tf.gather_nd(offsets, i)
	X = tf.concat([X, keep], axis=-2)
	return X, keep, bbox*0.5

def decode2(flags, bbox, radius, X, keep, means=0, batch_dims=0):
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
	X, nodes, pos, candidates, bbox, flags, uids, dims, means = encode2(X, nodes, pos, bbox, radius, means)
	print(X, nodes, pos, candidates, bbox, flags, uids, dims, means)