
## Installed
import tensorflow as tf

## Local
from . import bitops

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

def encode2(X, nodes, pos, bbox, radius, means=0, batch_dims=0):
	uids, inv = tf.unique(nodes)
	n = tf.math.reduce_max(inv)+1
	intervals = bbox / radius
	i = tf.argsort(intervals, axis=-1, direction='DESCENDING')
	bbox = tf.gather_nd(bbox, i, batch_dims)
	radius = tf.gather_nd(radius, i, batch_dims+1)
	pos = tf.gather_nd(pos, i, batch_dims+1)
	
	means = tf.gather_nd(means, i, batch_dims+1)
	means = tf.gather_nd(means, inv, batch_dims)

	bbox = tf.gather_nd(bbox, inv, batch_dims)
	i = tf.gather_nd(i, inv, batch_dims)
	X = tf.gather_nd(X, i, batch_dims+1)

	big = tf.cast(bbox > radius, tf.int32)
	dims = tf.math.reduce_sum(big, axis=-1)
	keep = tf.math.reduce_any(tf.math.abs(X) - means > radius, axis=-1, keepdims=True)
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
	
	big = tf.cast(big, tf.float32)
	candidates = pos - (bbox + means) * big * tokens[None,...]

	flags = tf.one_hot(bits[...,0], 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * keep

	i = tf.where(tf.reshape(flags, (-1,)))
	pos = tf.gather_nd(tf.reshape(candidates,(-1,3)), i, batch_dims)

	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	bbox *= 1.0 - big * 0.5
	bbox += means * big
	X += (1.0 - tf.cast(sign, tf.float32)*2.0) * bbox * tf.cast(mask, tf.float32)

	i = tf.where(mask)[...,0]
	X = tf.gather_nd(X, i, batch_dims)
	nodes = tf.gather_nd(nodes, i, batch_dims)[...,0]
	bbox = tf.gather_nd(bbox, i, batch_dims)[...,0]
	inv = tf.gather_nd(inv, i, batch_dims)[...,0]

	i = tf.argsort(nodes)
	X = tf.gather_nd(X, i, batch_dims)
	nodes = tf.gather_nd(nodes, i, batch_dims)
	bbox = tf.gather_nd(bbox, i, batch_dims)
	inv = tf.gather_nd(inv, i, batch_dims)
	means = tf.math.unsorted_segment_mean(X, inv, n)
	bbox = tf.math.unsorted_segment_max(bbox, inv, n)

	return X, nodes, pos, candidates, bbox, flags, uids, dims, means

def encode3(X, nodes, pos, bbox, radius, means=0, batch_dims=0):
	uids, inv = tf.unique(nodes)
	n = tf.math.reduce_max(inv) + 1
	bbox = tf.gather_nd(bbox, inv, batch_dims)

	big = tf.cast(bbox > radius, nodes.dtype)
	dims = tf.math.reduce_sum(big, axis=-1)
	keep = tf.math.reduce_any(tf.math.abs(X) - means > radius, axis=-1, keepdims=True)
	keep = tf.cast(keep, nodes.dtype)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = tf.gather(keep, inv)

	sign = tf.cast(X >= means, nodes.dtype)
	sign = bitops.bitwise_and(sign, big)
	bits = tf.cumsum(big, axis=-1) - tf.math.reduce_max(big, axis=-1)
	bits = bitops.left_shift(sign, bits)
	bits = tf.math.reduce_sum(bits, axis=-1, keepdims=True)
	
	nodes = nodes[...,None]
	nodes = bitops.left_shift(nodes, dims)
	nodes = bitops.bitwise_or(nodes, bits)
	
	big = tf.cast(big, X.dtype)
	pivots = pos - (bbox + means) * big * tokens[None,...]

	flags = tf.one_hot(bits[...,0], 8, dtype=nodes.dtype)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * keep

	i = tf.where(tf.reshape(flags, (-1,)))
	pos = tf.gather_nd(tf.reshape(pivots,(-1,3)), i, batch_dims)

	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	sign = tf.cast(sign, X.dtype)
	mask = tf.cast(mask, X.dtype)
	bbox *= 1.0 - big * 0.5
	bbox += means * big
	X += (1.0 - sign * 2.0) * bbox * mask

	i = tf.where(mask)[...,0]
	X = tf.gather_nd(X, i, batch_dims)
	nodes = tf.gather_nd(nodes, i, batch_dims)[...,0]
	bbox = tf.gather_nd(bbox, i, batch_dims)[...,0]
	inv = tf.gather_nd(inv, i, batch_dims)[...,0]

	i = tf.argsort(nodes)
	X = tf.gather_nd(X, i, batch_dims)
	nodes = tf.gather_nd(nodes, i, batch_dims)
	bbox = tf.gather_nd(bbox, i, batch_dims)
	inv = tf.gather_nd(inv, i, batch_dims)
	means = tf.math.unsorted_segment_mean(X, inv, n)
	bbox = tf.math.unsorted_segment_max(bbox, inv, n)
	inv = tf.math.invert_permutation(i)

	return X, nodes, pivots, pos, bbox, flags, uids, dims, means, inv

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