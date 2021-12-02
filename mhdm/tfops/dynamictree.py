
## Installed
import tensorflow as tf

## Local
from . import bitops


def encode(X, nodes, bbox, radius):
	u, inv = tf.unique(nodes)
	n = tf.reduce_max(inv)+1

	big = tf.cast(bbox > radius, tf.int32)
	dims = tf.math.reduce_sum(big, axis=-1, keepdims=True)
	keep = tf.math.reduce_any(tf.math.abs(X) > radius, axis=-1, keepdims=True)
	keep = tf.cast(keep, tf.int32)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = dims * tf.gather(keep, inv) > 0

	#bbox = tf.math.unsorted_segment_max(bbox, inv, n)
	#i = tf.argsort(bbox, axis=-1)[...,::-1]
	#i = tf.gather(i, inv)
	#bbox = tf.gather(bbox, inv)
	#sign = tf.gather(X, i, batch_dims=tf.rank(X)-1) >= 0
	#sign = tf.cast(sign, tf.int32)

	sign = tf.cast(X>=0, tf.int32) * big
	#input(tf.math.cumsum(big, exclusive=True, axis=-1))
	bits = bitops.left_shift(sign, tf.math.cumsum(big, exclusive=True, axis=-1))
	
	#bits = bitops.left_shift(sign, tf.range(3))
	bits = tf.math.reduce_sum(bits, axis=-1, keepdims=True)
	bits = bitops.bitwise_and(bits, bitops.left_shift(1,dims)-1)
	
	nodes = nodes[...,None]
	nodes = bitops.left_shift(nodes, tf.cast(dims, nodes.dtype))
	nodes = bitops.bitwise_or(nodes, tf.cast(bits, nodes.dtype))

	flags = tf.one_hot(bits[...,0], 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n)
	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1, keepdims=True)

	bbox *= 1.0 - tf.cast(big, tf.float32)*0.5
	X += (1.0 - tf.cast(X>=0.0, tf.float32)*2.0) * bbox * tf.cast(mask, tf.float32)
	
	dims = tf.math.unsorted_segment_max(dims, inv, n)
	idx = tf.where(dims>0)[...,0]
	flags = tf.gather(flags*keep, idx)[...,0]
	dims = tf.gather(dims, idx)

	idx = tf.where(mask)[...,0]
	X = tf.gather(X, idx)
	nodes = tf.gather(nodes, idx)[...,0]
	bbox = tf.gather(bbox, idx)

	i = tf.argsort(nodes)
	nodes = tf.gather(nodes, i)
	bbox = tf.gather(bbox, i)
	X = tf.gather(X, i)

	shifts = bitops.left_shift(1, dims)[...,0]
	return X, nodes, bbox, flags, shifts
