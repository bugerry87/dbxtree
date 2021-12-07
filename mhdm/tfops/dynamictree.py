
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
	
	pivots = pos - bbox * tokens[None,...]

	flags = tf.one_hot(bits[...,0], 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * keep

	i = tf.where(flags)
	pos = tf.gather(pivots, i)

	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1, keepdims=True) 
	flags = flags[...,0]

	bbox *= 1.0 - tf.cast(big, tf.float32)*0.5
	X += (1.0 - tf.cast(sign, tf.float32)*2.0) * bbox * tf.cast(mask, tf.float32)

	idx = tf.where(mask)[...,0]
	X = tf.gather(X, idx)
	nodes = tf.gather(nodes, idx)[...,0]

	i = tf.argsort(nodes)
	nodes = tf.gather(nodes, i)
	X = tf.gather(X, i)

	return X, nodes, pivots, pos, bbox, flags, dims, uids
