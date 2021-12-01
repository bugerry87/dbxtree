
## Installed
import tensorflow as tf

## Local
from . import bitops


def encode(X, nodes, bbox, radius):
	u, inv = tf.unique(nodes)
	u *= 0
	n = tf.reduce_max(inv)+1

	big = tf.cast(bbox > radius, tf.int32)
	dims = tf.math.reduce_sum(big, axis=-1)
	keep = tf.math.reduce_any(X > radius, axis=-1)
	keep = tf.cast(keep, tf.int32)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = dims * tf.gather(keep, inv) > 0

	i = tf.argsort(bbox, axis=-1)[...,::-1]
	sign = tf.gather(X, i, batch_dims=tf.rank(X)-1) >= 0
	sign = tf.cast(sign, tf.int32)
	bits = bitops.left_shift(sign, tf.range(3))
	bits = tf.math.reduce_sum(bits, axis=-1)

	nodes = bitops.left_shift(nodes, dims)
	nodes = bitops.bitwise_or(nodes, bits)

	hist = tf.one_hot(bits, 8)
	hist = tf.math.unsorted_segment_sum(hist, inv, n)

	flags = tf.cast(hist>0, tf.int32)
	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	bbox *= 1.0 - tf.cast(big, tf.float32)*0.5
	X += (1.0 - tf.cast(X>=0.0, tf.float32)*2.0) * bbox * tf.cast(mask, tf.float32)[...,None]

	dims = tf.math.unsorted_segment_sum(dims, inv, n)
	idx = tf.where(dims>0)
	flags = tf.gather(flags*keep, idx, batch_dims=tf.rank(flags)-1)

	idx = tf.where(mask)
	X = tf.gather(X, idx, batch_dims=tf.rank(X)-2)
	nodes = tf.gather(nodes, idx, batch_dims=tf.rank(nodes)-1)
	bbox = tf.gather(bbox, idx, batch_dims=tf.rank(bbox)-1)
	return X, nodes, bbox, flags
