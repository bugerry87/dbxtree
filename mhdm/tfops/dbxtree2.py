
## Installed
import tensorflow as tf

## Local
from . import bitops

@tf.function(experimental_relax_shapes=True)
def encode(X, nodes, inv, bbox, radius, means=None, pos=None):
	n = tf.math.reduce_max(inv) + 1
	
	big = tf.cast(bbox > radius, nodes.dtype)
	dims = tf.math.reduce_sum(big, axis=-1, keepdims=True)
	means = tf.gather(means, inv) if means is not None else 0
	pos = tf.gather(pos, inv) if pos is not None else 0
	X -= means
	pos += means
	keep = tf.math.reduce_any(tf.math.abs(X) > radius, axis=-1, keepdims=True)
	keep = tf.cast(keep, X.dtype)
	keep = tf.math.unsorted_segment_max(keep, inv, n)
	mask = tf.gather(keep, inv)

	sign = tf.cast(X >= 0, X.dtype)
	bbox = tf.gather(bbox, inv)
	big = tf.gather(big, inv)
	bbox *= 0.5 - sign
	bbox += means * 0.5
	bbox *= tf.cast(big, X.dtype) * mask
	X += bbox
	pos -= bbox
	bbox = tf.abs(bbox)

	sign = tf.cast(sign, nodes.dtype) * big
	shifts = tf.cumsum(big, exclusive=True, axis=-1)
	bits = bitops.left_shift(sign, shifts)
	bits = tf.math.reduce_sum(bits, axis=-1)
	
	nodes = bitops.left_shift(nodes, 3)
	nodes += bits

	flags = tf.one_hot(bits, 8, dtype=tf.int32)
	flags = tf.math.unsorted_segment_max(flags, inv, n) * tf.cast(keep, flags.dtype)
	flags = bitops.left_shift(flags, tf.range(8))
	flags = tf.math.reduce_sum(flags, axis=-1)

	i = tf.argsort(nodes)
	mask = tf.gather(mask[...,0], i)
	i = tf.gather(i, tf.where(mask)[...,0])
	nodes = tf.gather(nodes, i)
	bbox = tf.gather(bbox, i)
	X = tf.gather(X, i)
	pos = tf.gather(pos, i)
	uids, inv = tf.unique(nodes)
	n = tf.math.maximum(tf.math.reduce_max(inv) + 1, 0)
	
	bbox = tf.math.unsorted_segment_max(bbox, inv, n)
	means = tf.math.unsorted_segment_mean(X, inv, n)
	pos = tf.math.unsorted_segment_mean(pos, inv, n)
	return X, nodes, inv, bbox, flags, dims, means, pos, uids

def decode(flags, bbox, radius, X, keep, means=None):
	i = tf.where(flags == 0)[...,0]
	keep = tf.concat([keep, tf.gather(X, i)], axis=-2)

	bits = bitops.right_shift(flags[...,None], tf.range(8))
	i = tf.where(bitops.bitwise_and(bits, 1))
	dims = tf.cast(bbox > radius, bits.dtype)
	dims = tf.gather(dims, i[...,0])
	bbox = tf.gather(bbox, i[...,0])
	means = tf.gather(means, i[...,0]) if means is not None else 0

	bits = tf.cast(i[...,-1], bits.dtype)
	bits = bitops.right_shift(bits[...,None], tf.range(3))
	bits = bitops.bitwise_and(bits, 1)
	bits = tf.gather(bits, tf.cumsum(dims, axis=-1, exclusive=True), batch_dims=1)
	bbox *= 0.5 - tf.cast(bits, bbox.dtype)
	bbox += means * 0.5
	bbox *= tf.cast(dims, bbox.dtype)
	
	X = tf.gather(X, i[...,0]) + means - bbox
	X = tf.concat([X, keep], axis=-2)
	bbox = tf.abs(bbox)
	return X, keep, bbox