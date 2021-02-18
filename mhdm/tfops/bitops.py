
## Installed
import numpy as np
import tensorflow as tf

right_shift = tf.bitwise.right_shift
left_shift = tf.bitwise.left_shift
bitwise_and = tf.bitwise.bitwise_and
bitwise_or = tf.bitwise.bitwise_or
bitwise_xor = tf.bitwise.bitwise_xor
invert = tf.bitwise.invert


@tf.function
def serialize(X, bits_per_dim, offset=None, scale=None, axis=0, dtype=tf.uint64):
	with tf.name_scope("serialize"):
		inv = tf.constant(-1, dtype=dtype, name='one')
		bits_per_dim = tf.cast(bits_per_dim, dtype)
		lim = invert(left_shift(inv, bits_per_dim))
		if offset is None:
			offset = -tf.math.reduce_min(X, axis, keepdims=True)
		else:
			offset = tf.cast(offset, X.dtype)
		X = X + offset
		
		if scale is None:
			scale = tf.cast(lim, X.dtype)
			scale /= tf.math.reduce_max(X, axis, keepdims=True)
		else:
			scale = 1 / tf.cast(scale, X.dtype)
		X = X * scale
		
		X = tf.math.round(X)
		X = tf.cast(X, dtype)
		shifts = tf.math.cumsum(bits_per_dim, exclusive=True)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X, offset, scale


@tf.function
def realize(X, bits_per_dim, offset, scale, xtype=np.float32):
	assert(sum(bits_per_dim) < 64)
	with tf.name_scope("realize"):
		masks = (1<<np.array(bits_per_dim, dtype=np.int64)) - 1
		shifts = np.cumsum([0] + list(bits_per_dim[:-1]), dtype=np.int64)
		X = right_shift(X, shifts)
		X = bitwise_and(X, masks)
		X = tf.cast(X, xtype)
		X /= scale
		X -= offset
	return X


@tf.function
def sort(X, bits=64, reverse=False, absolute=False, axis=0):
	with tf.name_scope("sort_bits"):
		one = tf.constant(1, dtype=X.dtype, name='one')
		shifts = tf.range(bits, dtype=X.dtype)
		Y = right_shift(X, shifts)
		Y = bitwise_and(Y, one)
		keepdims = axis!=0
		
		p = tf.math.reduce_sum(Y, axis, keepdims)
		if absolute:
			p2 = tf.math.reduce_sum(one-Y, axis, keepdims)
			p = tf.math.reduce_max((p, p2), axis, keepdims)
		p = tf.argsort(p)
		if reverse:
			p = p[::-1]
		p = tf.cast(p, dtype=X.dtype)
		
		X = right_shift(X, p)
		X = bitwise_and(X, one)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X, p


@tf.function
def permute(X, p, bits=64):
	with tf.name_scope("permute_bits"):
		one = tf.constant(1, dtype=X.dtype, name='one')
		X = right_shift(X, tf.range(bits, dtype=X.dtype))
		X = bitwise_and(X, one)
		X = left_shift(X, p)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X


@tf.function
def tokenize(X, dim, depth, axis=0):
	with tf.name_scope("tokenize"):
		dtype = X.dtype
		X = tf.cast(X, tf.int64)
		X = bitwise_xor(X, -1<<63)
		X = tf.sort(X, axis=axis)
		X = tf.cast(X, dtype)
		shifts = tf.range(depth) * dim
		shifts = tf.cast(shifts, dtype)
		tokens = right_shift(X, shifts[::-1])
		tokens = tf.transpose(tokens)
	return tokens


@tf.function
def encode(nodes, idx, dim, dtype=tf.uint64, buffer=None):
	with tf.name_scope("encode"):
		bits = 1<<dim
		shifts = tf.range(bits)
		shifts = tf.cast(shifts, dtype)
		flags = bitwise_and(nodes, bits-1)
		flags = tf.cast(flags, tf.uint8)
		flags = tf.one_hot(flags, bits, dtype=dtype)
		flags = tf.math.unsorted_segment_max(flags, idx, idx[-1]+1)
		flags = left_shift(flags, shifts)
		flags = tf.math.reduce_sum(flags, axis=-1)
		if buffer is not None:
			flags = tf.concat([buffer, flags], axis=-1)
	return flags


@tf.function
def decode(flags, pos, dim, buffer=tf.constant([0], dtype=tf.int64)):
	with tf.name_scope("decode"):
		size = tf.reshape(tf.size(buffer), [-1])
		flags = tf.slice(flags, pos, size)
		flags = tf.reshape(flags, (-1,1))
		flags = right_shift(flags, np.arange(1<<dim))
		flags = bitwise_and(flags, 1)
		x = tf.where(flags)
		i = x[...,0]
		x = x[...,1]
		X = left_shift(buffer, dim)
		pos += tf.size(X)
		X = x + tf.gather(X, i)
	return X, pos