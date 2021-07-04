
## Installed
import numpy as np
import tensorflow as tf

right_shift = tf.bitwise.right_shift
left_shift = tf.bitwise.left_shift
bitwise_and = tf.bitwise.bitwise_and
bitwise_or = tf.bitwise.bitwise_or
bitwise_xor = tf.bitwise.bitwise_xor
invert = tf.bitwise.invert


def serialize(X, bits_per_dim, offset=None, scale=None, axis=0, dtype=tf.int64):
	with tf.name_scope("serialize"):
		one = tf.constant(1, dtype=dtype, name='one')
		bits_per_dim = tf.cast(bits_per_dim, dtype)
		lim = tf.cast(left_shift(one, bits_per_dim) - one, X.dtype) * 0.5
		
		if offset is None:
			offset = -tf.math.reduce_min(X, axis, keepdims=True)
		else:
			offset = tf.cast(offset, X.dtype)

		if scale is None:
			scale = tf.math.reduce_max(tf.abs(X), axis, keepdims=True)
		else:
			scale = tf.cast(scale, X.dtype)
		
		X = X + offset
		X = tf.math.divide_no_nan(X * lim, scale)
		X = tf.cast(X, dtype)
		shifts = tf.math.cumsum(bits_per_dim, exclusive=True)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X, offset, scale


def realize(X, bits_per_dim, offset=0.0, scale=1.0, xtype=tf.float32):
	with tf.name_scope("realize"):
		one = tf.constant(1, dtype=X.dtype, name='one')
		bits_per_dim = tf.cast(bits_per_dim, X.dtype)
		mask = left_shift(one, bits_per_dim) - one
		cells = tf.math.divide_no_nan(tf.cast(scale, xtype), tf.cast(mask, xtype) * 0.5)
		shifts = tf.math.cumsum(bits_per_dim, exclusive=True)
		X = right_shift(X, shifts)
		X = bitwise_and(X, mask)
		X = tf.cast(X, xtype)
		X *= cells
		X += cells * 0.5
		X -= tf.cast(offset, xtype)
	return X


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


def permute(X, p, bits=63):
	with tf.name_scope("permute_bits"):
		one = tf.constant(1, dtype=X.dtype, name='one')
		X = right_shift(X, tf.range(bits, dtype=X.dtype))
		X = bitwise_and(X, one)
		X = left_shift(X, p)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X


def tokenize(X, dim, depth, axis=0):
	with tf.name_scope("tokenize"):
		X = tf.sort(X, axis=axis)
		shifts = tf.range(depth, dtype=X.dtype) * tf.cast(dim, X.dtype)
		tokens = right_shift(X, shifts[::-1])
		tokens = tf.transpose(tokens)
	return tokens


def encode(nodes, idx, dim, ftype=tf.int64, htype=tf.int64):
	with tf.name_scope("encode"):
		one = tf.constant(1, nodes.dtype)
		bits = left_shift(one, dim)
		shifts = tf.range(bits)
		shifts = tf.cast(shifts, ftype)
		flags = bitwise_and(nodes, bits-1)
		hist = tf.one_hot(flags, tf.cast(bits, tf.int32), dtype=htype)
		hist = tf.math.unsorted_segment_sum(hist, idx, idx[-1]+1)
		flags = tf.cast(hist>0, ftype)
		flags = left_shift(flags, shifts)
		flags = tf.math.reduce_sum(flags, axis=-1)
	return flags, hist


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