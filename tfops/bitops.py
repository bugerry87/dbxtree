
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.bitwise import *


def serialize(X, bits_per_dim):
	assert(sum(bits_per_dim) < 64)
	with tf.name_scope("serialize"):
		offset = tf.math.reduce_min(X, axis=0)
		X = X - offset
		scale = (1<<np.array(bits_per_dim)) - 1
		scale /= tf.math.reduce_max(X, axis=0)
		X *= scale
		X = tf.math.round(X)
		X = tf.cast(X, tf.int64)
		shifts = np.cumsum([0] + list(bits_per_dim[:-1]), dtype=np.int64)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1)
		X = tf.reshape(X, (-1,1))
	return X, offset, scale


def realize(X, bits_per_dim, offset, scale, xtype=np.float32):
	assert(sum(bits_per_dim) < 64)
	with tf.name_scope("realize"):
		masks = (1<<np.array(bits_per_dim, dtype=np.int64)) - 1
		shifts = np.cumsum([0] + list(bits_per_dim[:-1]), dtype=np.int64)
		X = right_shift(X, shifts)
		X = bitwise_and(X, masks)
		X = tf.cast(X, xtype)
		X /= scale
		X += offset
	return X


def sort(X, bits=64, reverse=False, absolute=False):
	with tf.name_scope("sort_bits"):
		shifts = tf.range(bits, dtype=X.dtype)
		Y = right_shift(X, shifts)
		Y = bitwise_and(Y, 1)
		
		p = tf.math.reduce_sum(Y, axis=0)
		if absolute:
			p2 = tf.math.reduce_sum(1-Y, axis=0)
			p = tf.math.reduce_max((p, p2), axis=0)
		p = tf.argsort(p)
		if reverse:
			p = p[::-1]
		p = tf.cast(p, dtype=X.dtype)
		
		X = right_shift(X, p)
		X = bitwise_and(X, 1)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1)
		X = tf.reshape(X, (-1,1))
	return X, p


def permute(X, p, bits=64):
	with tf.name_scope("permute_bits"):
		X = right_shift(X, tf.range(bits, dtype=X.dtype))
		X = bitwise_and(X, 1)
		X = left_shift(X, p)
		X = tf.math.reduce_sum(X, axis=-1)
		X = tf.reshape(X, (-1,1))
	return X


def tokenize(X, dim, depth):
	with tf.name_scope("tokenize"):
		X = tf.sort(X, axis=0)
		shifts = np.arange(depth, dtype=np.int64) * dim
		tokens = right_shift(X, shifts[::-1])
		tokens = tf.transpose(tokens)
	return tokens


def encode(nodes, idx, dim, dtype=tf.uint8):
	bits = 1<<dim
	with tf.name_scope("encode"):
		flags = tfbitops.bitwise_and(nodes, bits-1)
		flags = tf.one_hot(flags, bits, dtype=dtype)
		flags = tf.math.unsorted_segment_max(flags, idx, idx[-1] + 1)
		flags = tf.math.reduce_sum(flags, axis=-1)
		idx = tf.concat((nodes[:,0], nodes), axis=-1)
		idx = idx[:,:-1] == idx[:,1:]
		idx = tf.cast(idx, tf.int32)
		idx = tf.math.cumsum(idx)
	return flags, idx