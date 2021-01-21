
## Installed
import tensorflow as tf

right_shift = tf.bitwise.right_shift
left_shift = tf.bitwise.left_shift
bitwise_and = tf.bitwise.bitwise_and
bitwise_or = tf.bitwise.bitwise_or

__one64__ = None
def one64():
	global __one64__
	if __one64__ is None:
		__one64__ = tf.constant(1, dtype=tf.int64)
	return __one64__


def serialize(X, bits_per_dim, offset=None, scale=None, axis=0):
	one = one64()
	with tf.name_scope("serialize"):
		if not isinstance(bits_per_dim, tf.Tensor):
			bits_per_dim = tf.constant(bits_per_dim, dtype=tf.int64)
		
		if offset is None or len(offset.shape) and offset.shape[0] == 0:
			offset = -tf.math.reduce_min(X, axis, keepdims=True)
		X = X + offset
		
		if scale is None or len(scale.shape) and scale.shape[0] == 0:
			scale = left_shift(one, bits_per_dim) - one
			scale = tf.cast(scale, X.dtype)
			scale /= tf.math.reduce_max(X, axis, keepdims=True)
		else:
			scale = 1 / scale
		X = X * scale
		
		X = tf.math.round(X)
		X = tf.cast(X, tf.int64)
		shifts = tf.math.cumsum(bits_per_dim, exclusive=True)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X, offset, scale


def realize(X, bits_per_dim, offset, scale, xtype=tf.float32):
	one = one64()
	with tf.name_scope("realize"):
		masks = left_shift(one, bits_per_dim) - one
		shifts = tf.math.cumsum(bits_per_dim, exclusive=True)
		X = right_shift(X, shifts)
		X = bitwise_and(X, masks)
		X = tf.cast(X, xtype)
		X /= scale
		X -= offset
	return X


def sort(X, bits=64, reverse=False, absolute=False, axis=0):
	with tf.name_scope("sort_bits"):
		shifts = tf.range(bits, dtype=X.dtype)
		Y = right_shift(X, shifts)
		Y = bitwise_and(Y, 1)
		keepdims = axis!=0
		
		p = tf.math.reduce_sum(Y, axis, keepdims)
		if absolute:
			p2 = tf.math.reduce_sum(1-Y, axis, keepdims)
			p = tf.math.reduce_max((p, p2), axis, keepdims)
		p = tf.argsort(p)
		if reverse:
			p = p[::-1]
		p = tf.cast(p, dtype=X.dtype)
		
		X = right_shift(X, p)
		X = bitwise_and(X, 1)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X, p


def permute(X, p, bits=64):
	with tf.name_scope("permute_bits"):
		X = right_shift(X, tf.range(bits, dtype=X.dtype))
		X = bitwise_and(X, one)
		X = left_shift(X, p)
		X = tf.math.reduce_sum(X, axis=-1, keepdims=True)
	return X


def tokenize(X, dim, depth, axis=0):
	one = one64()
	with tf.name_scope("tokenize"):
		X = tf.sort(X, axis=axis)
		shifts = tf.range(depth, dtype=tf.int64) * dim
		tokens = right_shift(X, shifts[::-1])
		tokens = tf.transpose(tokens)
	return tokens


def encode(nodes, idx, dim, dtype=tf.uint8, buffer=None):
	one = one64()
	with tf.name_scope("encode"):
		bits = left_shift(one, dim)
		flags = bitwise_and(nodes, bits-one)
		flags = tf.one_hot(flags, tf.cast(bits, tf.int32), dtype=tf.int64)
		flags = tf.math.unsorted_segment_max(flags, idx, idx[-1]+one)
		flags = left_shift(flags, tf.range(bits))
		flags = tf.math.reduce_sum(flags, axis=-1)
		flags = tf.cast(flags, dtype)
		if buffer is not None:
			flags = tf.concat([buffer, flags], axis=-1)
		uids, idx = tf.unique(nodes, out_idx=nodes.dtype)
	return flags, idx, uids


def decode(flags, pos, dim, buffer=tf.constant([0], dtype=tf.int64)):
	one = one64()
	with tf.name_scope("decode"):
		token = tf.range(left_shift(one, dim))
		size = tf.reshape(tf.size(buffer), [-1])
		flags = tf.slice(flags, pos, size)
		flags = tf.reshape(flags, (-1,1))
		flags = right_shift(flags, token)
		flags = bitwise_and(flags, one)
		x = tf.where(flags)
		i = x[:,0]
		x = x[:,1]
		X = left_shift(buffer, dim)
		pos += tf.size(X)
		X = x + tf.gather(X, i)
	return X, pos