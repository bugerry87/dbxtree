
import tensorflow as tf
import mhdm.tfops.bitops as bitops


def test_serialization():
	bits_per_dim = tf.constant([16,16,16], dtype=tf.int64)
	X = tf.range(30)
	X = tf.reshape(X, (-1,3))
	Z = tf.cast(X, tf.float32)
	Y, offset, scale = bitops.serialize(Z, bits_per_dim)
	Y = bitops.realize(Y, bits_per_dim, offset, scale, Z.dtype)
	Y = tf.math.round(Y)
	Y = tf.cast(Y, tf.int32)
	result = tf.math.reduce_all(X == Y)
	assert(result)
	