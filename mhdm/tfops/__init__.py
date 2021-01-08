

## Installed
import tensorflow as tf


def batched_identity(shape, dtype='float32'):
	return tf.eye(shape[2], shape[3], batch_shape=shape[1:2], dtype=dtype),


def range_like(input, start=0, stop=None):
	r = tf.ones_like(input)
	r = tf.math.cumsum(r, axis=-1)
	if stop is not None:
		r /= tf.math.reduce_max(r, axis=-1, keepdims=True)
		r *= stop - start
	r += start
	return r