

## Installed
import tensorflow as tf


def batched_identity(shape, dtype='float32'):
	return tf.eye(shape[2], shape[3], batch_shape=shape[1:2], dtype=dtype),


def range_like(input, start=0, stop=None, dtype=None):
	r = tf.ones_like(input, dtype=dtype)
	r = tf.math.cumsum(r, axis=-1, exclusive=True)
	if stop is not None:
		r /= tf.math.reduce_max(r, axis=-1, keepdims=True)
		r *= stop - start
	r += start
	return r


def count(input, dtype=None):
	c = tf.ones_like(input, dtype=dtype)
	c = tf.math.reduce_sum(c)
	return c

def yield_devices(prefer=None):
	devices = tf.python.client.device_lib.list_local_devices()
	devices = [d for d in devices if d.device_type in prefer] or devices
	i = 0
	while True:
		assert(devices[i % len(devices)].device_type != 'CPU')
		yield devices[i % len(devices)]
		i += 1