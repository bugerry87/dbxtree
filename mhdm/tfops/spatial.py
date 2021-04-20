
## Installed
import numpy as np
import tensorflow as tf


@tf.function
def xyz2uvd(X):
    x, y, z = X[...,0], X[...,1], X[...,2]
	pi = tf.where(x > 0.0, np.pi, -np.pi)
    pi = tf.where(y < 0.0, pi, 0.0)
	u = tf.math.atan(x / y) + pi
	d = tf.norm(X, axis=-1)
    v = tf.math.divide_no_nan(z, d)
	v = tf.math.asin(v)
	return tf.concat((u[...,None],v[...,None],d[...,None]), axis=-1)

@tf.function
def uvd2xyz(U):
    u, v, d = U[..., 0], U[..., 1], U[..., 2]
	x = tf.math.sin(u) * d 
	y = tf.math.cos(u) * d 
	z = tf.math.sin(v) * d
	return tf.concat((x[...,None],y[...,None],z[...,None]), axis=-1)