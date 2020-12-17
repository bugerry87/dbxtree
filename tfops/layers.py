
## Installed
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer, Dense, Permute, Dot


class NbitTreeEncoder(Layer):
	"""
	"""
	def __init__(self, dim, 
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTreeEncoder, name=name if name else "{}bitTreeEncoder".format(dim))
		self.dim = dim
		self.fbits = 1<<dim
		pass
	
	def __call__(self, inputs):
		idx = tf.concat((inputs[:,0], inputs), axis=-1)
		idx = idx[:,:-1] == idx[:,1:]
		idx = tf.cast(idx, tf.int32)
		idx = tf.math.cumsum(idx)
		size = idx[-1] + 1
		flags = tfbitops.bitwise_and(inputs, self.fbits-1)
		flags = tf.one_hot(flags, self.fbits, dtype=self.ftype)
		flags = tf.math.unsorted_segment_max(flags, idx, size)
		flags = tf.math.reduce_sum(flags, axis=-1)
		return flags, idx


class Transformer(Layer):
	"""
	"""
	def __init__(self, k,
		axes=(1,2),
		activation='relu',
		normalize=True,
		name='Transformer',
		**kwargs
		):
		"""
		"""
		super(Transformer, self).__init__(name=name, **kwargs)
		self.permute = Permute(axes[::-1], **kwargs)
		self.dot = Dot(axes, normalize, **kwargs)
		self.dense_n = Dense(k, activation=activation, **kwargs)
		self.dense_m = Dense(k, activation=activation, **kwargs)
		self.dense_t = Dense(k, activation=activation, **kwargs)
		pass
	
	def __call__(self, inputs)
		"""
		"""
		n = self.dense_n(inputs) #(b, n, k)
		m = self.dense_m(inputs) #(b, m, k)
		t = self.dense_t(inputs) #(b, t, k)
		m = self.permute(m) #(b, k, m)
		t = self.permute(t) #(b, k, t)
		T = self.dot((n,m)) #(b, k, k) Transformer!
		return self.dot((t,T)) #(b, t, k)
