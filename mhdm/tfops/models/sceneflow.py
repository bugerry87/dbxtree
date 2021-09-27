## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, ConvLSTM2D


class SceneFlow(Model):
	"""
	"""
	def __init__(self, filters):

		self.dense = Dense(filters)
		self.lstm = ConvLSTM2D(
			filters,
			[1, filters],
			stateful=True
			)
		pass
	
	def loader(self, index, bits_per_dim,
		offset=None,
		scale=None,
		xtype='float32',
		qtype='int64',
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index, bits_per_dim,
			offset=offset,
			scale=scale,
			xtype=xtype,
			qtype=qtype,
			**kwargs
		)
		
		@tf.function
		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, meta.input_dims))
			i = tf.math.reduce_all(tf.math.is_finite(X), axis=-1)
			X = X[i]
			X, offset, scale = bitops.serialize(X, meta.bits_per_dim, meta.offset, meta.scale, dtype=meta.qtype)
			E, X = map_entropy(X, self.bins)
			return E, X, offset, scale, filename
		
		if isinstance(index, str) and index.endswith('.txt'):
			mapper = tf.data.TextLineDataset(index)
		else:
			mapper = tf.data.Dataset.from_tensor_slices(index)

		if shuffle:
			mapper = mapper.shuffle(shuffle)
		mapper = mapper.map(parse)
		return mapper, meta
	
	def call(self, X):
		X = self.dense(X)
		X = self.lstm(X)
		pass