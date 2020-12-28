

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization

## Local
from . import bitops
from . import layers
from . import utils
	

class NbitTreeProbEncoder(Model):
	"""
	"""
	def __init__(self,
		dim=3,
		k=None,
		normalize=False,
		dtype=tf.float32,
		**kwargs
		):
		"""
		"""
		super(NbitTreeProbEncoder, self).__init__()
		self.dim = dim
		self.kernel_size = k if k else self.output_size
		self.transformer = layers.Transformer(self.kernel_size,
			dtype=dtype,
			normalize=normalize,
			**kwargs
			)
		#self.normalization = LayerNormalization()
		self.output_layer = Dense(
			self.output_size,
			#activation='relu',
			dtype=dtype,
			name='output_layer',
			**kwargs
			)
		pass
	
	@property
	def flag_size(self):
		return 1<<self.dim
	
	@property
	def output_size(self):
		return 1<<(self.flag_size)
	
	def gen_train_data(self, index_txt, bits_per_dim,
		permute=None,
		offset=None,
		scale=None,
		xtype='float32',
		ftype='uint8'
		):
		"""
		"""
		xdims = len(bits_per_dim)
		word_length = sum(bits_per_dim)
		tree_depth = word_length // self.dim + (word_length % self.dim != 0)

		def parse(filename):
			X0 = tf.io.read_file(filename)
			X0 = tf.io.decode_raw(X0, xtype)
			X0 = tf.reshape(X0, (-1, xdims))
			X0, _offset, _scale = bitops.serialize(X0, bits_per_dim, scale, offset)
			if permute is not None:
				X0 = bitops.permute(X0, permute, word_length)
			X0 = bitops.tokenize(X0, self.dim, tree_depth+1)
			X1 = tf.roll(X0, -1, 0)
			shifts = tf.range(tree_depth, -1, -1, dtype=X0.dtype)
			return X0, X1, shifts
		
		def encode(X0, X1, shifts):
			uids, idx0 = tf.unique(X0, out_idx=X0.dtype)
			flags, idx1, _ = bitops.encode(X1, idx0, self.dim, ftype)
			uids = bitops.left_shift(uids, shifts*self.dim)
			uids = bitops.right_shift(uids[:,None], np.arange(word_length))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, self.dtype)
			return uids, tf.one_hot(flags, self.output_size)
		
		dataset = tf.data.TextLineDataset(index_txt)
		dataset = dataset.map(parse)
		dataset = dataset.unbatch()
		dataset = dataset.map(encode)
		return dataset
	
	def build(self, input_shape):
		return super(NbitTreeProbEncoder, self).build(input_shape)
	
	def call(self, inputs, training=False):
			uids = inputs
			X = self.transformer(uids)
			#X = self.normalization(X)
			X = self.output_layer(X)
			X -= tf.math.reduce_min(X - 1.e-16)
			X /= tf.math.reduce_max(X + 1.e-16)
			return X
	
	def encode(self, X):
		pass
	