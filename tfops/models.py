

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense

## Local
from . import bitops
from . import layers
from . import utils
	

class NbitTreeProbEncoder(Model):
	"""
	"""
	def __init__(self, 
		dim=3,
		dtype=tf.float32,
		**kwargs
		):
		"""
		"""
		super(NbitTreeProbEncoder, self).__init__()
		self.dim = dim
		self.kernel_size = (1<<(1<<dim))
		self.transformer = layers.Transformer(self.kernel_size, dtype=dtype, **kwargs)
		pass
	
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
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, xdims))
			X = bitops.serialize(X, bits_per_dim, scale, offset)[0]
			if permute is not None:
				X = bitops.permute(X, permute, word_length)
			X = bitops.tokenize(X, self.dim, tree_depth+1)
			return X, tf.roll(X, -1, 0), tf.range(tree_depth, -1, -1, dtype=X.dtype)
		
		def encode(X0, X1, shifts):
			uids, idx0 = tf.unique(X0, out_idx=X0.dtype)
			flags, idx1, _ = bitops.encode(X1, idx0, self.dim, ftype)
			uids = bitops.left_shift(uids, shifts*self.dim)
			uids = bitops.right_shift(uids[:,None], np.arange(word_length))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, self.dtype)
			return uids, flags
		
		dataset = tf.data.TextLineDataset(index_txt)
		dataset = dataset.map(parse)
		dataset = dataset.unbatch()
		dataset = dataset.map(encode)
		return dataset
	
	def call(self, inputs):
		return self.transformer(inputs)
		
	