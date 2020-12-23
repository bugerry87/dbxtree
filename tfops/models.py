

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
		dtype=tf.float16,
		**kwargs
		):
		"""
		"""
		super(NbitTreeProbEncoder, self).__init__()
		self.dim = dim
		self.kernel_size = (1<<(1<<dim))-1
		self.transformer = layers.Transformer(self.kernel_size, axes=(0,1), dtype=dtype, **kwargs)
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
		
		files = utils.ifile(index_txt)
		for file in files:
			X = np.fromfile(file, dtype=xtype).reshape(-1, xdims)
			X = bitops.serialize(X, bits_per_dim, scale, offset)[0]
			if permute is not None:
				X = bitops.permute(X, permute, word_length)
			nodes = bitops.tokenize(X, self.dim, tree_depth)
			idx = tf.zeros_like(nodes[0])

			for layer in range(tree_depth):
				uids = bitops.left_shift(uids, (tree_depth-layer-1)*self.dim)
				uids = bitops.right_shift(uids[:,None], np.arange(word_length))
				uids = bitops.bitwise_and(uids, 1)
				uids = bitops.cast(uids, self.dtype)
				flags idx, uids = bitops.encode(nodes[layer+1], idx, self.dim, dtype=ftype)
				labels = tf.one_hot(flags, self.kernel_size, dtype=self.dtype)
				yield uids, labels
	
	def build(self, inputs):
		"""
		"""
		if self.is_build:
			return self
		
		nodes, idx, uids = inputs
		self.probs = self.transformer(uids)
		self.
		self.is_build = True
		return self
	
	def call(self, inputs, training=False):
		if training:
			probs = self.transformer(uids)
			return self.probs
		else:
			return self.flags, self.probs, inputs

