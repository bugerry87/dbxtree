

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.python.ops.bitwise_ops import bitwise_and, bitwise_or, right_shift, left_shift


def serialize(X, bits_per_dim):
	assert(sum(bits_per_dim) < 64)
	with tf.name_scope("serialize"):
		offset = tf.math.reduce_min(X, axis=0)
		X = X - offset
		scale = tf.math.reduce_max(X, axis=0)
		X = tf.where(scale == 0, 0.0, X / scale)
		X *= (1<<np.array(bits_per_dim)) - 1
		X = tf.math.round(X)
		X = tf.cast(X, tf.int64)
		shifts = np.cumsum([0] + bits_per_dim[:-1], dtype=np.int8)
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1)
		X = tf.unique(X, out_idx=tf.int64)[0]
		X = tf.reshape(X, (-1,1))
	return X


def sort_bits(X, bits=64, reverse=False, absolute=False):
	with tf.name_scope("sort_bits"):
		Y = right_shift(X, np.arange(bits))
		Y = bitwise_and(Y, 1)
		if absolute:
			p = tf.math.reduce_sum(Y, axis=0)
		
		
	return X, p


def tokenize(X, dim, depth):
	with tf.name_scope("tokenize"):
		shifts = np.arange(depth, dtype=np.int64) * dim
		mask = (1<<shifts+dim) - 1
		tokens = bitwise_and(X, mask)
		tokens = tf.transpose(tokens)
	return tokens


class NbitTree():
	"""
	"""
	def __init__(self, *args, build_encoder=True, build_decoder=True, **kwargs):
		"""
		"""		
		if build_encoder:
			self.build_encoder(*args, **kwargs)
		
		if build_decoder:
			self.build_decoder(*args, **kwargs)
		pass
	
	def build_encoder(self, dim, bits_per_dim, *args,
		xtype=tf.float32,
		ftype=tf.uint8,
		reverse=True,
		**kwargs
		):
		"""
		"""
		bits = 1<<dim
		tree_depth = int(np.ceil(sum(bits_per_dim) / dim))
		tokens = np.arange(bits, dtype=np.uint8)
		t_list = tf.TensorShape([None])
		
		def until_tree_end(*args):
			return True
		
		def iterate_layer(flags0, idx0, dummy0, layer):
			flags1 = nodes[layer]
			dummy1, idx1 = tf.unique(flags1, out_idx=tf.int64)
			dummy1 = tf.one_hot(dummy1, bits, dtype=ftype)
			flags1 = right_shift(flags1, layer*dim)
			flags1 = tf.one_hot(flags1, bits, dtype=ftype)
			flags1 = tf.tensor_scatter_nd_max(dummy0, idx0[:,None], flags1)
			flags1 = left_shift(flags1, tokens)
			flags1 = tf.math.reduce_sum(flags1, axis=1)
			flags1 = tf.reshape(flags1, (-1,))
			flags1 = tf.concat([flags0, flags1], axis=0)
			return flags1, idx1, dummy1 * 0, layer+1
		
		self._encoder_input = tf.compat.v1.placeholder(shape=(None,len(bits_per_dim)), dtype=xtype, name='encoder_input')
		with tf.name_scope("{}bitTree".format(bits)):
			with tf.name_scope("boot_encoder"):
				nodes = serialize(self._encoder_input, bits_per_dim)
				nodes = tokenize(nodes, dim, tree_depth)
				flags, idx = tf.unique(nodes[0], out_idx=tf.int64)
				flags = tf.cast(flags, ftype)
				dummy = tf.one_hot(flags, bits, dtype=ftype) * 0
				flags = left_shift(tf.constant(1, dtype=ftype), flags)
				flags = tf.math.reduce_sum(flags)[None,]
				layer = tf.constant(0, dtype=tf.int64)
			
			self._encoder_output, idx, dummy, layer = tf.while_loop(
				until_tree_end, iterate_layer,
				loop_vars=(flags, idx, dummy, layer),
				shape_invariants=(t_list, idx.get_shape(), dummy.get_shape(), layer.get_shape()),
				maximum_iterations=tree_depth-1,
				name='encoder'
				)
		pass
	
	def build_decoder(self, *args, **kwargs):
		pass
	
	def encode(self, X, callbacks=None):
		with tf.compat.v1.Session() as sess:
			with tf.compat.v1.summary.FileWriter("output", sess.graph) as writer:
				output = sess.run(self._encoder_output, feed_dict={self._encoder_input:X})
		return output


import datetime
tf.compat.v1.disable_eager_execution()
#log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
X = np.fromfile('data/0000000000.bin', dtype=np.float32).reshape(-1, 4)
tree = NbitTree(3, [16,16,16,0])
output = tree.encode(X)
print(output[-1000:], len(output))