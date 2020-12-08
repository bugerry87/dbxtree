

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.python.ops.bitwise_ops import bitwise_and, bitwise_or, right_shift, left_shift


def serialize(X, bits_per_dim, reverse=True):
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
		if reverse:
			shifts = shifts[::-1]
		X = left_shift(X, shifts)
		X = tf.math.reduce_sum(X, axis=-1)
		X = tf.unique(X, out_idx=tf.int64)[0]
		X = tf.reshape(X, (-1,1))
	return X


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
	def __init__(self, dim, bits_per_dim,
		xtype=tf.float32,
		ftype=tf.uint8,
		kernel=64
		):
		"""
		"""
		self._dim = dim
		self._tree_depth = int(np.ceil(sum(bits_per_dim) / dim))
		t_list = tf.TensorShape([None])
		t_scalar = tf.TensorShape([])
		
		def until_tree_end(flags, nodes, layer):
			return layer < self._tree_depth
		
		def iterate_layer(flags, nodes, layer):
			tokens, idx = tf.unique(nodes[layer], out_idx=tf.int64)
			tensor_scatter_nd_max()
			flag = right_shift(flag, layer*dim)
			flag = tf.cast(flag, ftype)
			flag = left_shift(tf.constant(1, dtype=ftype), flag)
			flag = tf.reshape(flag, (-1,))
			flags = tf.concat([flags, flag], axis=0)
			return flags, nodes, layer+1
		
		#input = Input(shape=len(bits_per_dim), dtype=xtype)
		self._input = tf.compat.v1.placeholder(shape=(None,len(bits_per_dim)), dtype=xtype)
		nodes = serialize(self._input, bits_per_dim)
		nodes = tokenize(nodes, dim, self._tree_depth)
		flags, idx = tf.unique(nodes[0], out_idx=tf.int64)
		
		with tf.name_scope("{}bitTree".format(1<<dim)):
			layer = tf.constant(0, dtype=tf.int64)
			flags = tf.constant([], dtype=ftype)
			
			self._output, nodes, layer = tf.while_loop(
				until_tree_end, iterate_layer,
				loop_vars=(flags, nodes, layer),
				shape_invariants=(t_list, nodes.get_shape(), t_scalar),
				name='iterate_layer'
				)
			#self._encoder = Model(input, flags)
	
	@property
	def dim(self):
		return self._dim
	
	@property
	def tree_depth(self):
		return self._tree_depth
	
	def encode(self, X, callbacks=None):
		with tf.compat.v1.Session() as sess:
			with tf.compat.v1.summary.FileWriter("output", sess.graph) as writer:
				output = sess.run(self._output, feed_dict={self._input:X})
		#return self._encoder.predict(X, callbacks=callbacks)[0]
		return output

import datetime
tf.compat.v1.disable_eager_execution()
#log_dir = "logs\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
tree = NbitTree(3, [16,16,16,0])
output = tree.encode(np.random.rand(1000,4).astype(np.float32))
print(output, len(output))