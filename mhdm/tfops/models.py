
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import range_like
from . import bitops
from . import layers
from .. import utils


class NbitTree(Model):
	"""
	"""
	def __init__(self,
		kernels=None,
		kernel_size=3,
		convolutions=0,
		transformers=0,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTree, self).__init__(name=name, **kwargs)
		self.kernels = kernels or self.output_size
		self.kernel_size = kernel_size

		self.conv = [Conv1D(
			self.kernels, self.kernel_size + 2*i, 1,
			activation='relu',
			padding='same',
			dtype=self.dtype,
			name='conv_{}'.format(i),
			**kwargs
			) for i in range(convolutions)]
		
		self.transformers = [layers.CovarTransformer(
			self.kernels,
			dtype=self.dtype,
			name='transformer_{}'.format(i),
			**kwargs
			) for i in range(transformers)]

		self.head = Dense(
			self.output_size,
			activation='softplus',
			dtype=self.dtype,
			name='head',
			**kwargs
			)
		pass

	def set_meta(self, index, bits_per_dim,
		sort_bits=None,
		permute=None,
		offset=None,
		scale=None,
		xtype='float32',
		qtype='int64',
		ftype='int64',
		**kwargs
		):
		"""
		"""
		meta = utils.Prototype(
			output_size = self.output_size,
			bits_per_dim = bits_per_dim,
			xtype = xtype,
			qtype = qtype,
			ftype = ftype,
			dtype = self.dtype,
			sort_bits = sort_bits,
			permute = permute,
			offset=offset,
			scale=scale,
			**kwargs
			)
		meta.input_dims = len(meta.bits_per_dim)
		meta.tree_depth = sum(meta.bits_per_dim)
		if isinstance(index, str) and index.endswith('.txt'):
			meta.index = index
			with open(index, 'r') as fid:
				meta.num_of_files = sum(len(L) > 0 for L in fid)
		else:
			meta.index = [f for f in utils.ifile(index)]
			meta.num_of_files = len(index)
		meta.num_of_samples = meta.num_of_files * meta.tree_depth
		self.meta = meta
		return meta
	
	def encoder(self, index, bits_per_dim, *args,
		sort_bits=None,
		permute=None,
		offset=None,
		scale=None,
		xtype='float32',
		qtype='int64',
		ftype='int64',
		shuffle=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index, bits_per_dim, sort_bits, permute, offset, scale, xtype, qtype, ftype, **kwargs)
		n_layers = meta.tree_depth+1

		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, meta.input_dims))
			X0, offset, scale = bitops.serialize(X, meta.bits_per_dim, meta.offset, meta.scale, dtype=meta.qtype)
			if meta.permute is not None:
				permute = tf.cast(meta.permute, dtype=X0.dtype)
				X0 = bitops.permute(X0, permute, meta.tree_depth)
			elif meta.sort_bits is not None:
				absolute = 'absolute' in meta.sort_bits
				reverse = 'reverse' in meta.sort_bits
				X0, permute = bitops.sort(X0, bits=meta.tree_depth, absolute=absolute, reverse=reverse)
			else:
				permute = tf.constant([], dtype=X0.dtype)
			X0 = bitops.tokenize(X0, 1, n_layers)
			X1 = tf.roll(X0, -1, 0)
			layer = tf.range(n_layers)
			X = [X] * n_layers
			permute = [permute] * n_layers
			offset = [offset] * n_layers
			scale = [scale] * n_layers
			return X0, X1, layer, X, permute, offset, scale
		
		def encode(X0, X1, layer, *args):
			uids, idx0 = tf.unique(X0)
			flags, labels = bitops.encode(X1, idx0, 1, ftype)
			uids = bitops.right_shift(uids[:,None], tf.range(meta.tree_depth, dtype=uids.dtype))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, meta.dtype)
			return (uids, flags, labels, layer, *args)
		
		def filter(uids, flags, labels, layer, *args):
			return layer < meta.tree_depth
		
		if isinstance(index, str) and index.endswith('.txt'):
			encoder = tf.data.TextLineDataset(index)
		else:
			encoder = tf.data.Dataset.from_tensor_slices(index)
		
		if shuffle:
			encoder = encoder.shuffle(shuffle)
		encoder = encoder.map(parse)
		encoder = encoder.unbatch()
		encoder = encoder.map(encode)
		encoder = encoder.filter(filter)
		return encoder, meta
	
	def trainer(self, *args,
		encoder=None,
		meta=None,
		balance=0,
		**kwargs
		):
		"""
		"""
		def filter_labels(uids, flags, labels, layer, *args):
			m = tf.range(uids.shape[-1]) <= layer
			uids = uids * 2 - tf.cast(m, self.dtype)
			uids = tf.roll(uids, uids.shape[-1]-(layer+1), axis=-1)
			labels = tf.cast(labels, self.dtype)
			labels /= tf.norm(labels, ord=1, axis=-1, keepdims=True)
			if balance:
				weights = tf.size(flags)
				weights = tf.cast(weights, self.dtype)
				weights = tf.ones_like(flags, dtype=self.dtype) - tf.math.exp(-weights/balance)
				return uids, labels, weights
			else:
				return uids, labels
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		trainer = encoder.map(filter_labels)
		trainer = trainer.batch(1)
		if meta is None:
			return trainer, encoder
		else:
			return trainer, encoder, meta
	
	def validator(self, *args,
		encoder=None,
		**kwargs
		):
		"""
		"""
		return self.trainer(*args, encoder=encoder, **kwargs)
	
	def tester(self, *args,
		encoder=None,
		**kwargs
		):
		"""
		"""
		return self.trainer(*args, encoder=encoder, **kwargs)
	
	def build(self, input_shape=None, meta=None):
		if input_shape:
			super(NbitTree, self).build(input_shape)
		else:
			self.meta = meta or self.meta
			super(NbitTree, self).build(tf.TensorShape((None, None, self.meta.tree_depth)))
	
	def call(self, inputs, training=False):
		"""
		"""
		X = inputs
		stack = [X]

		if self.conv:
			x = tf.concat([X>0, X<0], axis=-1)
			x = tf.cast(x, self.dtype)
			stack += [conv(x) for conv in self.conv]
		
		if self.transformers:
			stack += [transformer(X) for transformer in self.transformers]
		
		X = tf.concat(stack, axis=-1)
		X = self.head(X)
		return X
	
	@property
	def output_size(self):
		return 2
	
	@staticmethod
	def parse(filename, meta):
		from .. import bitops
		buffer = bitops.BitBuffer(filename)
		remains = buffer.read(5)
		while len(buffer):
	
	@staticmethod
	def decode(probs, nodes, remains, buffer=tf.constant([0], dtype=tf.int64)):
		"""
		"""
		flags = tf.reshape(probs, (-1,1))
		flags = bitops.right_shift(flags, np.arange(2))
		flags = bitops.bitwise_and(flags, 1)
		X = tf.where(flags)
		i = X[...,0]
		X = X[...,1]
		buffer = bitops.left_shift(buffer, 1)
		X = X + tf.gather(buffer, i)
		remains = remains 
		return X, remains
	
	@staticmethod
	def finalize(X, meta, permute=None, offset=None, scale=None):
		"""
		"""
		X = tf.reshape(X, (-1,1))
		if permute is not None:
			X = bitops.permute(X, permute, meta.tree_depth)
		X = bitops.realize(X, meta.bits_per_dim, offset, scale, meta.xtype)
		return X
		