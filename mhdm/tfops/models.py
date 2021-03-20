
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

## Optional
try:
	import tensorflow_compression as tfc
except ModuleNotFoundError:
	tfc = None


class NbitTree(Model):
	"""
	"""
	def __init__(self,
		dim=2,
		kernels=None,
		kernel_size=3,
		convolutions=0,
		transformers=0,
		floor=0.0,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTree, self).__init__(name=name, **kwargs)
		self.mode = dim
		self.dim = max(dim, 1)
		self.kernels = kernels or self.output_size
		self.kernel_size = kernel_size
		self.floor = floor

		self.transformers = [layers.InnerTransformer(
			self.kernels,
			dtype=self.dtype,
			layer_types=[Dense, Dense, layers.Euclidean],
			layer_A_args=dict(
				kernel_initializer='random_normal'
				),
			layer_B_args=dict(
				kernel_initializer='random_normal',
				trainable=False
				),
			layer_t_args=dict(inverted=True),
			name='transformer_{}'.format(i),
			**kwargs
			) for i in range(transformers)]

		self.conv = [Conv1D(
			self.kernels, self.kernel_size + 2*i, 1,
			activation='relu',
			padding='same',
			dtype=self.dtype,
			name='conv_{}'.format(i),
			**kwargs
			) for i in range(convolutions)]

		self.head = Dense(
			self.output_size,
			activation='softplus',
			dtype=self.dtype,
			name='head',
			**kwargs
			)
		pass
	
	@property
	def flag_size(self):
		return 1<<self.dim
	
	@property
	def bins(self):
		return 2

	@property
	def output_size(self):
		return self.bins

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
			dim = self.dim,
			mode = self.mode,
			output_size = self.output_size,
			flag_size = self.flag_size,
			bins = self.bins,
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
			X0 = bitops.tokenize(X0, meta.dim, n_layers)
			X1 = tf.roll(X0, -1, 0)
			layer = tf.range(n_layers, dtype=X0.dtype)
			filename = [filename] * n_layers
			permute = [permute] * n_layers
			offset = [offset] * n_layers
			scale = [scale] * n_layers
			return X0, X1, layer, filename, permute, offset, scale
		
		def encode(X0, X1, layer, filename, permute, offset, scale):
			uids, idx0 = tf.unique(X0)
			flags, hist = bitops.encode(X1, idx0, meta.dim, ftype)
			pos = NbitTree.finalize(uids, meta, permute, offset, scale)
			pos_max = tf.math.reduce_max(tf.math.abs(pos), axis=0, keepdims=True)
			pos = tf.math.divide_no_nan(pos, pos_max)
			uids = bitops.right_shift(uids[:,None], tf.range(meta.tree_depth, dtype=uids.dtype))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, meta.dtype)
			return (uids, pos, flags, hist, layer, filename, permute, offset, scale)
		
		def filter(uids, pos, flags, hist, layer, *args):
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
		def feature_label_filter(uids, pos, flags, hist, layer, *args):
			m = tf.range(uids.shape[-1], dtype=layer.dtype) <= layer
			uids = uids * 2 - tf.cast(m, self.dtype)
			feature = tf.concat((uids, pos), axis=-1)
			labels = tf.math.argmin(hist, axis=-1)
			labels = tf.one_hot(labels, self.bins, dtype=self.dtype)
			if balance:
				weights = tf.size(counts)
				weights = tf.cast(weights, self.dtype)
				weights = tf.zeros_like(counts, dtype=self.dtype) + tf.math.exp(-weights/balance)
				return feature, labels, weights
			else:
				return feature, labels
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		trainer = encoder.map(feature_label_filter)
		trainer = trainer.batch(1)
		if meta is None:
			return trainer, encoder
		else:
			meta.feature_size = meta.tree_depth + meta.input_dims
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
		"""
		"""
		if input_shape:
			super(NbitTree, self).build(input_shape)
		else:
			self.meta = meta or self.meta
			super(NbitTree, self).build(tf.TensorShape((None, None, self.meta.feature_size)))
	
	def call(self, inputs, training=False):
		"""
		"""
		X = inputs

		if self.transformers:
			X = tf.concat([transformer(X) for transformer in self.transformers], axis=-1)
		else:
			Xmin = tf.math.minimum(X, 0.0)
			Xmax = tf.math.maximum(X, 0.0)
			X = tf.concat([Xmin, Xmax], axis=-1)

		C = X
		for conv in self.conv:
			x = conv(C)
			C = tf.concat([X, x], axis=-1)
		X = C
		
		X = self.head(X)
		return X
	
	def predict_step(self, data):
		"""
		"""
		if self.mode <= 0:
			X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
			feature, hist = X
			probs = tf.reshape(self(feature, training=False), (-1, self.bins))
			return probs

		def encode():
			if tfc is None:
				tf.get_logger().warn(
					"Model has no range_encoder and will only return raw probabilities and an empty string. " \
					"Please install 'tensorflow-compression' to obtain encoded bit-streams."
					)
				return tf.constant([''])

			cdf = tf.math.reduce_max(probs, axis=-1, keepdims=True)
			cdf = tf.math.divide_no_nan(probs, cdf)[:,1:]
			cdf = tf.clip_by_value(cdf, self.floor, 1.0)
			cdf = tf.math.cumsum(cdf, axis=-1)
			cdf /= tf.math.reduce_max(cdf, axis=-1, keepdims=True)
			cdf = tf.math.round(cdf * float(1<<16))
			cdf = tf.cast(cdf, tf.int32)
			cdf = tf.pad(cdf, [(0,0),(1,0)])
			code = tfc.range_encode(flags-1, cdf, precision=16)
			return tf.expand_dims(code, axis=0)
		
		def ignore():
			return tf.constant([''])
		
		X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
		feature, probs, flags, do_encode = X
		flags = tf.cast(flags, tf.int16)
		pred = tf.reshape(self(feature, training=False), (-1, self.bins))
		probs = tf.concat([probs, pred], axis=0)
		code = tf.cond(do_encode, encode, ignore)
		return probs, code

	@staticmethod
	def parse(self, filename):
		from ..bitops import BitBuffer
		buffer = BitBuffer(filename)
		remains = np.array([buffer.read(5)])
		while len(buffer):
			counts = [buffer.read(r) for r in remains]
			remains = remains - counts
			remains = np.hstack([remains, counts]).T
			flags = remains > 0
			remains = remains[flags]
			yield flags
	
	@staticmethod
	def decode(flags, meta, X=tf.constant([0], dtype=tf.int64)):
		"""
		"""
		shifts = tf.range(meta.flag_size, dtype=X.dtype)
		flags = tf.cast(flags, dtype=X.dtype)
		flags = tf.reshape(flags, (-1, 1))
		flags = bitops.right_shift(flags, shifts)
		flags = bitops.bitwise_and(flags, 1)
		x = tf.where(flags)
		i = x[...,0]
		x = x[...,1]
		X = bitops.left_shift(X, meta.dim)
		X = x + tf.gather(X, i)
		return X
	
	@staticmethod
	def finalize(X, meta, permute=None, offset=None, scale=None):
		"""
		"""
		X = tf.reshape(X, (-1,1))
		if permute is not None and permute.shape[0]:
			X = bitops.permute(X, permute, meta.tree_depth)
		X = bitops.realize(X, meta.bits_per_dim, offset, scale, meta.xtype)
		return X