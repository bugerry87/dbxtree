
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import range_like
from . import bitops
from . import spatial
from .. import utils

## Optional
try:
	import tensorflow_compression as tfc
except ModuleNotFoundError:
	tfc = None


class NbitTree(Model):
	"""
	"""
	def __init__(self, dim,
		heads=1,
		kernels=16,
		kernel_size=3,
		convolutions=4,
		branches=('uids', 'pos', 'voxels', 'meta'),
		dense=0,
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
		self.kernels = kernels
		self.kernel_size = kernel_size
		self.floor = floor

		if 'uids' in branches:
			self.conv_uids = [Conv1D(
				self.kernels, self.kernel_size, 1,
				activation='relu',
				padding='same',
				dtype=self.dtype,
				name='conv_uids_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
		else:
			self.conv_uids = []
		
		if 'pos' in branches:
			self.conv_pos = [Conv1D(
				self.kernels, self.kernel_size, 1,
				activation='relu',
				padding='same',
				dtype=self.dtype,
				name='conv_pos_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
		else:
			self.conv_pos = []
		
		if 'voxels' in branches:
			self.conv_voxels = [Conv1D(
				self.kernels, self.kernel_size, 1,
				activation='relu',
				padding='same',
				dtype=self.dtype,
				name='conv_voxels_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
		else:
			self.conv_voxels = []
		
		if 'meta' in branches:
			self.conv_meta = [Conv1D(
				self.kernels, self.kernel_size, 1,
				activation='relu',
				padding='same',
				dtype=self.dtype,
				name='conv_meta_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
		else:
			self.conv_meta = []
		
		self.dense = [Dense(
			self.kernels,
			activation='relu',
			dtype=self.dtype,
			name='dense_{}'.format(i),
			**kwargs
			) for i in range(dense)]


		self.heads = [Dense(
			self.bins,
			activation='softplus',
			dtype=self.dtype,
			name='head',
			**kwargs
			) for i in range(heads)]
		pass
	
	@property
	def flag_size(self):
		return 1<<self.dim
	
	@property
	def bins(self):
		if self.mode > 0:
			return 1<<self.flag_size
		else:
			return 2

	@property
	def output_shape(self):
		return (len(self.heads), self.bins)

	def set_meta(self, index, bits_per_dim,	**kwargs):
		"""
		"""
		meta = utils.Prototype(
			bits_per_dim=bits_per_dim,
			dim=self.dim,
			mode=self.mode,
			output_shape=self.output_shape,
			flag_size=self.flag_size,
			bins=self.bins,
			dtype = self.dtype,
			**kwargs
			)
		meta.input_dims = len(meta.bits_per_dim)
		meta.word_length = sum(meta.bits_per_dim)
		meta.tree_depth = int(np.ceil(meta.word_length / self.dim))
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
		payload=False,
		spherical=False,
		xtype='float32',
		qtype='int64',
		ftype='int64',
		shuffle=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index, bits_per_dim,
			sort_bits=sort_bits,
			permute=permute,
			payload=payload,
			spherical=spherical,
			offset=offset,
			scale=scale,
			xtype=xtype,
			qtype=qtype,
			ftype=ftype,
			**kwargs
			)

		@tf.function
		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, meta.input_dims))
			if meta.spherical:
				X = spatial.xyz2uvd(X)
			X, offset, scale = bitops.serialize(X, meta.bits_per_dim, meta.offset, meta.scale, dtype=meta.qtype)
			if meta.permute is not None:
				permute = tf.cast(meta.permute, dtype=X.dtype)
				X = bitops.permute(X, permute, meta.word_length)
			elif meta.sort_bits is not None:
				absolute = 'absolute' in meta.sort_bits
				reverse = 'reverse' in meta.sort_bits
				X, permute = bitops.sort(X, bits=meta.word_length, absolute=absolute, reverse=reverse)
			else:
				permute = tf.constant([], dtype=X.dtype)
			X = bitops.tokenize(X, meta.dim, meta.tree_depth+1)
			X0 = X[:-1]
			X1 = X[1:]
			layer = tf.range(meta.tree_depth, dtype=X.dtype)
			filename = [filename] * meta.tree_depth
			permute = [permute] * meta.tree_depth
			offset = [offset] * meta.tree_depth
			scale = [scale] * meta.tree_depth
			if payload:
				mask = [tf.ones_like(X0[0], dtype=bool)] + [tf.gather(*tf.unique_with_counts(X0[i])[-1:0:-1]) > 1 for i in range(meta.tree_depth-1)]
			else:
				mask = [tf.ones_like(X0[0], dtype=bool)] * meta.tree_depth
			return X0, X1, layer, filename, permute, offset, scale, mask
		
		@tf.function
		def encode(X0, X1, layer, filename, permute, offset, scale, mask):
			uids, idx, counts = tf.unique_with_counts(X0[mask])
			flags, hist = bitops.encode(X1[mask], idx, meta.dim, ftype)
			if meta.payload:
				flags *= tf.cast(counts > 1, flags.dtype)
			pos = NbitTree.finalize(uids, meta, permute, offset, scale)
			pos_max = tf.math.reduce_max(tf.math.abs(pos), axis=0, keepdims=True)
			pos = tf.math.divide_no_nan(pos, pos_max)
			return (uids, pos, flags, counts, hist, layer, filename, permute, offset, scale, X0, mask)
		
		if isinstance(index, str) and index.endswith('.txt'):
			encoder = tf.data.TextLineDataset(index)
		else:
			encoder = tf.data.Dataset.from_tensor_slices(index)
		
		if shuffle:
			encoder = encoder.shuffle(shuffle)
		encoder = encoder.map(parse)
		encoder = encoder.unbatch()
		encoder = encoder.map(encode)
		return encoder, meta
	
	def trainer(self, *args,
		encoder=None,
		meta=None,
		**kwargs
		):
		"""
		"""
		def feature_label_filter(uids, pos, flags, counts, hist, layer, *args):
			counts = tf.cast(counts[...,None], self.dtype)

			half = self.kernels//2
			kernel = tf.concat([
				tf.eye(self.kernels, dtype=tf.float64)[:half],
				-tf.ones([1, self.kernels], dtype=tf.float64),
				tf.eye(self.kernels, dtype=tf.float64)[half:]
				], axis=0)
			
			voxels = tf.concat([tf.roll(uids, half-i, 0)[...,None] for i in range(self.kernels+1)], axis=-1)
			voxels = tf.cast(voxels, kernel.dtype)
			voxels = voxels @ kernel - tf.cast(tf.roll(tf.range(self.kernels), half, 0), kernel.dtype)
			voxels = tf.cast(voxels, self.dtype)
			voxels = tf.math.divide_no_nan(2*voxels, tf.reduce_mean(tf.abs(voxels)))
			voxels = tf.exp(-voxels*voxels)

			uids = bitops.right_shift(uids[...,None], tf.range(meta.word_length, dtype=uids.dtype))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, self.dtype)
			m = tf.range(uids.shape[-1], dtype=layer.dtype) <= layer
			uids = uids * 2 - tf.cast(m, self.dtype)
			uids = tf.concat([tf.math.minimum(uids, 0.0), tf.math.maximum(uids, 0.0)], axis=-1)
			pos = tf.concat([tf.math.minimum(pos, 0.0), tf.math.maximum(pos, 0.0)], axis=-1)

			if self.mode > 0:
				labels = tf.one_hot(flags, self.bins, dtype=self.dtype)
			elif self.mode == 0:
				labels = tf.math.argmax(hist, axis=-1)
				labels = tf.one_hot(labels, self.bins, dtype=self.dtype)
			else:
				labels = tf.cast(hist, self.dtype) / counts
			
			labels = tf.repeat(labels[...,None,:], len(self.heads), axis=-2)
			counts /= tf.math.reduce_sum(counts)
			feature = tf.concat((uids, pos, voxels, counts), axis=-1)

			weights = tf.one_hot(layer, len(self.heads), dtype=self.dtype)
			weights = tf.repeat(weights[...,None], self.bins, axis=-1)
			return feature, labels, weights
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		trainer = encoder.map(feature_label_filter)
		trainer = trainer.batch(1)
		if meta is None:
			return trainer, encoder
		else:
			meta.feature_size = meta.word_length*2 + meta.input_dims*2 + self.kernels + 1
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
		def incr(j):
			incr.i += j
			return incr.i
		incr.i = 0

		X = inputs
		stack = [X]
		uids = X[...,incr(0):incr(self.meta.word_length*2)]
		pos = X[...,incr(0):incr(self.meta.input_dims*2)]
		voxels = X[...,incr(0):incr(self.kernels)]
		meta = X[..., incr(0):]
		
		if self.conv_uids:
			X = uids
			for conv in self.conv_uids:
				x = conv(X)
				X = tf.concat((X, x), axis=-1)
			stack.append(X)

		if self.conv_pos:
			X = pos
			for conv in self.conv_pos:
				x = conv(X)
				X = tf.concat((X, x), axis=-1)
			stack.append(X)

		if self.conv_voxels:
			X = voxels
			for conv in self.conv_voxels:
				x = conv(X)
				X = tf.concat((X, x), axis=-1)
			stack.append(X)

		if self.mode <= 0 and self.conv_meta:
			X = meta
			for conv in self.conv_meta:
				x = conv(X)
				X = tf.concat((X, x), axis=-1)
			stack.append(X)

		X = tf.concat(stack, axis=-1)
		for dense in self.dense:
			X = dense(X)
		
		X = [head(X)[...,None,:] for head in self.heads]
		return tf.concat(X, axis=-2)
	
	def predict_step(self, data):
		"""
		"""
		if self.mode <= 0:
			X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
			feature, layer = X[:2]
			probs = tf.reshape(self(feature, training=False)[...,layer,:], (-1, self.bins))
			return probs

		def encode():
			if tfc is None:
				tf.get_logger().warn(
					"Model has no range_encoder and will only return raw probabilities and an empty string. " \
					"Please install 'tensorflow-compression' to obtain encoded bit-streams."
					)
				return tf.constant([''])
			
			cdf = probs[...,1:]
			symbols = tf.cast(labels-1, tf.int16)
			
			pmax = tf.math.reduce_max(cdf, axis=-1, keepdims=True)
			cdf = tf.math.divide_no_nan(cdf, pmax)
			cdf = tf.clip_by_value(cdf, self.floor, 1.0)
			cdf = tf.math.cumsum(cdf, axis=-1)
			cdf /= tf.math.reduce_max(cdf, axis=-1, keepdims=True)
			cdf = tf.math.round(cdf * float(1<<16))
			cdf = tf.cast(cdf, tf.int32)
			cdf = tf.pad(cdf, [(0,0),(1,0)])
			code = tfc.range_encode(symbols, cdf, precision=16)
			return tf.expand_dims(code, axis=0)
		
		def ignore():
			return tf.constant([''])
		
		X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
		feature, layer, probs, labels, do_encode = X
		pred = tf.reshape(self(feature, training=False)[...,layer,:], (-1, self.bins))
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
			X = bitops.permute(X, permute, meta.word_length)
		X = bitops.realize(X, meta.bits_per_dim, offset, scale, meta.xtype)
		if meta.spherical:
			X = spatial.uvd2xyz(X)
		return X