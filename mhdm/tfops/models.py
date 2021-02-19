
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Conv1D, LSTM
from tensorflow.python.keras.engine import data_adapter

try:
	import tensorflow_compression as range_encoder
except ModuleNotFoundError:
	range_encoder = None

## Local
from . import range_like
from . import bitops
from . import layers
from .. import utils


class NbitTreeProbEncoder(Model):
	"""
	"""
	def __init__(self,
		dim=3,
		kernel=None,
		kernel_width=3,
		strides=3,
		convolutions=0,
		unet=False,
		transformer=False,
		tensorflow_compression=False,
		floor=0.0,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTreeProbEncoder, self).__init__(name=name, **kwargs)
		self.dim = dim
		self.kernel = kernel if kernel else self.output_size
		self.kernel_width = kernel_width
		self.strides = strides
		self.unet = unet
		self.tensorflow_compression = tensorflow_compression
		self.floor = floor

		if unet:
			self.conv_down = [Conv1D(
				self.kernel, 3, 1,
				activation='relu',
				padding='same',
				name='conv_down_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
			self.conv_up = [Conv1D(
				self.kernel, 3, 1,
				activation='relu',
				padding='same',
				name='conv_up_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
		
		if transformer:
			self.conv_strid = [Conv1D(
				self.kernel, strides, strides,
				activation='relu',
				padding='same',
				name='conv_strid_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]
			self.ABt = [layers.Dense(
				self.kernel,
				activation='relu',
				dtype=dtype,
				name=n,
				**kwargs
				) for n in 'ABt']
			self.transformer = layers.OuterTransformer(layer_type=None)
		else:
			self.transformer = None
		
		if not unet and not transformer:
			self.conv = [Conv1D(
				self.kernel, kernel_width, 1,
				activation='relu',
				padding='same',
				name='conv_{}'.format(i),
				**kwargs
				) for i in range(convolutions)]

		self.output_layer = layers.Dense(
			self.output_size,
			activation='softplus',
			dtype=dtype,
			name='output_layer',
			**kwargs
			)
		pass

	def set_meta(self, index, bits_per_dim,
		sort_bits=None,
		permute=None,
		offset=None,
		scale=None,
		xtype='float32',
		ftype='uint8',
		**kwargs
		):
		"""
		"""
		meta = utils.Prototype(
			dim = self.dim,
			flag_size = self.flag_size,
			bins = self.bins,
			output_size = self.output_size,
			bits_per_dim = bits_per_dim,
			xtype = xtype,
			ftype = ftype,
			dtype = self.dtype,
			sort_bits = sort_bits,
			permute = permute,
			offset=offset,
			scale=scale,
			**kwargs
			)
		meta.input_dims = len(meta.bits_per_dim)
		meta.word_length = sum(meta.bits_per_dim)
		meta.tree_depth = meta.word_length // meta.dim + (meta.word_length % meta.dim != 0)
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
		ftype='uint8',
		shuffle=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index, bits_per_dim, sort_bits, permute, offset, scale, xtype, ftype, **kwargs)
		n_layers = meta.tree_depth+1

		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, meta.input_dims))
			X0, offset, scale = bitops.serialize(X, meta.bits_per_dim, meta.offset, meta.scale)
			if meta.permute is not None:
				permute = tf.cast(meta.permute, dtype=X0.dtype)
				X0 = bitops.permute(X0, permute, meta.word_length)
			elif meta.sort_bits is not None:
				absolute = 'absolute' in meta.sort_bits
				reverse = 'reverse' in meta.sort_bits
				X0, permute = bitops.sort(X0, bits=meta.word_length, absolute=absolute, reverse=reverse)
			else:
				permute = tf.constant([], dtype=X0.dtype)
			X0 = bitops.tokenize(X0, meta.dim, n_layers)
			X1 = tf.roll(X0, -1, 0)
			layer = tf.range(n_layers, dtype=X0.dtype)
			X = tf.repeat(X[None,...], n_layers, axis=0)
			permute = tf.repeat(permute[None,...], n_layers, axis=0)
			offset = tf.repeat(offset[None,...], n_layers, axis=0)
			scale = tf.repeat(scale[None,...], n_layers, axis=0)
			return X0, X1, layer, X, permute, offset, scale
		
		def encode(X0, X1, layer, *args):
			uids, idx0 = tf.unique(X0, out_idx=X0.dtype)
			flags, idx1, _ = bitops.encode(X1, idx0, meta.dim, ftype)
			uids = bitops.left_shift(uids, meta.tree_depth-layer*meta.dim)
			uids = bitops.right_shift(uids[:,None], np.arange(meta.word_length))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, meta.dtype)
			return (uids, flags, layer, *args)
		
		def filter(uids, flags, layer, *args):
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
		relax=100000,
		smoothing=0,
		mask=None,
		**kwargs
		):
		"""
		"""
		def filter_labels(uids, flags, layer, *args):
			m = tf.range(uids.shape[-1]) <= tf.cast(layer * self.dim, tf.int32)
			uids = uids * 2 - tf.cast(m, tf.float32)
			weights = tf.size(flags)
			weights = tf.cast(weights, tf.float32)
			weights = tf.ones_like(flags, dtype=tf.float32) - tf.math.exp(-weights/relax)
			labels = tf.one_hot(flags, self.bins)
			if smoothing:
				if mask is not None:
					layer_flags = tf.reduce_sum(labels, axis=-2, keepdims=True)
					layer_flags /= tf.reduce_max(layer_flags)
					if isinstance(mask, tf.Variable):
						mask.scatter_nd_add(layer[..., None, None], layer_flags)
						label_flags = mask[layer] / tf.reduce_max(mask)
					labels += layer_flags * smoothing
				else:
					labels *= 1.0 - smoothing
					labels += smoothing / 2
			return uids, labels, weights
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = None
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
	
	def build_by_meta(self, meta):
		if self.transformer:
			dummy = tf.ones(meta.word_length)[None, None, ...]
			self(dummy)
		else:
			self.build(tf.TensorShape([1, None, meta.word_length]))
	
	def call(self, inputs, training=False):
		"""
		"""
		X = inputs
		X = tf.concat([X>0, X<0], axis=-1)
		X = tf.cast(X, self.dtype)
		stack = [X]

		if self.unet:
			for conv in self.conv_down:
				X = conv(X)
				stack.append(X)
			for conv, Z in zip(self.conv_up, stack):
				Z = tf.concat((X,Z), axis=-1)
				X = conv(Z)
			X = tf.concat((X,stack[0]), axis=-1)
		
		if self.transformer:
			def branch(i, x):
				for conv in self.conv_strid[:i+1]:
					x = conv(x)
				return lambda: x
			
			convs = len(self.conv_strid)
			offset = inputs.shape[-1] / self.dim - convs
			i = tf.math.reduce_sum(tf.math.abs(inputs), axis=-1)
			i = tf.math.reduce_max(i) / self.dim - offset
			i = tf.clip_by_value(i, 0, convs-1)
			i = tf.cast(i, tf.int32)
			branches = [branch(b, X) for b in range(convs)]
			X = tf.switch_case(i, branches)
			ABt = [ABt(x) for ABt, x in zip(self.ABt, [X, stack[0], X])]
			X = self.transformer(ABt)
			pass

		if not self.unet and not self.transformer:
			for conv in self.conv:
				X = conv(X)
		
		X = self.output_layer(X)
		return X
	
	def predict_step(self, data):
		"""
		"""
		if self.tensorflow_compression and range_encoder is None:
			tf.get_logger().warn(
				"Model has no range_encoder and will only return raw probabilities and an empty string. " \
				"Please install 'tensorflow-compression' to obtain encoded bit-streams."
				)
			self.tensorflow_compression = False
		
		if not self.tensorflow_compression:
			X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
			do_encode, uids, probs, flags = X
			probs = self(uids, training=False)[0]
			return probs, tf.constant([''])
		
		def encode():
			cdf = probs[:,1:]
			cdf /= tf.norm(cdf, ord=1, axis=-1, keepdims=True)
			cdf = tf.math.cumsum(cdf + self.floor, axis=-1)
			cdf /= tf.math.reduce_max(cdf, axis=-1, keepdims=True)
			cdf = tf.cast(cdf * float(1<<16), tf.int32)
			cdf = tf.pad(cdf, [(0,0),(1,0)])
			data = tf.cast(flags-1, tf.int16)
			code = range_encoder.range_encode(data, cdf, precision=16)
			return tf.expand_dims(code, axis=0)
		
		def ignore():
			return tf.constant([''])
		
		X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
		do_encode, uids, probs, flags = X
		flags = tf.cast(flags, tf.int32)
		probs = tf.concat([probs, self(uids, training=False)[0]], axis=0)
		code = tf.cond(do_encode, encode, ignore)
		return probs, code

	@property
	def flag_size(self):
		return 1<<self.dim
	
	@property
	def bins(self):
		return 1<<(self.flag_size)
	
	@property
	def output_size(self):
		return self.bins
	
	@staticmethod
	def decode(flags, meta, buffer=tf.constant([0], dtype=tf.int64)):
		"""
		"""
		flags = tf.reshape(flags, (-1,1))
		flags = bitops.right_shift(flags, np.arange(1<<meta.dim))
		flags = bitops.bitwise_and(flags, 1)
		X = tf.where(flags)
		i = X[...,0]
		X = X[...,1]
		buffer = bitops.left_shift(buffer, meta.dim)
		X = X + tf.gather(buffer, i)
		return X
	
	@staticmethod
	def finalize(X, meta, permute=None, offset=None, scale=None):
		"""
		"""
		X = tf.reshape(X, (-1,1))
		if permute is not None:
			X = bitops.permute(X, permute, meta.word_length)
		X = bitops.realize(X, meta.bits_per_dim, offset, scale, meta.xtype)
		return X
		