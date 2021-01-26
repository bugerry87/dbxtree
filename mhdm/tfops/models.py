
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Concatenate, Conv1D
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
		k=None,
		transformers=1,
		convolutions=0,
		normalize=False,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTreeProbEncoder, self).__init__(name=name, **kwargs)
		self.dim = dim
		self.kernel_size = k if k else self.output_size
		
		self.transformers = [layers.Transformer(
			units=self.kernel_size,
			normalize=normalize,
			layer_types=layers.Dense,
			kernel_initializer='random_normal',
			dtype=dtype,
			name='transformer_{}'.format(i),
			**kwargs
			) for i in range(transformers)]
		if transformers > 1:
			self.concatenate = Concatenate()
		#if transformers > 0:
			#self.norm_trans = LayerNormalization(name='norm')

		self.convolutions = [Conv1D(
			self.kernel_size, 3,
			#kernel_initializer='random_normal',
			#kernel_regularizer='l2',
			activation='relu',
			padding='same',
			name='conv1d_{}'.format(i),
			**kwargs
			) for i in range(convolutions)]

		self.output_layer = layers.Dense(
			self.output_size,
			activation='softmax',
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
			X0 = tf.io.read_file(filename)
			X0 = tf.io.decode_raw(X0, xtype)
			X0 = tf.reshape(X0, (-1, meta.input_dims))
			X0, offset, scale = bitops.serialize(X0, meta.bits_per_dim, meta.offset, meta.scale)
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
			permute = tf.repeat(permute[None,...], n_layers, axis=0)
			offset = tf.repeat(offset[None,...], n_layers, axis=0)
			scale = tf.repeat(scale[None,...], n_layers, axis=0)
			return X0, X1, layer, permute, offset, scale
		
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
		relax=10000,
		smoothing=0,
		**kwargs
		):
		"""
		"""
		def filter_labels(uids, flags, layer, *args):
			mask = tf.range(uids.shape[-1]) <= tf.cast(layer * self.dim, tf.int32)
			uids = uids * 2 - tf.cast(mask, tf.float32)
			weights = tf.size(flags)
			weights = tf.cast(weights, tf.float32)
			weights = tf.ones_like(flags, dtype=tf.float32) - tf.math.exp(-weights/relax)
			labels = tf.one_hot(flags, self.output_size)
			if smoothing:
				labels *= 1.0 - smoothing
				labels += smoothing / 2
			return uids, labels, weights
		
		def filter_args(uids, *args):
			return (*args,)
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = None
		trainer = encoder.map(filter_labels)
		trainer = trainer.batch(1)
		trainer_args = encoder.map(filter_args)
		if meta is None:
			return trainer, trainer_args
		else:
			return trainer, trainer_args, meta
	
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
	
	def call(self, inputs, training=False):
		"""
		"""
		X = inputs
		X = tf.concat([X>0, X<0], axis=-1)
		X = tf.cast(X, self.dtype)
		if len(self.transformers) > 0:
			X = [t(X) for t in self.transformers]
			if len(self.transformers) > 1:
				X = self.concatenate(X)
			else:
				X = X[0]
			#X = self.norm_trans(X)
		
		for conv in self.convolutions:
			X = conv(X)
		
		X = self.output_layer(X)
		#X = X**2
		#X = tf.math.exp(-X / tf.math.reduce_max(X+1, axis=-1, keepdims=True))
		#X /= tf.math.reduce_max(X, axis=-1, keepdims=True)
		return X
	
	def predict_step(self, data):
		"""
		"""
		if range_encoder is None:
			tf.get_logger().warn(
				"Model has no range_encoder and will only return raw probabilities and an empty string. " \
				"Please install 'tensorflow-compression' to obtain encoded bit-streams."
				)
			X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
			pred, uids, probs, flags = X
			probs = self(uids, training=False)[0]
			return probs, tf.constant([''])
		
		def encode():
			_probs = tf.roll(probs, -1, axis=-1) / tf.math.reduce_max(probs + 0.0625, axis=-1, keepdims=True) + 0.0625
			cdf = tf.math.cumsum(_probs, axis=-1, exclusive=True)
			cdf = cdf / tf.math.reduce_max(cdf, axis=-1, keepdims=True) * float(1<<16) 
			cdf = tf.cast(cdf, tf.int32)
			index = range_like(flags, dtype=tf.int32)
			cdf_size = tf.zeros_like(flags, dtype=tf.int32) + cdf.shape[-1]
			offset = tf.ones_like(flags, dtype=tf.int32)
			code = range_encoder.unbounded_index_range_encode(
				flags, index, cdf, cdf_size, offset,
				precision=16,
				overflow_width=4
				)
			return tf.expand_dims(code, axis=0)
		
		def ignore():
			return tf.constant([''])
		
		X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
		pred, uids, probs, flags = X
		flags = tf.cast(flags, tf.int32)
		probs = tf.concat([probs, self(uids, training=False)[0]], axis=0)
		code = tf.cond(pred, encode, ignore)
		return probs, code

	@property
	def flag_size(self):
		return 1<<self.dim
	
	@property
	def output_size(self):
		return 1<<(self.flag_size)
	
	@staticmethod
	def decode(flags, meta,
		buffer=tf.constant([0], dtype=tf.int64)
		):
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
		