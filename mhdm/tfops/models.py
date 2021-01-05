

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LayerNormalization, Concatenate

## Local
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
			dtype=dtype,
			normalize=normalize,
			layer_type=Dense,
			**kwargs
			) for i in range(transformers)]
		
		self.concatenate = Concatenate()
			
		#self.normalization = LayerNormalization()
		self.output_layer = Dense(
			self.output_size,
			#activation='relu',
			dtype=dtype,
			name='output_layer',
			**kwargs
			)
		pass
	
	def encoder(self, filenames, bits_per_dim, *args,
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
			permute = permute
			)
		
		meta.input_dims = len(meta.bits_per_dim)
		meta.word_length = sum(meta.bits_per_dim)
		meta.tree_depth = meta.word_length // meta.dim + (meta.word_length % meta.dim != 0)
		n_layers = meta.tree_depth+1

		def parse(filename):
			X0 = tf.io.read_file(filename)
			X0 = tf.io.decode_raw(X0, xtype)
			X0 = tf.reshape(X0, (-1, meta.input_dims))
			X0, _offset, _scale = bitops.serialize(X0, bits_per_dim, offset, scale)
			if permute is not None:
				X0 = bitops.permute(X0, permute, meta.word_length)
			X0 = bitops.tokenize(X0, meta.dim, n_layers)
			X1 = tf.roll(X0, -1, 0)
			layer = tf.range(n_layers, dtype=X0.dtype)
			_offset = tf.repeat(_offset[:,None], n_layers, axis=0)
			_scale = tf.repeat(_scale[:,None], n_layers, axis=0)
			return X0, X1, layer, _offset, _scale
		
		def encode(X0, X1, layer, *args):
			uids, idx0 = tf.unique(X0, out_idx=X0.dtype)
			flags, idx1, _ = bitops.encode(X1, idx0, meta.dim, ftype)
			labels = tf.one_hot(flags, meta.output_size)
			uids = bitops.left_shift(uids, meta.tree_depth-layer*meta.dim)
			uids = bitops.right_shift(uids[:,None], np.arange(meta.word_length))
			uids = bitops.bitwise_and(uids, 1)
			uids = tf.cast(uids, meta.dtype)
			return (uids, labels, flags, layer, *args)
		
		if isinstance(filenames, str) and filenames.endswith('.txt'):
			encoder = tf.data.TextLineDataset(filenames)
		else:
			encoder = tf.data.Dataset.from_tensor_slices([f for f in ifile(filenames)])
		encoder = encoder.map(parse)
		encoder = encoder.unbatch()
		encoder = encoder.map(encode)
		return encoder, meta
	
	def trainer(self, *args, encoder=None, **kwargs):
		"""
		"""
		def filter(uids, labels, *args):
			return uids, labels
	
		if encoder is None:
			encoder = self.encoder(*args, **kwargs)
		encoder = encoder.map(filter)
		encoder = encoder.batch(1)
		return encoder
	
	def call(self, inputs, training=False):
		"""
		"""
		X = [t(inputs) for t in self.transformers]
		X = self.concatenate(X)
		#X = self.normalization(X)
		X = self.output_layer(X)
		X -= tf.math.reduce_min(X - 1.e-16)
		X /= tf.math.reduce_max(X + 1.e-16)
		return X
	
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
		i = X[:,0]
		X = X[:,1]
		buffer = bitops.left_shift(buffer, meta.dim)
		X = X + tf.gather(buffer, i)
		return X
	
	@staticmethod
	def finalize(X, meta):
		"""
		"""
		if meta.permute is not None:
			X = bitops.permute(X[:,None], meta.permute, meta.word_length)
		X = bitops.realize(X, meta.bits_per_dim, meta.offset, meta.scale, meta.xtype)
		return X
		