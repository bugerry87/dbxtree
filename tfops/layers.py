
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Permute, Dot

## Local
from . import bitops


class NbitTreeEncoder(Layer):
	"""
	"""
	def __init__(self, dim, bits_per_dim,
		offset=None,
		scale=None,
		permute=True,
		absolute=True,
		reverse=False,
		dtype='uint8',
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTreeEncoder, self).__init__(
			name=name if name else "{}bitTreeEncoder".format(dim),
			trainable=False,
			dtype=dtype,
			**kwargs
			)
		self.dim = dim
		self.bits_per_dim = bits_per_dim
		self.offset=offset
		self.scale=scale
		self.permute = permute
		self.absolute = absolute
		self.reverse = reverse
		pass
	
	@property
	def fbits(self):
		return 1<<self.dim
	
	@property
	def xdim(self):
		return len(self.bits_per_dim)
	
	@property
	def word_length(self):
		return sum(self.bits_per_dim)
	
	@property
	def tree_depth(self):
		return int(np.ceil(self.word_length / self.dim))
	
	def __call__(self, X):
		X, offset, scale = bitops.serialize(X, self.bits_per_dim, offset=self.offset, scale=self.scale)
		
		if self.permute is True or self.absolute:
			X, permute = bitops.sort(X, self.word_length, self.reverse, self.absolute)
		elif self.permute is False or self.permute is None:
			if self.reverse:
				permute = tf.range(self.word_length, dtype=X.dtype)[::-1]
				X = bitops.permute(X, permute, self.word_length)
			else:
				permute = tf.range(self.word_length, dtype=X.dtype)
		else:
			permute = self.permute[::-1] if self.reverse else self.permute
			X = bitops.permute(X, permute, self.word_length)
		
		cond = lambda *args: True
		nodes = bitops.tokenize(X, self.dim, self.tree_depth)
		idx = tf.zeros_like(nodes[0], name='idx')
		flags = tf.constant([], dtype=self.dtype, name='flags')
		layer = tf.constant(0, name='layer')
		
		def body(idx, flags, layer):
			flags, idx, uids = bitops.encode(nodes[layer], idx, self.dim, self.dtype, flags)
			return idx, flags, layer+1
		
		idx, flags, layer = tf.while_loop(
			cond, body,
			loop_vars=(idx, flags, layer),
			shape_invariants=(idx.get_shape(), [None], layer.get_shape()),
			maximum_iterations=self.tree_depth,
			name='encoder_loop'
			)
		
		return flags, permute, offset, scale


class NbitTreeDecoder(Layer):
	"""
	"""
	def __init__(self, dim, bits_per_dim,
		offset=None,
		scale=None,
		permute=None,
		dtype='float32',
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTreeDecoder, self).__init__(
			name=name if name else "{}bitTreeDecoder".format(dim),
			trainable=False,
			dtype=dtype,
			**kwargs
			)
		self.dim = dim
		self.bits_per_dim = bits_per_dim
		self.offset=offset
		self.scale=scale
		self.permute = permute
		pass
	
	@property
	def fbits(self):
		return 1<<self.dim
	
	@property
	def xdim(self):
		return len(self.bits_per_dim)
	
	@property
	def word_length(self):
		return sum(self.bits_per_dim)
	
	@property
	def tree_depth(self):
		return int(np.ceil(self.word_length / self.dim))
	
	def __call__(self, flags, permute=None, offset=None, scale=None):
		if offset is None:
			offset = self.offset
		if scale is None:
			scale = self.scale
		if permute is None:
			permute = self.permute
		
		cond = lambda *args: True
		X = tf.constant([0], dtype=tf.int64, name='X')
		pos = tf.constant([0], dtype=tf.int32, name='pos')
		
		def body(X, pos):
			return bitops.decode(flags, pos, self.dim, X)
		
		X, pos = tf.while_loop(
			cond, body,
			loop_vars=(X, pos),
			shape_invariants=([None], pos.get_shape()),
			maximum_iterations=self.tree_depth,
			name='decoder_loop'
			)
		
		if permute is not None:
			X = bitops.permute(X[:,None], permute, self.word_length)
		X = bitops.realize(X, self.bits_per_dim, offset, scale, self.dtype)
		return X


class Transformer(Layer):
	"""
	"""
	def __init__(self, k,
		axes=(1,2),
		#activation='relu',
		activation=None,
		normalize=False,
		name='Transformer',
		**kwargs
		):
		"""
		"""
		super(Transformer, self).__init__(name=name, **kwargs)
		self.permute = Permute(axes[::-1], **kwargs)
		self.dot = Dot(axes, normalize, **kwargs)
		self.dense_n = Dense(k, activation=activation, **kwargs)
		self.dense_m = Dense(k, activation=activation, **kwargs)
		self.dense_t = Dense(k, activation=activation, **kwargs)
		
		self.config = kwargs
		self.config['k'] = k
		self.config['activation'] = activation
		self.config['normalize'] = normalize
		self.config['name'] = name
		pass
	
	def __call__(self, inputs):
		"""
		"""
		n = self.dense_n(inputs) #(b, n, k)
		m = self.dense_m(inputs) #(b, m, k)
		t = self.dense_t(inputs) #(b, t, k)
		m = self.permute(m) #(b, k, m)
		t = self.permute(t) #(b, k, t)
		T = self.dot([n,m]) #(b, k, k) Transformer!
		return self.dot([t,T]) #(b, t, k)
	
	def count_params(self):
		return self.dense_n.count_params() \
			+ self.dense_m.count_params() \
			+ self.dense_t.count_params()
	
	def get_config(self):
		return self.config
