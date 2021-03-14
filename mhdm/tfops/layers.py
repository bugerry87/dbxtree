
## Build In
from collections import Iterable

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Activation, Permute, Dot

## Local
from . import batched_identity, range_like
from . import bitops


def arg_filter(trainable=True, dtype=None, dynamic=False, **kwargs):
	return dict(trainable=trainable, dtype=dtype, dynamic=dynamic)

@tf.function
def cov(a, b=None, axis=0, normalize=True):
	m = tf.math.reduce_mean(a, axis=axis, keepdims=True)
	am = a - m

	if b is None:
		bm = am
	else:
		m = tf.math.reduce_mean(b, axis=axis, keepdims=True)
		bm = b - m
	c = tf.linalg.matmul(am, bm, transpose_a=True)

	if normalize:
		n = tf.ones_like(a)
		n = tf.math.reduce_sum(n, axis=axis, keepdims=True)
		c /= tf.math.reduce_max(n)
	return c


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
	
	def call(self, X):
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
	
	def call(self, flags, permute=None, offset=None, scale=None):
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


class Euclidean(Layer):
	"""
	"""
	def __init__(self, units,
		initializer='random_normal',
		regularizer=None,
		inverted=False,
		activation=None,
		matrix_mode=False,
		name='Euclidean',
		**kwargs
		):
		"""
		"""
		super(Euclidean, self).__init__(name=name, **arg_filter(**kwargs))
		self.units = units
		self.initializer = initializer
		self.regularizer = regularizer
		self.inverted=inverted
		self.matrix_mode = matrix_mode
		if activation is not None:
			self.activation = Activation(activation)
		else:
			self.activation = None
		
		self.config = dict(
			units = units,
			initializer = initializer,
			regularizer = regularizer,
			inverted = inverted,
			activation = activation,
			matrix_mode = matrix_mode,
			name = name,
			**kwargs
			)
		pass
	
	def call(self, inputs):
		if self.matrix_mode:
			a = tf.expand_dims(inputs, axis=-1)
			a = a - self.w
			a *= a 
			a = tf.math.reduce_sum(a, axis=-2)
		else:
			def cond(*args):
				return True
			
			def body(a, i):
				d = inputs[...,i,None] - self.w[i]
				d *= d
				a += a
				return a, i+1
			
			i = tf.constant(1)
			a = inputs[...,0,None] - self.w[0] #b,n,k
			a *= a
			a, i = tf.while_loop(cond, body,
				loop_vars=(a, i),
				maximum_iterations=self.dims-1,
				name='sum_loop'
			)
		
		if self.inverted:
			a = tf.math.exp(-a)
		if self.activation:
			a = self.activation(a)
		return a
	
	def build(self, input_shape):
		self.batch_size = input_shape[0]
		self.dims = input_shape[-1]
		self.w = self.add_weight(
			shape=(self.dims, self.units),
			initializer=self.initializer,
			regularizer=self.regularizer,
			trainable=self.trainable,
			name='kernel'
		)
		return self
	
	def get_config(self):
		return self.config


class Mahalanobis(Layer):
	"""
	"""
	def __init__(self, units,
		kernel_initializer='random_normal',
		kernel_regularizer=None,
		bias_initializer=batched_identity,
		bias_regularizer=None,
		inverted=False,
		activation=None,
		name='Mahalanobis',
		**kwargs
		):
		"""
		"""
		super(Mahalanobis, self).__init__(name=name, **arg_filter(**kwargs))
		self.units = units
		self.kernel_initializer = kernel_initializer
		self.kernel_regularizer = kernel_regularizer
		self.bias_initializer = bias_initializer
		self.bias_regularizer = bias_regularizer
		self.inverted=inverted
		if activation is not None:
			self.activation = Activation(activation)
		else:
			self.activation = None
		
		self.config = dict(
			units = units,
			kernel_initializer = kernel_initializer,
			kernel_regularizer = kernel_regularizer,
			bias_initializer = bias_initializer,
			bias_regularizer = bias_regularizer,
			inverted = inverted,
			activation = activation,
			**kwargs
			)
		pass
	
	def call(self, inputs):
		x = tf.expand_dims(inputs, axis=-1) #b,n,d,1
		x = x - self.w #b,n,d,k
		x = tf.transpose(x, (0,1,3,2)) #b,n,k,d
		x = tf.expand_dims(x, axis=-1) #b,n,d,k,1
		z = tf.matmul(x, self.b, transpose_a=True) #b,n,k,1,d
		z = tf.matmul(z, x) #b,n,k,1,1
		z = tf.math.reduce_sum(z, axis=(-2,-1)) #b,n,k
		if self.inverted:
			z = tf.math.exp(-z)
		if self.activation:
			z = self.activation(z)
		return z
	
	def build(self, input_shape):
		self.w = self.add_weight(
			shape=(input_shape[-1], self.units),
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			trainable=self.trainable,
			name='kernel'
			)

		self.b = self.add_weight(
			shape=(1, self.units, input_shape[-1], input_shape[-1]),
			initializer=self.bias_initializer,
			regularizer=self.bias_regularizer,
			trainable=self.trainable,
			name='bias'
			)
		
		return self
	
	def get_config(self):
		return self.config


class InnerTransformer(Layer):
	"""
	"""
	def __init__(self,
		*args,
		axes=(1,2),
		normalize=False,
		layer_types=Dense,
		layer_A_args={},
		layer_B_args={},
		layer_t_args={},
		name='InnerTransformer',
		**kwargs
		):
		"""
		"""
		super(InnerTransformer, self).__init__(name=name, **arg_filter(**kwargs))
		self.permute = Permute(axes[::-1], **arg_filter(**kwargs))
		self.dot = Dot(axes, normalize, **arg_filter(**kwargs))
		self.layer_types = layer_types
		
		if layer_types is not None:
			try:
				self.A = layer_types[0](*args, name='A', **layer_A_args, **kwargs)
				self.B = layer_types[1](*args, name='B', **layer_B_args, **kwargs)
				self.t = layer_types[2](*args, name='t', **layer_t_args, **kwargs)
			except TypeError:
				self.A = layer_types(*args, name='A', **layer_A_args, **kwargs)
				self.B = layer_types(*args, name='B', **layer_B_args, **kwargs)
				self.t = layer_types(*args, name='t', **layer_t_args, **kwargs)
		
		self.config = dict(
			args = args,
			normalize = normalize,
			layer_types = layer_types,
			layer_A_args = layer_A_args,
			layer_B_args = layer_B_args,
			layer_t_args = layer_t_args,
			name = name,
			**kwargs
			)
		pass

	def build(self, input_shape):
		if self.layer_types is not None:
			self.A.build(input_shape)
			self.B.build(input_shape)
			self.t.build(input_shape)
		return self
	
	def call(self, inputs):
		"""
		"""
		if self.layer_types is None:
			A, B, t = inputs
		else:
			A = self.A(inputs)
			B = self.B(inputs)
			t = self.t(inputs)
		B = self.permute(B) #(b, k, B)
		t = self.permute(t) #(b, k, t)
		T = self.dot([A,B]) #(b, k, k) Transformer!
		return self.dot([t,T]) #(b, t, k) Positional query
	
	def count_params(self):
		if self.layer_types is None:
			return 0
		else:
			return self.A.count_params() \
				+ self.B.count_params() \
				+ self.t.count_params()
	
	def get_config(self):
		return self.config


class OuterTransformer(Layer):
	"""
	"""
	def __init__(self,
		*args,
		layer_type=Dense,
		layer_t_args={},
		name='OuterTransformer',
		**kwargs
		):
		"""
		"""
		super(OuterTransformer, self).__init__(name=name, **arg_filter(**kwargs))
		self.layer_type = layer_type
		if layer_type:
			self.t = self.layer_type(*args, name='T', **layer_t_args, **kwargs)
		
		self.config = dict(
			args = args,
			layer_type = layer_type,
			layer_t_args = layer_t_args,
			name = name,
			**kwargs
			)
		pass

	def build(self, input_shape):
		if self.layer_type:
			self.t.build(input_shape)
		return self
	
	def call(self, inputs):
		"""
		"""
		if self.layer_type:
			A = inputs
			B = inputs
			t = self.t(inputs)
		else:
			A, B, t = inputs
		
		i = tf.constant([0], dtype=tf.int32)
		X = tf.zeros_like(B)
		B = tf.transpose(B, [0,2,1])
		#X = tf.transpose(X, [1,0,2])
		bound = tf.ones_like(A[0,:,0], dtype=tf.int32)
		bound = tf.math.reduce_sum(bound)

		def cond(i, X):
			return i[0] < bound
		
		def body(i, X):
			x = A[:,i[0]]@B
			X += tf.transpose(x, [0,2,1])@t[:,i[0]]
			return i+1, X

		i, X = tf.while_loop(cond, body,
			loop_vars=(i, X),
			parallel_iterations=64,
			name='dot_loop'
			)
		return X
	
	def count_params(self):
		if self.layer_type:
			return self.t.count_params()
		else:
			return 0		
	
	def get_config(self):
		return self.config


class CovarTransformer(Layer):
	"""
	"""
	def __init__(self,
		*args,
		axis=1,
		layer_type=Dense,
		layer_args={},
		normalize=True,
		name='CovarTransformer',
		**kwargs
		):
		"""
		"""
		super(CovarTransformer, self).__init__(name=name, **arg_filter(**kwargs))
		self.normalize = normalize
		
		if layer_type is not None:
			self.layer = layer_type(*args, name='out', **layer_args, **kwargs)
		else:
			self.layer = None
		
		self.config = dict(
			args = args,
			normalize = normalize,
			layer_type = layer_type,
			layer_args = layer_args,
			name = name,
			**kwargs
			)
		pass

	def build(self, input_shape):
		if self.layer is not None:
			self.layer.build(input_shape)
		return self
	
	def call(self, X):
		"""
		"""
		if self.layer:
			X = self.layer(X)
		C = cov(X, normalize=self.normalize)
		X = tf.linalg.matmul(X, C)
		return X
	
	def count_params(self):
		if self.layer is None:
			return 0
		else:
			return self.layer.count_params()
	
	def get_config(self):
		return self.config