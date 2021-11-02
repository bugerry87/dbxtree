
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import normalize
from .. import bitops, spatial, yield_devices, count
from ... import utils

## Optional
try:
	import tensorflow_compression as tfc
except ModuleNotFoundError:
	tfc = None

class NbitTree(Model):
	"""
	"""
	def __init__(self, dim,
		heads = 1,
		kernels=16,
		kernel_size=3,
		convolutions=4,
		branches=('uids', 'pos', 'pivots', 'meta'),
		dense=0,
		activation='softmax',
		floor=0.0,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(NbitTree, self).__init__(name=name, dtype=dtype, **kwargs)
		self.mode = dim
		self.dim = max(dim, 1)
		self.heads = max(heads, 1)
		self.kernels = kernels
		self.kernel_size = kernel_size
		self.floor = floor
		self.branches = dict()
		self.layer_register = []
		self.devices = iter(yield_devices('GPU'))
		#branches = set(branches)

		for branch in branches:
			if branch in ('uids', 'pos', 'pivots', 'meta'):
				self.branches[branch] = utils.Prototype(
					merge = Conv1D(
						self.kernels, self.flag_size, self.flag_size,
						activation='relu',
						padding='valid',
						dtype=self.dtype,
						name='merge_{}'.format(branch),
						**kwargs
						),
					conv = [Conv1D(
						self.kernels, self.kernel_size, 1,
						activation='relu',
						padding='same',
						dtype=self.dtype,
						name='conv_{}_{}'.format(branch, i),
						**kwargs
						) for i in range(convolutions)]
					)
				self.layer_register += list(self.branches[branch].__dict__.values())
			pass
		
		self.dense = [Dense(
			self.kernels,
			activation='relu',
			#kernel_initializer='random_uniform',
			dtype=self.dtype,
			name='dense_{}'.format(i),
			**kwargs
			) for i in range(dense)]

		self.head = Dense(
			self.bins,
			activation=activation,
			#kernel_initializer='random_uniform',
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
		return 1<<self.flag_size

	def set_meta(self, index, bits_per_dim,	**kwargs):
		"""
		"""
		meta = utils.Prototype(
			bits_per_dim=bits_per_dim,
			dim=self.dim,
			mode=self.mode,
			flag_size=self.flag_size,
			bins=self.bins,
			dtype=self.dtype,
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
	
	def encoder(self, index, bits_per_dim,
		sort_bits=None,
		permute=None,
		offset=None,
		scale=None,
		payload=False,
		spherical=False,
		keypoints=False,
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
			keypoints=keypoints,
			offset=offset,
			scale=scale,
			xtype=xtype,
			qtype=qtype,
			ftype=ftype,
			**kwargs
			)
		
		@tf.function
		def augment(X):
			a = tf.random.uniform([1], dtype=X.dtype) * 6.3
			M = tf.concat([tf.cos(a), -tf.sin(a), [0], tf.sin(a), -tf.cos(a), [0,0,0,1]], axis=0)
			M = tf.reshape(M, [3,3])
			return tf.concat([X[:,:3] @ M, X[:,3:]], axis=-1)

		@tf.function
		def parse(filename):
			with tf.device(next(self.devices).name):
				X = tf.io.read_file(filename)
				X = tf.io.decode_raw(X, xtype)
				X = tf.reshape(X, (-1, meta.input_dims))
				i = tf.math.reduce_all(tf.math.is_finite(X), axis=-1)
				X = X[i]
				points = count(X[...,0])
				#X = augment(X)
				if meta.keypoints:
					X = tf.gather(X, spatial.edge_detection(X[...,:,:3], meta.keypoints)[0])
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
				points = [points] * meta.tree_depth
				if payload:
					mask = [tf.ones_like(X0[0], dtype=bool)] + [tf.gather(*tf.unique_with_counts(X0[i])[-1:0:-1]) > 1 for i in range(meta.tree_depth-1)]
				else:
					mask = [tf.ones_like(X0[0], dtype=bool)] * meta.tree_depth
			return X0, X1, layer, filename, permute, offset, scale, mask, points
		
		@tf.function
		def encode(X0, X1, layer, filename, permute, offset, scale, mask, points):
			with tf.device(next(self.devices).name):
				uids, idx, counts = tf.unique_with_counts(X0[mask])
				flags = bitops.encode(X1[mask], idx, meta.dim, ftype)[0]
				if meta.payload:
					flags *= tf.cast(counts > 1, flags.dtype)
			return (uids, flags, layer, permute, offset, scale, X0, mask, points, filename)
		
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
		@tf.function
		def features(uids, flags, layer, permute, *args):
			pivots = uids[...,None]
			token = bitops.left_shift(tf.range(meta.flag_size, dtype=pivots.dtype), layer*meta.dim)
			uids = bitops.bitwise_or(pivots, token)
			uids = tf.reshape(uids, [-1])
			pos = None

			if 'pos' in self.branches:
				with tf.device(next(self.devices).name):
					pos = NbitTree.finalize(uids, meta, permute, scale=0.5, word_length=(layer+1)*meta.dim)
					meta.features['pos'] = pos
			
			if 'pivots' in self.branches:
				with tf.device(next(self.devices).name):
					kernel = [*range(-self.kernels//4, 0), *range(1, self.kernels//4+1)]
					pos = pos if pos is not None else NbitTree.finalize(uids, meta, permute, scale=0.5, word_length=(layer+1)*meta.dim)
					pivots = NbitTree.finalize(pivots, meta, permute, scale=0.5, word_length=(layer+1)*meta.dim)[...,None,:]
					pivots = tf.concat([tf.roll(pivots, i, 0) for i in kernel], axis=-2)
					pivots = pos[...,None,:] - tf.repeat(pivots, meta.flag_size, axis=0) 
					pivots = tf.math.reduce_sum(pivots * pivots, axis=-1)
					pivots = tf.exp(-pivots)
					meta.features['pivots'] = pivots
			
			if 'meta' in self.branches:
				with tf.device(next(self.devices).name):
					indices = tf.ones_like(uids, dtype=layer.dtype)[...,None] * layer * meta.dim
					indices = tf.concat([indices + i for i in range(meta.dim)], axis=-1)
					permute = tf.gather(permute, indices)
					permute = tf.one_hot(permute, meta.word_length, dtype=meta.dtype)
					permute = tf.reshape(permute, [-1, meta.word_length * meta.dim])
					meta.features['meta'] = permute

			if 'uids' in self.branches:
				with tf.device(next(self.devices).name):
					uids = bitops.right_shift(uids[...,None], tf.range(meta.word_length, dtype=uids.dtype))
					uids = bitops.bitwise_and(uids, 1)
					uids = tf.cast(uids, meta.dtype)
					m = tf.range(uids.shape[-1], dtype=layer.dtype) < (layer+1) * meta.dim
					uids = uids * 2 - tf.cast(m, meta.dtype)
					uids = tf.concat([tf.math.minimum(uids, 0.0), tf.math.maximum(uids, 0.0)], axis=-1)
					meta.features['uids'] = uids
			
			feature = tf.concat([meta.features[k] for k in self.branches], axis=-1)
			return feature, flags
		
		@tf.function
		def labels(feature, flags):
			labels = tf.one_hot(flags, meta.bins, dtype=meta.dtype)
			return feature, labels
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		meta.features = dict()
		trainer = encoder.map(features)
		trainer = trainer.map(labels)
		trainer = trainer.batch(1)
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
		def incr(j):
			incr.i += j
			return incr.i
		incr.i = 0

		if input_shape:
			super(NbitTree, self).build(input_shape)
		else:
			self.meta = meta or self.meta
			for k in self.branches:
				size = self.meta.features[k].shape[-1]
				self.branches[k].size = size
				self.branches[k].offsets = (incr(0), incr(size))
			feature_size = sum([b.size for b in self.branches.values()])
			super(NbitTree, self).build(tf.TensorShape((None, None, feature_size)))
	
	def call(self, inputs, *args):
		"""
		"""
		X = 0
		for (name, branch), device in zip(self.branches.items(), self.devices):
			with tf.device(device.name):
				x = inputs[...,branch.offsets[0]:branch.offsets[1]]
				x = branch.merge(x)
				x0 = tf.stop_gradient(x)
				for conv in branch.conv:
					x = tf.concat([x0, conv(x)], axis=-1)
					x = normalize(x)
				X += x
		x = tf.stop_gradient(X)

		with tf.device(next(self.devices).name):
			for dense in self.dense:
				X = tf.concat([x, dense(X)], axis=-1)
				X = normalize(X)
			X = self.head(X)
		return X
	
	def predict_step(self, data):
		"""
		"""
		if self.mode <= 0:
			X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
			feature = X[0]
			probs = self(feature, training=False)[...,1-self.bins:]
			return probs

		def encode():
			if tfc is None:
				tf.get_logger().warn(
					"Model has no range_encoder and will only return raw probabilities and an empty string. " \
					"Please install 'tensorflow-compression' to obtain encoded bit-streams."
					)
				return empty_code
			
			cdf = tf.reshape(probs, (-1, self.bins-1))
			symbols = tf.reshape(labels-1, [-1])
			symbols = tf.cast(symbols, tf.int16)
			
			pmax = tf.math.reduce_max(cdf, axis=-1, keepdims=True, name='pmax')
			cdf = tf.math.divide_no_nan(cdf, pmax)
			cdf = tf.clip_by_value(cdf, self.floor, 1.0)
			cdf = tf.math.cumsum(cdf, axis=-1)
			cdf /= tf.math.reduce_max(cdf, axis=-1, keepdims=True, name='cdf_max')
			cdf = tf.math.round(cdf * float(1<<16))
			cdf = tf.cast(cdf, tf.int32)
			cdf = tf.pad(cdf, [(0,0),(1,0)])
			code = tfc.range_encode(symbols, cdf, precision=16)
			return code[None,...]
		
		def ignore():
			return empty_code
		
		empty_code = tf.constant([''], name='ignore')
		X, _, _ = data_adapter.unpack_x_y_sample_weight(data)
		feature, probs, labels, do_encode = X
		do_encode = tf.math.reduce_all(do_encode)
		#pred = self(feature, training=False)[...,1-self.bins:]
		probs = tf.one_hot(labels-1, self.bins-1, dtype=probs.dtype)[None,...]
		#probs = tf.concat([probs, pred], axis=-2, name='concat_probs')
		code = tf.cond(do_encode, encode, ignore, name='do_encode_cond')
		return probs, code

	@staticmethod
	def parse(filename):
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
	def finalize(X, meta, permute=None, offset=0.0, scale=1.0, word_length=None):
		"""
		"""
		X = tf.reshape(X, (-1,1))
		bits_per_dim = meta.bits_per_dim
		if permute is not None and permute.shape[0]:
			X = bitops.permute(X, permute, meta.word_length)
		
		'''
		else:
			permute = tf.range(meta.word_length)[::-1]
		
		if word_length is not None:
			low = tf.math.cumsum(bits_per_dim, exclusive=True)
			high = tf.math.cumsum(bits_per_dim, exclusive=False)
			permute = tf.cast(permute[:word_length, None], low.dtype)
			bits_per_dim = tf.math.reduce_sum(tf.cast(permute >= low, meta.qtype) * tf.cast(permute < high, meta.qtype), axis=0)
			'''

		X = bitops.realize(X, bits_per_dim, offset, scale, meta.xtype)
		if meta.spherical:
			X = spatial.uvd2xyz(X)
		return X

__all__ = [NbitTree]