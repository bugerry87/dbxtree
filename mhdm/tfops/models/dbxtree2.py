
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Dense, Conv1D, Dropout, Conv1DTranspose as Deconv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import normalize
from .. import bitops, dbxtree2 as dbxtree, spatial
from ... import utils, lidar

## Optional
try:
	import tensorflow_transform as tfx
except ModuleNotFoundError:
	tfx = None

class DynamicTree2(Model):
	"""
	"""
	def __init__(self,
		features=('org', 'pos', 'uids'),
		kernels=128,
		pre_conv=2,
		post_conv=2,
		dense=0,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(DynamicTree2, self).__init__(name=name, dtype=dtype, **kwargs)
		self.kernels = kernels
		self.features = { f: utils.Prototype() for f in features }

		self.pre_conv = [(
			Conv1D(
				self.kernels, 3, 1,
				padding='same',
				activation='relu',
				dtype=self.dtype,
				name='pre_conv_{}'.format(i),
				**kwargs
			),
		) for i in range(pre_conv)]
		
		self.post_conv = [(
			Conv1D(
				self.kernels, self.flag_size, self.flag_size,
				padding='valid',
				activation='relu',
				dtype=self.dtype,
				name='merge',
				**kwargs
			),
			Conv1D(
				self.kernels, 3, 1,
				padding='same',
				activation='relu',
				dtype=self.dtype,
				name='post_conv_{}'.format(i),
				**kwargs
			),
			Deconv1D(
				self.kernels, self.flag_size, self.flag_size,
				padding='valid',
				activation='relu',
				dtype=self.dtype,
				name='post_deconv_{}'.format(i),
				**kwargs
			)
		) for i in range(post_conv)]
		
		self.flags = [
			*(Dense(
				self.kernels,
				activation='relu',
				dtype=self.dtype,
				name='dense_flags_{}'.format(i),
				**kwargs
			) for i in range(dense)),
			Dense(
				2,
				activation='softmax',
				dtype=self.dtype,
				name='flags_head',
				**kwargs
			)
		]

		self.means = [
			Conv1D(
				self.kernels, self.flag_size, self.flag_size,
				padding='valid',
				activation='relu',
				dtype=self.dtype,
				name='merge_means',
				**kwargs
			),
			*(Dense(
				self.kernels,
				activation='relu',
				dtype=self.dtype,
				name='dense_mean_{}'.format(i),
				**kwargs
			) for i in range(dense)),
			Dense(
				6,
				activation='tanh',
				dtype=self.dtype,
				name='mean_head',
				**kwargs
			)
		]
		
		self.symbols = [
			Conv1D(
				self.kernels, self.flag_size, self.flag_size,
				padding='valid',
				activation='relu',
				dtype=self.dtype,
				name='merge_symbols',
				**kwargs
			),
			*(Dense(
				self.kernels,
				activation='relu',
				dtype=self.dtype,
				name='dense_symbols_{}'.format(i),
				**kwargs
			) for i in range(dense)),
			Dense(
				self.bins,
				activation='softmax',
				dtype=self.dtype,
				name='symbols_head',
				**kwargs
			)
		]
		
		self.reshape_flags = Reshape((-1, self.flag_size))
		self.reshape_means = Reshape((-1, 3))
		self.dropout = Dropout(0.1)
		pass
	
	@property
	def dims(self):
		return 2

	@property
	def flag_size(self):
		return 1<<(self.dims)
	
	@property
	def bins(self):
		return 1<<self.flag_size

	def set_meta(self, index, **kwargs):
		"""
		"""
		meta = utils.Prototype(
			flag_size=self.flag_size,
			bins=self.bins,
			dtype=self.dtype,
			**kwargs
			)
		if isinstance(index, str) and index.endswith('.txt'):
			meta.index = index
			with open(index, 'r') as fid:
				meta.num_of_files = sum(len(L) > 0 for L in fid)
		else:
			meta.index = [f for f in utils.ifile(index)]
			meta.num_of_files = len(index)
		self.meta = meta
		return meta
	
	def parser(self, index,
		dim=3,
		xtype='float32',
		xshape=(-1,4),
		shuffle=0,
		take=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index,
			dim=dim,
			xtype=xtype,
			xshape=xshape,
			**kwargs
			)
		
		@tf.function
		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, xshape)[...,:meta.dim]
			return X, filename

		if isinstance(index, str) and index.endswith('.txt'):
			parser = tf.data.TextLineDataset(index)
		else:
			parser = tf.data.Dataset.from_tensor_slices(index)

		if shuffle:
			if take:
				parser = parser.batch(take)
				parser = parser.shuffle(shuffle)
				parser = parser.unbatch()
			else: 
				parser = parser.shuffle(shuffle)
		if take:
			parser = parser.take(take)
		return parser.map(parse), meta
	
	def encoder(self, *args,
		radius=(np.math.pi / 2048, 1, 0.003),
		max_layers=0,
		augmentation=False,
		scale_u=False,
		parser=None,
		meta=None,
		**kwargs
		):
		"""
		"""
		def encode():
			for X, filename in parser:
				v = tf.math.logical_and(X[...,:-1,1] < 0, X[...,1:,1] >= 0)
				v = tf.cast(v, tf.int64)
				v = tf.cumsum(v, axis=-1)
				v = tf.pad(v, [(1,0)])

				'''
				if augmentation:
					# Random Quantization
					scale = tf.random.uniform([1], 10, 1000, dtype=X.dtype)
					X = tf.math.round(X * scale) / scale
					# Random Rotation
					a = tf.random.uniform([1], -np.math.pi, np.math.pi, dtype=X.dtype)[0]
					M = tf.reshape([tf.cos(a),-tf.sin(a),0,tf.sin(a),tf.cos(a),0,0,0,1], (-1,3))
					X = X@M
				'''

				u = tf.atan2(X[...,1], X[...,0])
				d = tf.norm(X, axis=-1)
				vmax = tf.math.reduce_max(v) + 1
				vmean = tf.math.unsorted_segment_mean(X[...,-1], v, vmax)
				i = tf.argsort(vmean)
				v = tf.gather(i, v)
				dmax = tf.math.reduce_max(d)

				if scale_u:
					u *= d / dmax
					
				U = tf.stack([u, tf.cast(v, u.dtype), d - dmax * 0.5], axis=1)
				absU = tf.math.abs(U)

				nodes = tf.cast(v, tf.int64)
				inv = nodes
				r = tf.constant(radius)[None,...]
				pos = tf.range(vmax, dtype=U.dtype)[...,None] * (0.0, 1.0, 0.0)
				bbox = tf.math.unsorted_segment_max(absU, nodes, vmax) * (1.0, 0.0, 1.0)
				dims = tf.math.reduce_sum(bbox, axis=-1)
				uids = tf.zeros(vmax, dtype=tf.int64)
				u = U

				layer = 0
				while np.any(dims.numpy()) and (max_layers == 0 or max_layers > layer):
					layer += 1
					#features, labels = self.featurize(self, flags, means, uids, pos, bbox)
					#means = self.predict_on_batch(features) 
					means = (0.5 - tf.random.uniform(bbox.shape)) 
					u, nodes, inv, _bbox, flags, dims, _means, _pos, _uids = dbxtree.encode(u, nodes, inv, bbox, r, means * bbox, pos)
					yield flags, means, uids, pos, bbox, layer, U, filename
					pos = _pos
					means = _means
					bbox = _bbox
					uids = _uids
					pass
				pass
			pass
		
		if parser is None:
			parser, meta = self.parser(*args, **kwargs)
		meta.radius = radius
		meta.max_layers = max_layers
		meta.augmentation = augmentation
		encoder = tf.data.Dataset.from_generator(encode,
			output_types=(
				tf.int32,
				tf.float32,
				tf.int64,
				tf.float32,
				tf.float32,
				tf.int32,
				tf.float32,
				tf.string
				),
			output_shapes=(
				tf.TensorShape([None]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([None]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([])
				)
			)
		return encoder, meta
	
	@tf.function
	def featurize(self, flags, means, uids, pos, bbox, *args):
		pos = tf.repeat(pos[...,None,:], self.flag_size, axis=-2)

		if 'org' in self.features:
			self.features['org'].feature = tf.reshape(pos, (-1, self.meta.dim))

		if 'pos' in self.features:
			sign = tf.range(self.flag_size)
			sign = bitops.right_shift(sign[...,None], tf.range(self.meta.dim))
			sign = bitops.bitwise_and(sign, 1)
			sign = 0.5 - tf.cast(sign, self.dtype)
			sign *= (1.0, 0.0, 1.0)
			pos = pos - sign * bbox[...,None,:]
			pos = tf.reshape(pos, (-1, self.meta.dim))
			self.features['pos'].feature = pos

		if 'uids' in self.features:
			uids = bitops.left_shift(uids[...,None], self.dims)
			uids = bitops.bitwise_or(uids, tf.range(self.flag_size, dtype=uids.dtype))
			uids = tf.reshape(uids, (-1,1))
			uids = bitops.right_shift(uids, tf.range(63, dtype=uids.dtype))
			uids = bitops.bitwise_and(uids, 1)
			mask = tf.math.reduce_max(uids, axis=0)
			uids = mask - uids * 2
			self.features['uids'].feature = tf.cast(uids, self.dtype)
		
		index = tf.argsort(pos, axis=-2)
		index = tf.cast(index, pos.dtype)
		feature = tf.concat([*(f.feature for f in self.features.values()), index], axis=-1, name='concat_features')
		bits = bitops.right_shift(flags[...,None], tf.range(self.flag_size, dtype=flags.dtype))
		bits = bitops.bitwise_and(bits, 1)
		bits = tf.cast(bits, feature.dtype)
		labels = tf.concat([
			bits,
			tf.one_hot(flags, self.bins, dtype=self.dtype),
			means
		], axis=-1, name='concat_labels')
		#sample_weight = tf.cast(dim, self.dtype)
		return feature, labels #, sample_weight

	def trainer(self, *args, 
		encoder=None,
		meta=None,
		**kwargs
		):
		"""
		"""
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		else:
			meta = meta if meta else self.meta
		meta.features = self.features
		trainer = encoder.map(self.featurize)
		trainer = trainer.prefetch(2)
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
	
	def build(self):
		"""
		"""
		def incr(j):
			incr.i += j
			return incr.i
		incr.i = 0
		for k in self.features:
			size = self.features[k].feature.shape[-1]
			self.features[k].size = size
			self.features[k].offsets = (incr(0), incr(size))
		feature_size = sum([f.size for f in self.features.values()]) + self.meta.dim
		super(DynamicTree2, self).build(tf.TensorShape((None, None, feature_size)))
	
	def call(self, inputs, *args):
		sort = tf.cast(inputs[..., -3:], tf.int32)
		X = x = tf.gather(inputs, sort[...,1], batch_dims=1)[...,:-self.meta.dim]
		for conv_layers in self.pre_conv:
			for layer in conv_layers:
				X = layer(X)
				X = self.dropout(X)
			X = tf.concat([X, x], axis=-1)
		
		X = tf.gather(X, sort[...,0], batch_dims=1)
		for conv_layers in self.post_conv:
			for layer in conv_layers:
				X = layer(X)
				X = self.dropout(X)
			X = tf.concat([X, x], axis=-1)
		
		F = X
		for layer in self.flags:
			F = layer(F)
		
		X = tf.concat([F, X], axis=-1)
		M = S = tf.stop_gradient(X)
		for layer in self.means:
			M = layer(M)
		
		for layer in self.symbols:
			S = layer(S)
		
		M = M[...,::2] - M[...,1::2]
		M = self.reshape_means(M)

		F = F[...,::2] - F[...,1::2]
		F = self.reshape_flags(F)

		X = tf.concat([F,S,M], axis=-1, name='concat_estimation')
		return X

__all__ = [DynamicTree2]