
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPool1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import normalize, batching
from .. import bitops, dbxtree, spatial
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
		kernels=16,
		kernel_size=3,
		convolutions=4,
		branches=('uids', 'pos', 'pivots', 'vsort'),
		dense=0,
		activation='softmax',
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(DynamicTree2, self).__init__(name=name, dtype=dtype, **kwargs)
		self.kernels = kernels
		self.kernel_size = kernel_size
		self.branches = dict()
		self.layer_register = []
		#branches = set(branches)

		conv = [Conv1D(
			self.kernels, self.kernel_size, 1,
			activation='relu',
			padding='valid',
			dtype=self.dtype,
			name='conv_{}_{}'.format(branch, i),
			**kwargs
		) for i in range(convolutions)]
		
		maxp = [MaxPool1D(

		) for i in range(convolutions)]

		for branch in branches:
			if branch in ('uids', 'pos', 'pivots'):
				self.branches[branch] = utils.Prototype(
					merge = Conv1D(
						self.kernels, self.flag_size, self.flag_size,
						#activation='softsign',
						padding='valid',
						dtype=self.dtype,
						name='merge_{}'.format(branch),
						**kwargs
						),
					
					)
				self.layer_register += list(self.branches[branch].__dict__.values())
			pass
		
		self.dense = [Dense(
			self.kernels,
			activation='relu',
			dtype=self.dtype,
			name='dense_{}'.format(i),
			**kwargs
			) for i in range(dense)]

		self.head = Dense(
			self.bins,
			activation=activation,
			dtype=self.dtype,
			name='head',
			**kwargs
			)
		pass
	
	@property
	def flag_size(self):
		return 8
	
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
		radius=(np.math.pi / 4096, 1, 0.015),
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
				if augmentation:
					# Random Quantization
					scale = tf.random.uniform([1], 10, 1000, dtype=X.dtype)
					X = tf.math.round(X * scale) / scale
					# Random Rotation
					a = tf.random.uniform([1], -np.math.pi, np.math.pi, dtype=X.dtype)[0]
					M = tf.reshape([tf.cos(a),-tf.sin(a),0,tf.sin(a),tf.cos(a),0,0,0,1], (-1,3))
					X = X@M

				v = bitops.bitwise_and(X[...,:-1,1] < 0, X[...,1:,1] >= 0)
				v = tf.cast(v, tf.int32)
				v = tf.cumsum(v, axis=-1)[...,None]
				d = tf.norm(X, axis=-1, keepdims=True)
				u = tf.atan2(X[...,1], X[...,0])[...,None]

				if scale_u:
					u *= d / tf.math.reduce_max(d)
					
				U = tf.concat([u,v,d], axis=-1)
				abs = tf.math.abs(U)
				vmax = tf.math.reduce_max(v) + 1

				nodes = tf.cast(v, dtype=tf.int64)
				bbox = tf.math.unsorted_segment_max(abs, nodes, vmax)
				pos = bbox * tf.constant((0, 1, 0.5))
				dim = tf.zeros_like(bbox) + meta.dim
				means = tf.math.unsorted_segment_mean(U, nodes, vmax)
				radius = tf.repeat([radius], vmax, axis=0)
				u = U

				layer = 0
				while np.any(dim) and (max_layers == 0 or max_layers > layer):
					layer += 1
					vsort = tf.math.reduce_max(pos, axis=-2)
					vsort = tf.math.cumprod(vsort[...,::-1], axis=-1, exclusive=True)[...,::-1]
					vsort = tf.math.reduce_sum(pos * vsort)
					vsort = tf.argsort(vsort)
					u, nodes, pivots, _pos, bbox, flags, uids, dim, means = dbxtree.encode2(u, nodes, pos, bbox, radius, means)
					yield flags, uids, pos, pivots, means, vsort, bbox, dim, layer, U, filename
					pos = _pos
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
				tf.int64,
				tf.float32,
				tf.float32,
				tf.float32,
				tf.int32,
				tf.int32,
				tf.float32,
				tf.string
				),
			output_shapes=(
				tf.TensorShape([None]),
				tf.TensorShape([None]),
				tf.TensorShape([None, 1, meta.dim]),
				tf.TensorShape([None, self.flag_size, meta.dim]),
				tf.TensorShape([meta.dim]),
				tf.TensorShape([]),
				tf.TensorShape([]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([])
				)
			)
		return encoder, meta

	def trainer(self, *args, 
		encoder=None,
		meta=None,
		batch_size=1,
		**kwargs
		):
		"""
		"""
		@tf.function
		def features(flags, uids, pos, pivots, index, bbox, dim, *args):

			index = tf.repeat(index, self.kernels, axis=0)
			index = tf.roll(index, tf.range(self.kernels) - self.kernels//2, axis=-1)

			if 'pos' in self.branches:
				meta.features['pos'] = tf.reshape(pivots, (-1,meta.dim))

			if 'uids' in self.branches:
				token = bitops.left_shift(tf.range(meta.flag_size, dtype=uids.dtype), meta.dim)
				uids = bitops.left_shift(uids[...,None], meta.dim)
				uids = bitops.bitwise_or(uids, token)
				uids = tf.reshape(uids, (-1,1))
				uids = bitops.right_shift(uids, tf.range(63, dtype=uids.dtype))
				uids = bitops.bitwise_and(uids, 1)
				mask = tf.math.reduce_max(uids, axis=0)
				uids = mask - uids * -2
				meta.features['uids'] = tf.cast(uids, self.dtype)
			
			feature = tf.concat([meta.features[k] for k in self.branches], axis=-1, name='concat_features')
			labels = tf.one_hot(flags, meta.bins, dtype=meta.dtype)
			sample_weight = tf.cast(dim, self.dtype)
			return feature, labels, sample_weight
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		meta.features = dict()
		trainer = encoder.map(features)
		trainer = trainer.prefetch(2)
		trainer = batching(trainer, self.window_size, batch_size)
		meta.batch_size = batch_size
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
			super(DynamicTree2, self).build(input_shape)
		else:
			self.meta = meta or self.meta
			for k in self.branches:
				size = self.meta.features[k].shape[-1]
				self.branches[k].size = size
				self.branches[k].offsets = (incr(0), incr(size))
			feature_size = sum([b.size for b in self.branches.values()])
			super(DynamicTree2, self).build(tf.TensorShape((None, None, feature_size)))
	
	def call(self, inputs, *args):
		"""
		"""
		X = 0



		for name, branch in self.branches.items():
			if name == 'vsort':
				continue
			x = inputs[...,branch.offsets[0]:branch.offsets[1]]
			x = branch.merge(x)
			x0 = tf.stop_gradient(x)
			for conv in branch.conv:
				x = tf.concat([x0, conv(x)], axis=-1)
				x = normalize(x)
			X += x
		x = tf.stop_gradient(X)

		for dense in self.dense:
			X = tf.concat([x, dense(X)], axis=-1)
			X = normalize(X)
		X = self.head(X)
		return X

__all__ = [DynamicTree2]