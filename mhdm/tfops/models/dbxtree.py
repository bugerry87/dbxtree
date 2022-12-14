
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import normalize
from .. import bitops, dbxtree, spatial
from ... import utils

## Optional
try:
	import tensorflow_transform as tfx
except ModuleNotFoundError:
	tfx = None

class DynamicTree(Model):
	"""
	"""
	def __init__(self,
		kernels=16,
		kernel_size=3,
		convolutions=4,
		branches=('uids', 'pos', 'pivots'),
		dense=0,
		activation='softmax',
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(DynamicTree, self).__init__(name=name, dtype=dtype, **kwargs)
		self.kernels = kernels
		self.kernel_size = kernel_size
		self.branches = dict()
		self.layer_register = []
		#branches = set(branches)

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
					conv = [Conv1D(
						self.kernels, self.kernel_size, 1,
						#activation='relu',
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
		radius=0.015,
		max_layers=0,
		keypoints=0.0,
		use_pca = False,
		augmentation = False,
		parser=None,
		meta=None,
		**kwargs
		):
		"""
		"""
		def encode():
			for X, filename in parser:
				if keypoints:
					X = tf.gather(X, spatial.edge_detection(X[...,:], keypoints)[0])
				
				if augmentation:
					# Random Quantization
					scale = tf.random.uniform([1], 10, 1000, dtype=X.dtype)
					X = tf.math.round(X * scale) / scale
					# Random Rotation
					if not use_pca:
						a = tf.random.uniform([1], -np.math.pi, np.math.pi, dtype=X.dtype)[0]
						M = tf.reshape([tf.cos(a),-tf.sin(a),0,tf.sin(a),tf.cos(a),0,0,0,1], (-1,3))
						X = X@M
				
				offset = tf.math.reduce_min(X, axis=-2, keepdims=True)
				offset += tf.math.reduce_max(X, axis=-2, keepdims=True)
				X -= offset * 0.5
				
				bbox = tf.math.reduce_max(tf.math.abs(X), axis=-2)
				pos = tf.zeros_like(bbox)[None, None, ...]
				nodes = tf.ones_like(X[...,0], dtype=tf.int64)
				dim = tf.constant(meta.dim)

				if use_pca and tfx:
					pca = tfx.pca(X, meta.dim)
					X = X @ pca
				else:
					pca = None
					i = tf.argsort(bbox)[::-1]
					X = tf.gather(X, i, batch_dims=-1)
					bbox = tf.gather(bbox, i)
				x = X

				layer = 0
				while np.any(dim) and (max_layers == 0 or max_layers > layer):
					layer += 1
					x, nodes, pivots, _pos, bbox, flags, uids, dim = dbxtree.encode(x, nodes, pos, bbox, radius)
					yield flags, uids, pos, pivots, bbox, dim, layer, X, filename
					pos = _pos
		
		if parser is None:
			parser, meta = self.parser(*args, **kwargs)
		meta.radius = radius
		meta.max_layers = max_layers
		meta.keypoints = keypoints
		meta.use_pca = use_pca
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
		**kwargs
		):
		"""
		"""
		@tf.function
		def features(flags, uids, pos, pivots, bbox, dim, *args):
			if 'pos' in self.branches:
				meta.features['pos'] = tf.reshape(pivots, (-1,meta.dim))

			if 'pivots' in self.branches:
				kernel = [*range(-self.kernels//2, 0), *range(1, self.kernels//2+1)]
				pivots = tf.concat([tf.roll(pivots, i, 0) for i in kernel], axis=-2, name='concat_pivot_kernel')
				pivots = pivots - pos
				pivots = tf.math.reduce_sum(pivots * pivots, axis=-1)
				pivots = tf.exp(-pivots)
				meta.features['pivots'] = tf.reshape(pivots, (-1,self.kernels))

			if 'uids' in self.branches:
				token = bitops.left_shift(tf.range(meta.flag_size, dtype=uids.dtype), meta.dim)
				uids = bitops.left_shift(uids[...,None], meta.dim)
				uids = bitops.bitwise_or(uids, token)
				uids = tf.reshape(uids, (-1,1))
				uids = bitops.right_shift(uids, tf.range(63, dtype=uids.dtype))
				uids = bitops.bitwise_and(uids, 1)
				mask = tf.math.reduce_max(uids, axis=0)
				uids = mask - uids * 2
				meta.features['uids'] = tf.cast(uids, self.dtype)
			
			feature = tf.concat([meta.features[k] for k in self.branches], axis=-1, name='concat_features')
			labels = tf.one_hot(flags, meta.bins, dtype=meta.dtype)
			#sample_weight = tf.cast(dim, self.dtype)
			return feature, labels #, sample_weight
	
		if encoder is None:
			encoder, meta = self.encoder(*args, **kwargs)
		meta.features = dict()
		trainer = encoder.map(features)
		#trainer = trainer.prefetch(8)
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
			super(DynamicTree, self).build(input_shape)
		else:
			self.meta = meta or self.meta
			for k in self.branches:
				size = self.meta.features[k].shape[-1]
				self.branches[k].size = size
				self.branches[k].offsets = (incr(0), incr(size))
			feature_size = sum([b.size for b in self.branches.values()])
			super(DynamicTree, self).build(tf.TensorShape((None, None, feature_size)))
	
	def call(self, inputs, *args):
		"""
		"""
		X = 0
		for name, branch in self.branches.items():
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
			#X = tf.math.log(X + 1.0)
			X = normalize(X)
		X = self.head(X)
		return X

__all__ = [DynamicTree]