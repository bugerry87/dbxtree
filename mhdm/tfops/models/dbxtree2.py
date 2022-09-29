
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Reshape, Dense, Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import normalize, batching
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
		kernels=128,
		pre_conv=2,
		post_conv=4,
		dense=0,
		dtype=tf.float32,
		name=None,
		**kwargs
		):
		"""
		"""
		super(DynamicTree2, self).__init__(name=name, dtype=dtype, **kwargs)
		self.kernels = kernels

		self.conv = [Conv1D(
			self.kernels, 3, 1,
			padding='same',
			activation='relu',
			dtype=self.dtype,
			name='pre_conv_{}'.format(i),
			**kwargs
			) for i in range(pre_conv)]
		
		self.flags = [
			*(Dense(
				self.kernels,
				activation='relu',
				dtype=self.dtype,
				name='dense_flags_{}'.format(i),
				**kwargs
			) for i in range(dense)),
			Dense(
				self.dims,
				activation='softmax',
				dtype=self.dtype,
				name='flags_head',
				**kwargs
			)]
		
		self.symbols = [
			Conv1D(
				self.kernels, self.flag_size, self.flag_size,
				padding='valid',
				activation='relu',
				dtype=self.dtype,
				name='merge_symbols',
				**kwargs
			),
			*[Conv1D(
				self.kernels, 3, 1,
				padding='same',
				activation='relu',
				dtype=self.dtype,
				name='post_conv_{}'.format(i),
				**kwargs
			) for i in range(post_conv)]
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
			)]
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
				if augmentation:
					# Random Quantization
					scale = tf.random.uniform([1], 10, 1000, dtype=X.dtype)
					X = tf.math.round(X * scale) / scale
					# Random Rotation
					a = tf.random.uniform([1], -np.math.pi, np.math.pi, dtype=X.dtype)[0]
					M = tf.reshape([tf.cos(a),-tf.sin(a),0,tf.sin(a),tf.cos(a),0,0,0,1], (-1,3))
					X = X@M

				v = bitops.bitwise_and(X[...,:-1,1] < 0, X[...,1:,1] >= 0)
				v = tf.cast(v, tf.int64)
				v = tf.cumsum(v, axis=-1)[...,None]
				d = tf.norm(X, axis=-1, keepdims=True)
				u = tf.atan2(X[...,1], X[...,0])[...,None]
				vmax = tf.math.reduce_max(v) + 1
				vmean = tf.math.unsorted_segment_mean(X[...,:-1], v, vmax)
				v = tf.gather(vmean, v)
				v = tf.argsort(v)
				dmax = tf.math.reduce_max(d)

				if scale_u:
					u *= d / dmax
					
				U = tf.concat([u,v,d], axis=-1)
				absU = tf.math.abs(U)

				nodes = v
				inv = v
				pos = tf.range(vmax, delta=U.dtype)[...,None] * (0.0, 1.0, 0.0)
				bbox = tf.math.unsorted_segment_max(absU, nodes, vmax) * (1.0, 0.0, 1.0)
				dims = tf.math.reduce_sum(bbox, axis=-1)
				#means = tf.zeros_like(bbox)
				radius = tf.constant(radius)[None,...]
				u = U

				layer = 0
				while tf.math.reduce_any(dims) and (max_layers == 0 or max_layers > layer):
					layer += 1
					u, nodes, inv, _pos, bbox, flags, dims, means, uids = dbxtree.encode(u, nodes, inv, bbox, radius, None, pos)
					yield flags, means, uids, pos, bbox, dims, layer, U, filename
					pos = _pos
					pass
				pass
			pass
		
		if parser is None:
			parser, meta = self.parser(*args, **kwargs)
		meta.radius = radius
		meta.shape = tf.cast((2 * np.math.pi, vmax, dmax) / radius, tf.int64).numpy()
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
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([None, meta.dim]),
				tf.TensorShape([None]),
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
		def features(flags, means, uids, pos, bbox, dims, *args):
			token = bitops.left_shift(tf.range(meta.flag_size, dtype=uids.dtype), meta.dim)
			pos = tf.repeat(pos[...,None,:], meta.flag_size, axis=-2)

			if 'org' in self.branches:
				meta.features['org'] = tf.reshape(pos, (-1, meta.dim))

			if 'pos' in self.branches:
				pos = pos + tf.cast(token, pos.dtype) * bbox[...,None,:]
				meta.features['pos'] = tf.reshape(pos, (-1, meta.dim))

			if 'uids' in self.branches:
				uids = bitops.left_shift(uids[...,None], meta.dim)
				uids = bitops.bitwise_or(uids, token)
				uids = tf.reshape(uids, (-1,1))
				uids = bitops.right_shift(uids, tf.range(63, dtype=uids.dtype))
				uids = bitops.bitwise_and(uids, 1)
				mask = tf.math.reduce_max(uids, axis=0)
				uids = mask - uids * 2
				meta.features['uids'] = tf.cast(uids, self.dtype)

			feature = tf.concat([*(meta.features[k] for k in self.branches)], axis=-1)
			bits = bitops.right_shift(flags, tf.range(self.flag_size, dtype=flags.dtype))
			bits = bitops.bitwise_and(bits, 1)
			bits = tf.cast(bits, feature.dtype)
			labels = tf.concat([
				bits,
				tf.one_hot(flags, self.bins, dtype=self.dtype),
				tf.gather(means, (0,2), batch_dims=2)
			], axis=-1)
			#sample_weight = tf.cast(dim, self.dtype)
			return feature, labels #, sample_weight

		meta = meta if meta else self.meta
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
	
	def build(self, meta=None):
		"""
		"""
		def incr(j):
			incr.i += j
			return incr.i
		incr.i = 0
		self.meta = meta or self.meta
		for k in self.branches:
			size = self.meta.features[k].shape[-1]
			self.branches[k].size = size
			self.branches[k].offsets = (incr(0), incr(size))
		feature_size = sum([b.size for b in self.branches.values()])
		self.meta.shape2d = tf.cast((*self.meta.shape[:2], feature_size), tf.int64)
		super(DynamicTree2, self).build(tf.TensorShape((None, None, feature_size)))
	
	def call(self, inputs, *args):
		"""
		"""
		def umap(X, layer_iter):
			try:
				layer = next(layer_iter)
				x = tf.stop_gradient(X)
				X = layer.conv(X)
				X = layer.maxpool(X)
				X = umap(X, layer_iter)
				X = layer.deconv(X)
				return tf.concat([x,X], axis=-1)
			except:
				return X
		
		offsets = self.meta.branches['pos'].offsets
		pos = inputs[..., offsets[0]:offsets[1]-1]
		sort = tf.argsort(pos, axis=-2)
		X = tf.gather(inputs, sort[...,1], batch_dims=2)

		x = tf.stop_gradient(X)
		for conv in self.conv:
			X = conv(X)
			X = tf.concat([x,X], axis=-1)
		
		X = tf.gather(inputs, sort[...,0], batch_dims=2)
		F = X
		for layer in self.flags:
			F = layer(F)
		
		S = tf.concat([F, X], axis=-1)
		S = tf.stop_gradient(S)
		for layer in self.symbols:
			S = layer(S)

		F = F[...,0] - F[...,1]
		F = Reshape((-1, self.flag_size))(F)
		
		M = tf.transpose([
			[-1,-1,1,1],
			[-1,1,-1,1]
		])
		M = tf.keras.activation.softmax(F, axis=-1) @ tf.cast(M, F.dtype)	

		return tf.concat([F,S,M], axis=-1)

__all__ = [DynamicTree2]