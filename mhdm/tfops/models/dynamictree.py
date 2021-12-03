
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv1D
from tensorflow.python.keras.engine import data_adapter

## Local
from . import normalize
from .. import bitops, spatial, dynamictree
from ... import utils

## Optional
try:
	import tensorflow_compression as tfc
except ModuleNotFoundError:
	tfc = None

try:
	import tensorflow_transform as tfx
except ModuleNotFoundError:
	tfx = None

class DynamicTree(Model):
	"""
	"""
	def __init__(self,
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
		self.mode = 3
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
		return 1<<3
	
	@property
	def bins(self):
		return 1<<self.flag_size

	def set_meta(self, index, **kwargs):
		"""
		"""
		meta = utils.Prototype(
			mode=self.mode,
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
		meta.num_of_samples = meta.num_of_files * meta.tree_depth
		self.meta = meta
		return meta
	
	def parser(self, index,
		xtype='float32',
		ntype='int64',
		ftype='int8',
		keypoints=0.0,
		augment=False,
		shuffle=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index,
			xtype=xtype,
			ntype=ntype,
			ftype=ftype,
			keypoints=keypoints,
			**kwargs
			)
		
		@tf.function
		def augment(X):
			a = tf.random.uniform([1], dtype=X.dtype) * 6.3
			M = tf.concat([tf.cos(a), -tf.sin(a), [0], tf.sin(a), -tf.cos(a), [0,0,0,1]], axis=0)
			M = tf.reshape(M, [3,3])
			return tf.concat([X @ M, X], axis=-1)
		
		@tf.function
		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, 4))
			i = tf.math.reduce_all(tf.math.is_finite(X), axis=-1)
			X = X[i,:3]

			if augment:
				X = augment(X)
			if meta.keypoints:
				X = tf.gather(X, spatial.edge_detection(X[...,:], meta.keypoints)[0])
			
			if tfx:
				pca = tfx.pca(X, 3)
				X = X @ pca
			else:
				pass
			
			X0 = X[:-1]
			X1 = X[1:]
			layer = tf.range(meta.tree_depth, dtype=X.dtype)
			filename = [filename] * meta.tree_depth
			points = [points] * meta.tree_depth
		return X, pca

		if isinstance(index, str) and index.endswith('.txt'):
			parser = tf.data.TextLineDataset(index)
		else:
			parser = tf.data.Dataset.from_tensor_slices(index)

		if shuffle:
			parser = parser.shuffle(shuffle)
		return parser.map(parse)
	
	def encoder(self, index,
		radius=0.015,
		xtype='float32',
		ntype='int64',
		ftype='int8',
		keypoints=0.0,
		augment=False,
		shuffle=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index,
			radius=radius,
			xtype=xtype,
			ntype=ntype,
			ftype=ftype,
			keypoints=keypoints,
			**kwargs
			)
		
		

		@tf.function
		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, meta.input_dims))
			i = tf.math.reduce_all(tf.math.is_finite(X), axis=-1)
			X = X[i]
			points = count(X[...,0])
			if augment:
				X = augment(X)
			if meta.keypoints:
				X = tf.gather(X, spatial.edge_detection(X[...,:,:3], meta.keypoints)[0])
			
			if tfx:
				pca = tfx.pca(X, 3)
				X = X @ pca
			else:
				pass

			
			X0 = X[:-1]
			X1 = X[1:]
			layer = tf.range(meta.tree_depth, dtype=X.dtype)
			filename = [filename] * meta.tree_depth
			points = [points] * meta.tree_depth
		return X, pca
		
		@tf.function
		def encode(X, *args):
			bbox = tf.math.reduce_max(tf.math.abs(X), axis=-2)
			nodes = tf.ones_like(X[...,0], dtype=ntype)
			dim = tf.constant(3)
			Flags = []
			Uids = []
			Dims = []
			BBoxes = []
			Frames = []

			def cond(x, nodes, bbox, dim):
				return dims > 0
			
			def body(x, nodes, bbox, dim):
				x, nodes, bbox, flags, dim, uids, frame = dynamictree.encode(x, nodes, bbox, radius)
				Flags.append(flags)
				Uids.append(uids)
				Dims.append(dim)
				BBoxes.append(bbox)
				Frames.append(frame)
				return x, nodes, bbox, dim
			
			tf.while_loop(cond, body, 
				loop_vars=[X, nodes, bbox, dim],
				shape_invariants=[(None,3), (None,), (3,) (1,)]
			)

			flags = tf.concat(Flags, axis=0)
			uids = tf.concat(Uids, axis=0)
			dims = tf.concat(Dims, axis=0)
			bboxes = tf.concat(BBoxes, axis=0)
			frames = tf.concat(Frames, axis=0)
			return flags, uids, bboxes, dims, frames, X, pca, filename,
		
		if isinstance(index, str) and index.endswith('.txt'):
			encoder = tf.data.TextLineDataset(index)
		else:
			encoder = tf.data.Dataset.from_tensor_slices(index)
		
		if shuffle:
			encoder = encoder.shuffle(shuffle)
		encoder = encoder.map(parse)
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
				pos = NbitTree.finalize(uids, meta, permute, scale=0.5, word_length=(layer+1)*meta.dim)
				meta.features['pos'] = pos
			
			if 'pivots' in self.branches:
				kernel = [*range(-self.kernels//4, 0), *range(1, self.kernels//4+1)]
				pos = pos if pos is not None else NbitTree.finalize(uids, meta, permute, scale=0.5, word_length=(layer+1)*meta.dim)
				pivots = NbitTree.finalize(pivots, meta, permute, scale=0.5, word_length=(layer+1)*meta.dim)[...,None,:]
				pivots = tf.concat([tf.roll(pivots, i, 0) for i in kernel], axis=-2)
				pivots = pos[...,None,:] - tf.repeat(pivots, meta.flag_size, axis=0) 
				pivots = tf.math.reduce_sum(pivots * pivots, axis=-1)
				pivots = tf.exp(-pivots)
				meta.features['pivots'] = pivots

			if 'uids' in self.branches:
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
		pred = self(feature, training=False)[...,1-self.bins:]
		#probs = tf.one_hot(labels-1, self.bins-1, dtype=probs.dtype)[None,...]
		probs = tf.concat([probs, pred], axis=-2, name='concat_probs')
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