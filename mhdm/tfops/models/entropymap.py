
## Installed
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose

## Local
from .. import count
from .. import bitops
from ... import utils

@tf.function
def map_entropy(X, bits):
	"""
	"""
	def cond(*args):
		return tf.constant(True)
	
	def body(E, X, idx, counts):
		num_seg = tf.math.reduce_max(idx) + 1
		e = tf.math.unsorted_segment_sum(X, idx, num_seg) #n,b
		e = tf.gather(e, idx) #n,b
		eid = tf.argmax(tf.abs(e), axis=-1)[...,None]
		eid = tf.concat([point_range, eid], axis=-1)
		e = tf.gather_nd(e, eid)
		x = tf.gather_nd(X, eid)

		idx = bitops.left_shift(idx, 1) + tf.cast(x > 0, idx.dtype)
		sort = tf.argsort(idx)
		unsort = tf.argsort(sort)
		idx = tf.gather(idx, sort)
		idx, counts = tf.unique_with_counts(idx, X.dtype)[1:]
		idx = tf.gather(idx, unsort)
		
		X = tf.tensor_scatter_nd_update(X, eid, zeros)
		E = tf.tensor_scatter_nd_update(E, eid, e)
		return E, X, idx, counts

	with tf.name_scope("map_entropy"):
		counts = count(X)
		point_range = tf.range(counts)[...,None]
		shifts = tf.range(bits, dtype=X.dtype)
		one = tf.constant(1, dtype=X.dtype)
		idx = tf.zeros_like(X[...,0])
		zeros = tf.zeros_like(X[...,0])
		X = bitops.right_shift(X, shifts)
		X = bitops.bitwise_and(X, one)
		X = X * (one+one) - one
		E = tf.zeros_like(X)
		
		E = tf.while_loop(
			cond, body,
			loop_vars=(E, X, idx, counts[...,None]),
			shape_invariants=(E.shape, X.shape, idx.shape, tf.TensorShape((None))),
			maximum_iterations=bits
			)[0]
		E = tf.cast(E, tf.float32) / tf.cast(counts, tf.float32)
	return E


def permute(X, E):
	with tf.name_scope("entropy_permute"):
		E = tf.abs(E)
		p = tf.argsort(E, axis=-1)[:,::-1]
		X = bitops.permute(X, p, E.shape[-1])
	return X


class EntropyMapper(Model):
	"""
	"""
	def __init__(self,
		bins=48,
		kernels=16,
		kernel_size=3,
		strides=1,
		layers=3,
		**kwargs
		):
		"""
		"""
		super(EntropyMapper, self).__init__(**kwargs)
		self.bins = bins
		self.kernels = kernels

		self.encoder = Sequential([
			Conv1D(2**layers, kernel_size, strides,
				activation='relu',
				padding='same',
				name='encoder_conv_{}'.format(i)
				) for i in range(layers)
			], 'encoder')
		self.encoder.add(
			Dense(kernels,
				activation='tanh',
				name='encoder_dense_out'
			))
		
		self.teacher = Sequential([
			Conv1DTranspose(2**layers, kernel_size, strides,
				activation='relu',
				padding='same',
				name='teacher_conv_{}'.format(i)
				) for i in range(layers)
			], 'teacher')
		self.teacher.add(
			Dense(self.bins,
				activation='tanh',
				name='teacher_dense_out'
			))
		
		self.decoder = Sequential([
			Conv1DTranspose(2**layers, kernel_size, strides,
				activation='relu',
				padding='same',
				name='decoder_conv_{}'.format(i)
				) for i in range(1, layers)
			], 'decoder')
		self.decoder.add(
			Dense(self.bins,
				activation='tanh',
				name='decoder_dense_out'
			))
		pass

	def set_meta(self, index, bits_per_dim, **kwargs):
		"""
		"""
		meta = utils.Prototype(
			bits_per_dim=bits_per_dim,
			bins=self.bins,
			dtype=self.dtype,
			**kwargs
		)
		meta.input_dims = len(meta.bits_per_dim)
		meta.word_length = sum(meta.bits_per_dim)
		if isinstance(index, str) and index.endswith('.txt'):
			meta.index = index
			with open(index, 'r') as fid:
				meta.num_of_samples = sum(len(L) > 0 for L in fid)
		else:
			meta.index = [f for f in utils.ifile(index)]
			meta.num_of_samples = len(index)
		self.meta = meta
		return meta

	def mapper(self, index, bits_per_dim,
		offset=None,
		scale=None,
		xtype='float32',
		qtype='int64',
		shuffle=0,
		**kwargs
		):
		"""
		"""
		meta = self.set_meta(index, bits_per_dim,
			offset=offset,
			scale=scale,
			xtype=xtype,
			qtype=qtype,
			**kwargs
		)
		
		@tf.function
		def parse(filename):
			X = tf.io.read_file(filename)
			X = tf.io.decode_raw(X, xtype)
			X = tf.reshape(X, (-1, meta.input_dims))
			i = tf.math.reduce_all(tf.math.is_finite(X), axis=-1)
			X = X[i]
			X, offset, scale = bitops.serialize(X, meta.bits_per_dim, meta.offset, meta.scale, dtype=meta.qtype)
			E = map_entropy(X, self.bins)
			return E, filename, offset, scale
		
		if isinstance(index, str) and index.endswith('.txt'):
			mapper = tf.data.TextLineDataset(index)
		else:
			mapper = tf.data.Dataset.from_tensor_slices(index)

		if shuffle:
			mapper = mapper.shuffle(shuffle)
		mapper = mapper.map(parse)
		return mapper, meta

	def trainer(self, *args,
		mapper=None,
		meta=None,
		cache=None,
		**kwargs
		):
		"""
		"""
		def samples(E, *args):
			return E, (E, E)

		if mapper is None:
			trainer, meta = self.mapper(*args, **kwargs)
		else:
			trainer = mapper
		
		trainer = trainer.map(samples)
		trainer = trainer.batch(1)
		if cache:
			trainer = trainer.cache(cache)
		return trainer, meta
	
	def validator(self, *args,
		**kwargs
		):
		"""
		"""
		return self.trainer(*args, **kwargs)

	def encode(self, X):
		X = self.encoder(X)
		return X
	
	def teach(self, X):
		X = self.teacher(X)
		return X

	def decode(self, X):
		X = tf.cast(X, self.dtype)
		X = self.decoder(X)
		return X
	
	def build(self):
		self.encoder.build((None, None, self.bins))
		self.teacher.build((None, None, self.kernels))
		self.decoder.build((None, None, self.kernels))
		super(EntropyMapper, self).build((None, None, self.bins))

	def call(self, X, training=False):
		C = self.encode(X)
		if training:
			E = self.teach(C)
		Eb = self.decode(C > 0)
		Eb = tf.math.divide_no_nan(Eb, tf.stop_gradient(tf.math.reduce_max(Eb, axis=-1, keepdims=True)))
		if training:
			E = tf.math.divide_no_nan(E, tf.stop_gradient(tf.math.reduce_max(E, axis=-1, keepdims=True)))
			return Eb, E
		else:
			return Eb
	
	@staticmethod
	def map_entropy(X, bits):
		return map_entropy(X, bits)
