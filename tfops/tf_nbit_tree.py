
## BuildIn
import sys

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

## Local
import mhdm.viz as viz
from mhdm import tfbitops
from mhdm import utils


class NbitTree():
	"""
	"""
	def __init__(self, dim, bits_per_dim, *args,
		encoder_input=True,
		decoder_input=True,
		xtype=np.float32,
		ftype=np.uint8,
		**kwargs
		):
		"""
		"""
		self._dim = dim
		self._bits_per_dim = bits_per_dim
		self._xtype = xtype
		self._ftype = ftype
		self._has_encoder = False
		self._has_decoder = False
		
		with tf.name_scope("{}bitTree".format(self.fbits)):
			if encoder_input is not None:
				self.build_encoder(encoder_input, **kwargs)
			if decoder_input is not None:
				self.build_decoder(decoder_input)	
		pass
	
	@property
	def xtype(self):
		return self._xtype
	
	@property
	def ftype(self):
		return self._ftype
	
	@property
	def dim(self):
		return self._dim
	
	@property
	def fbits(self):
		return 1<<self.dim
	
	@property
	def bits_per_dim(self):
		return tuple(self._bits_per_dim)
	
	@property
	def xdim(self):
		return len(self._bits_per_dim)
	
	@property
	def word_length(self):
		return sum(self.bits_per_dim)
	
	@property
	def tree_depth(self):
		return int(np.ceil(self.word_length / self.dim))
	
	@property
	def has_encoder(self):
		return self._has_encoder
	
	@property
	def has_decoder(self):
		return self._has_decoder
	
	def build_encoder(self,
		encoder_input=None,
		sort_bits=True,
		reverse=False,
		absolute=True,
		**kwargs
		):
		"""
		"""
		if self._has_encoder:
			return
		else:
			self._has_encoder = True
		
		tokens = tf.constant(np.arange(self.fbits, dtype=self.ftype))
		t_list = tf.TensorShape([None])
		
		def until_tree_end(*args):
			return True
		
		def iterate_layer(flags0, idx0, size0, layer):
			flags1 = nodes[layer]
			size1, idx1 = tf.unique(flags1, out_idx=nodes.dtype)
			size1 = tf.size(size1)
			flags1 = tfbitops.bitwise_and(flags1, self.fbits-1)
			flags1 = tf.one_hot(flags1, self.fbits, dtype=self.ftype)
			flags1 = tf.math.unsorted_segment_max(flags1, idx0, size0)
			flags1 = tfbitops.left_shift(flags1, tokens)
			flags1 = tf.math.reduce_sum(flags1, axis=-1)
			flags1 = tf.reshape(flags1, (-1,))
			flags1 = tf.concat([flags0, flags1], axis=0)
			return flags1, idx1, size1, layer+1
		
		if encoder_input is None or encoder_input is True:
			self._encoder_input = tf.compat.v1.placeholder(shape=[None,self.xdim], dtype=self.xtype, name='encoder_input')
		else:
			self._encoder_input = encoder_input
		
		with tf.name_scope("Encoder"):
			X, self._offset_output, self._scale_output = tfbitops.serialize(self._encoder_input, self.bits_per_dim)
			if sort_bits or absolute:
				X, self._permute_output = tfbitops.sort(X, self.word_length, reverse, absolute)
			elif reverse:
				self._permute_output = tf.range(self.word_length, dtype=X.dtype)[::-1]
				X = tfbitops.permute(X, self._permute_output, self.word_length)
			else:
				self._permute_output = tf.range(self.word_length, dtype=X.dtype)
			
			nodes = tfbitops.tokenize(X, self.dim, self.tree_depth)
			
			with tf.name_scope("boot_encoder"):
				flags, idx = tf.unique(nodes[0], out_idx=nodes.dtype)
				size = tf.size(flags)
				flags = tf.cast(flags, self.ftype)
				flags = tfbitops.left_shift(tf.constant(1, dtype=flags.dtype), flags)
				flags = tf.math.reduce_sum(flags)[None,]
				layer = tf.constant(1, dtype=nodes.dtype)
			
			self._encoder_output, idx, size, layer = tf.while_loop(
				until_tree_end, iterate_layer,
				loop_vars=(flags, idx, size, layer),
				shape_invariants=(t_list, idx.get_shape(), size.get_shape(), layer.get_shape()),
				maximum_iterations=self.tree_depth-1,
				name='loop_encoder'
				)
		pass
	
	def build_decoder(self, decoder_input=None):
		"""
		"""
		if self._has_decoder:
			return
		else:
			self._has_decoder = True
		
		tokens = tf.constant(np.arange(self.fbits, dtype=self.ftype))
		t_list = tf.TensorShape([None])
		
		def until_data_end(*args):
			return True
		
		def iterate_data(X, pos, layer):
			X = tfbitops.left_shift(X, self.dim)
			size = tf.reshape(tf.size(X), (-1,))
			flags = tf.slice(data, pos, size)
			flags = tf.reshape(flags, (-1,1))
			flags = tfbitops.right_shift(flags, tokens)
			flags = tfbitops.bitwise_and(flags, 1)
			x = tf.where(flags)
			i = x[:,0]
			x = x[:,1]
			pos += tf.size(X)
			X = x + tf.gather(X, i)
			return X, pos, layer+self.dim
		
		if decoder_input is None or decoder_input is True:
			self._decoder_input = tf.compat.v1.placeholder(shape=[None], dtype=self.ftype, name='decoder_input')
			self._offset_input = tf.compat.v1.placeholder(shape=self.xdim, dtype=self.xtype, name='offset_input')
			self._scale_input = tf.compat.v1.placeholder(shape=self.xdim, dtype=self.xtype, name='scale_input')
			self._permute_input = tf.compat.v1.placeholder(shape=self.word_length, dtype=tf.int64, name='permute_input')
		else:
			self._decoder_input, self._offset_input, self._scale_input, self._permute_input = decoder_input
		
		with tf.name_scope("Decoder"):
			with tf.name_scope("boot_decoder"):
				data = self._decoder_input
				X = tf.zeros([1], dtype=tf.int64, name='points')
				pos = tf.constant([0], dtype=tf.int32, name='pos')
				layer = tf.constant(0, dtype=X.dtype, name='layer')
			
			X, pos, layer = tf.while_loop(
				until_data_end, iterate_data,
				loop_vars=(X, pos, layer),
				shape_invariants=(t_list, pos.get_shape(), layer.get_shape()),
				maximum_iterations=self.tree_depth,
				name='loop_decoder'
				)
			
			X = tfbitops.permute(X[:,None], self._permute_input, self.word_length)
			X = tfbitops.realize(X, self.bits_per_dim, self._offset_input, self._scale_input, self.xtype)
			self._decoder_output = X
		pass
	
	def encode(self, X):
		if not self.has_encoder:
			raise RuntimeError("Encoder was not built!")
		with tf.compat.v1.Session() as sess:
			with tf.compat.v1.summary.FileWriter("logs", sess.graph) as writer:
				timer = utils.time_delta()
				next(timer)
				output, offset, scale, permute = sess.run(
					(self._encoder_output, self._offset_output, self._scale_output, self._permute_output),
					feed_dict={self._encoder_input:X}
					)
				print("Encoding time:", next(timer))
				
		header = utils.Prototype(
			offset=offset.tolist(),
			scale=scale.tolist(),
			permute=permute.tolist()
			)
		return output, header
	
	def decode(self, Y, header):
		if not self.has_decoder:
			raise RuntimeError("Decoder was not built!")
		
		with tf.compat.v1.Session() as sess:
			with tf.compat.v1.summary.FileWriter("logs", sess.graph) as writer:
				timer = utils.time_delta()
				next(timer)
				X = sess.run(
					self._decoder_output,
					feed_dict={
						self._decoder_input:Y, 
						self._offset_input:header.offset,
						self._scale_input:header.scale,
						self._permute_input:header.permute
						}
					)
				print("Decoding time:", next(timer))
		return X
