#!/usr/bin/env python3

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.bitwise import bitwise_and, right_shift, left_shift
from tensorflow.math import reduce_sum, top_k
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Contatenate, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback


class Decoder(Model):
	"""
	"""
	def __init__(self, depth=8, k=64, max_pts=0):
		"""
		"""
		super(MyModel, self).__init__()
		self.depth = depth
		self.kernel = k
		self.max_pts = max_pts

	def call(self, inputs, training=False):
		"""
		"""
		self.training = training
		self.P = []
		
		Xi = iter(inputs)
		a = next(Xi)
		intype = a.dtype
		dim = a.shape[-1]
		classes = 1<<dim
		offset = depth*2
		
		depth = self.depth
		k = self.kernel
		
		for b in Xi:
			#FUS
			fus = []
			for d in range(dim):
				c = tf.transpose(a[...,None,:,d] + b[...,d])
				fus.append(c) #(m, n, b)
			fus = tf.stack(fus) #(dim, m, n, b)
			fus = tf.transpose(fus) #(b, n, m, dim)
			
			#HOT
			shifts = tf.range(offset, dtype=intype)
			token = right_shift(fus[..., None], shifts) #(b, n, m, dim, bits)
			token = bitwise_and(token, 1)
			token = tf.transpose(token, perm=(0,1,2,4,3)) #(b, n, m, bits, dim)
			shifts = tf.range(dim, dtype=intype)
			token = left_shift(token, shifts)
			token = reduce_sum(token, axis=-1) #(b, n, m, bits)
			hot = tf.one_hot(token, classes) #(b, n, m, bits, classes)
			hot = tf.transpose(hot)
			hot = Concatenate()([*hot])
			hot = tf.transpose(hot) #(b, n, m, bits*classes)
			
			#N CONV
			N = Dense(k, activation='relu')(hot) #(b, n, m, k)
			N = reduce_sum(N/k, axis=-1) #(b, n, m)
			
			#M CONV
			hot = tf.transpose(hot, perm=[0,2,1,3,4])
			M = Dense(k, activation='relu')(hot) #(b, m, n, k)
			M = reduce_sum(M/k, axis=-1) #(b, m, n)
			
			#PROB
			P = tf.matmul(N, M) #(b, n, m)
			self.P.append(P)
			
			if self.max_pts:
				m = Flatten()(P)
				m = top_k(m, k=self.max_pts, sorted=True)
				m = tf.boolean_mask(m.indices, m.values > 0)
				a = Flatten()(a)
				a = tf.gather(a, m, axis=-1)
			else:
				m = P > 0
				a = tf.boolean_mask(fus, m)
			offset += depth
		return a