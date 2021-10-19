
## Build In
import os.path as path
import pickle

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

from mhdm.tfops.models.nbittree import NbitTree

## Local
from . import bitops
from ..bitops import BitBuffer

## Optional
try:
	import py7zr
except:
	py7zr = None


class NbitTreeCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, info, meta,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		range_encode=True,
		output=None,
		):
		"""
		"""
		super(NbitTreeCallback, self).__init__(**{w:self for w in when})
		self.samples = samples
		self.info = info
		self.meta = meta
		self.steps = steps or meta.num_of_samples
		self.freq = freq
		self.writer = writer
		self.range_encode = range_encode
		self.output = output
		self.buffer = BitBuffer() if output else None
		self.mode = self.flag_mode
		pass

	def flag_mode(self, step, sample, info, tree_start, tree_end):
		feature = sample[0]
		encode = tf.constant(tree_end and self.range_encode)
		flags = info[1]
		layer = info[2].numpy()
		if self.meta.payload:
			mask = info[-3].numpy()

		if tree_start:
			self.probs = tf.zeros((1, 0, self.meta.bins-1), dtype=self.meta.dtype)
			self.flags = flags
			if self.meta.payload:
				counts = len(mask)
				self.bits = np.zeros(counts, int)
		else:
			if self.meta.payload:
				self.bits[mask] = self.meta.word_length - (layer+1) * self.meta.dim
			self.flags = tf.concat([self.flags, flags], axis=-1)

		self.probs, code = self.model.predict_on_batch((feature, self.probs, self.flags, encode))
		if self.meta.payload and tree_end:
			payload = info[-4].numpy()
			bits = self.bits
		else:
			payload = []
			bits = []
		return self.probs[0], code[0], payload, bits

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, epoch = args[:2]
		if epoch % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		bpp_zip = 0
		self.model.reset_metrics()

		for step, sample, info in zip(range(self.steps), self.samples, self.info):
			filename = str(info[-1].numpy())
			tree_start = step % self.meta.tree_depth == 0
			tree_end = (step+1) % self.meta.tree_depth == 0
			metrics = self.model.test_on_batch(*sample, reset_metrics=False, return_dict=True)
			probs, code, payload, bits = self.mode(step, sample, info, tree_start, tree_end)

			if tree_start:
				X=tf.constant([0], dtype=tf.int64)
				points = float(info[-2])
				bit_count = 0
				if self.output:
					if py7zr:
						arcfile = path.join(self.output, path.splitext(path.basename(filename))[0] + '.nbit.7z')
						arcname = path.splitext(path.basename(filename))[0] + '.nbit.bin'
						buffer = path.join(self.output, '~nbit.tmp')
					else:
						buffer = path.join(self.output, path.splitext(path.basename(filename))[0] + '.nbit.bin')
					self.buffer.open(buffer, 'wb')
			
			if self.output:
				for f in info[1]:
					self.buffer.write(f, 1<<self.meta.dim, soft_flush=True)
				
				for p, b in zip(payload, bits):
					self.buffer.write(p, b, soft_flush=True)
			bit_count += len(code)*8 + int(sum(bits))
			X = NbitTree.decode(info[1], self.meta, X)
			
			if tree_end:
				self.buffer.close()
				if self.output and py7zr:
					with py7zr.SevenZipFile(arcfile, 'w') as z:
						z.write(buffer, arcname)
					bpp_zip += path.getsize(arcfile) * 8 / points
				bpp = bit_count / points
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
				#X = NbitTree.finalize(X, self.meta, info[3], info[4], info[5])
				#X.numpy().tofile('data/test.bin')

		metrics['bpp'] = bpp_sum / self.meta.num_of_files
		metrics['bpp_min'] = bpp_min
		metrics['bpp_max'] = bpp_max
		if bpp_zip:
			metrics['bpp_zip'] = bpp_zip / self.meta.num_of_files
		
		for name, metric in metrics.items():
			name = 'test_' + name
			log[name] = metric
		
		if self.writer is not None:
			with self.writer.as_default():
				for name, metric in metrics.items():
					name = 'epoch_' + name
					tf.summary.scalar(name, metric, epoch)
			self.writer.flush()
		pass


class LogCallback(Callback):
	"""
	"""
	def __init__(self, logger):
		super(LogCallback, self).__init__()
		self.logger = logger
		self.msg = None
		pass

	def __call__(self, log):
		self.logger.info("Test: " + ", ".join(['{} = {}'.format(k,v) for k,v in log.items()]))

	def on_epoch_end(self, epoch, log):
		self.msg = "Epoch {}: ".format(epoch+1) + ", ".join(['{} = {}'.format(k,v) for k,v in log.items()])
	
	def on_epoch_begin(self, epoch, log):
		if self.msg:
			self.logger.info(self.msg)
	
	def on_train_end(self, log):
		if self.msg:
			self.logger.info(self.msg)


class SaveOptimizerCallback(Callback):
	"""
	"""
	def __init__(self, optimizer, file_pattern,
		monitor='loss',
		save_best_only=True,
		mode='min'
		):
		"""
		"""
		super(SaveOptimizerCallback, self).__init__()
		self.optimizer = optimizer
		self.file_pattern = file_pattern
		self.monitor = monitor
		self.save_best_only = save_best_only
		self.best = None
		self.mode = mode
		self.epoch = 0
		self.__dict__['max'] = self.max
		self.__dict__['min'] = self.min
		pass

	def min(self, val):
		if self.best is None or self.best > val:
			self.best = val
			return True
		else:
			return False
	
	def max(self, val):
		if self.best is None or self.best < val:
			self.best = val
			return True
		else:
			return False

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, self.epoch = args[:2]
		if not self.save_best_only or self.__dict__[self.mode](log[self.monitor]):
			optimizer_weights = tf.keras.backend.batch_get_value(self.optimizer.weights)
			with open(self.file_pattern.format(epoch=self.epoch, **log), 'wb') as f:
				pickle.dump(optimizer_weights, f)
		pass

	def on_epoch_end(self, epoch, log):
		self(epoch, log)

	def on_train_end(self, log):
		self(self.epoch, log)
	
	'''
	def on_epoch_begin(self, epoch, log):
		pass
	
	def on_train_end(self, log):
		pass
	'''


class EntropyMapCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, info, meta,
		dim=1,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		):
		"""
		"""
		super(EntropyMapCallback, self).__init__(**{w:self for w in when})
		self.dim = dim
		self.samples = samples
		self.info = info
		self.meta = meta
		self.freq = freq
		self.steps = steps
		self.writer = writer
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, epoch = args[:2]
		if epoch % self.freq != 0:
			return
		
		gt_bpp_sum = 0.0
		prd_bpp_sum = 0.0
		self.model.reset_metrics()

		true = tf.constant(True)
		dim = tf.constant(self.dim, self.meta.qtype)

		def cond(*args):
			return true
		
		def body(nodes, flags, layer):
			idx = tf.unique(nodes[layer])[-1]
			layer += 1
			flags = tf.concat((flags, bitops.encode(nodes[layer], idx, dim, tf.int8)[0]), axis=-1)
			return nodes, flags, layer
		
		def encode(X, E):
			nodes = bitops.tokenize(X, dim, E.shape[-1])
			flags = tf.constant([], dtype=tf.int8)
			layer = tf.constant(0)

			flags = tf.while_loop(
				cond, body,
				loop_vars=(nodes, flags, layer),
				shape_invariants=(nodes.get_shape(), [None], layer.get_shape()),
				maximum_iterations=self.meta.bins-1,
				name='encoder_loop'
				)[1]
			return flags

		for step, sample, info in zip(range(self.steps), self.samples, self.info):
			gtE = sample[0]
			X = info[1]
			prdE = self.model.predict_on_batch(gtE)
			metrics = self.model.test_on_batch(*sample, reset_metrics=False, return_dict=True)
			
			gt_flags = encode(self.model.permute(X, gtE[0]), gtE)
			prd_flags = encode(self.model.permute(X, prdE[0]), prdE)
			gt_bpp = float(len(gt_flags.numpy()) * 2) / len(X.numpy())	
			prd_bpp = float(len(prd_flags.numpy()) * 2) / len(X.numpy())

			gt_bpp_sum += gt_bpp
			prd_bpp_sum += prd_bpp
			pass

		metrics['gt_bpp'] = gt_bpp_sum / self.meta.num_of_samples
		metrics['prd_bpp'] = prd_bpp_sum / self.meta.num_of_samples

		for name, metric in metrics.items():
			name = 'test_' + name
			log[name] = metric
		
		if self.writer is not None:
			with self.writer.as_default():
				for name, metric in metrics.items():
					name = 'epoch_' + name
					tf.summary.scalar(name, metric, epoch)
			self.writer.flush()
		pass
	pass		