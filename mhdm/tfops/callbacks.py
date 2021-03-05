
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from .. import bitops


class TestCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, data, meta,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		encoder=None
		):
		"""
		"""
		super(TestCallback, self).__init__(**{w:self for w in when})
		self.samples = samples
		self.data = data
		self.meta = meta
		self.steps = steps or meta.num_of_samples
		self.freq = freq
		self.writer = writer
		self.buffer = bitops.BitBuffer()
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, step = args[:2]
		if step % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		overflows = 0
		
		for i, sample, data in zip(range(self.steps), self.samples, self.data):
			uids, labels = sample[:2]
			hist = data[3].numpy()
			layer = data[4].numpy()
			if layer == 0:
				overflow = 0
				counts = data[1].numpy()
				total_points = counts.sum()
				bits = int(total_points).bit_length()
				bits_per_node = [bits]
				self.buffer.open('data/test_{}.bin'.format(i), mode='wb')
				self.buffer.write(bits, 5, soft_flush=True)
				self.buffer.write(total_points, bits, soft_flush=True)
			probs = self.model.predict_on_batch(uids)
			probs = probs.reshape(-1, self.meta.output_size)
			probs /= np.linalg.norm(probs, ord=1, axis=-1, keepdims=True)
			minor = np.where(probs[:,0]<probs[:,1], hist[:,0], hist[:,1])
			probs = np.maximum(probs.min(axis=-1), overflow)
			bits_per_node = np.ceil(np.log2(probs * counts + 1)).astype(int)
			mask = (1<<bits_per_node)-1
			nodes = np.minimum(minor, mask)
			counts = np.minimum(hist, mask[...,None])
			counts = counts[counts>0]
			overflow = counts > hist[hist>0]
			
			for bits, points in zip(bits_per_node, nodes):
				self.buffer.write(points, bits, soft_flush=True)

			if layer == self.meta.tree_depth-1:
				bpp = len(self.buffer) / float(total_points)
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
			else:
				overflows += (minor > nodes).sum()

		
		self.buffer.close()
		metrics = dict(
			bpp = bpp_sum / self.meta.num_of_files,
			bpp_min = bpp_min,
			bpp_max = bpp_max,
			overflows = overflows / self.meta.num_of_files
			)
		
		for name, metric in metrics.items():
			name = 'test_' + name
			log[name] = metric
		
		if self.writer is not None:
			with self.writer.as_default():
				for name, metric in metrics.items():
					name = 'epoch_' + name
					tf.summary.scalar(name, metric, step)
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

	def on_epoch_end(self, epoch, log):
		self.msg = "Epoch {}: ".format(epoch+1) + ", ".join(['{} = {}'.format(k,v) for k,v in log.items()])
	
	def on_epoch_begin(self, epoch, log):
		if self.msg:
			self.logger.info(self.msg)
	
	def on_train_end(self, log):
		if self.msg:
			self.logger.info(self.msg)