
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
		
		for i, sample, data in zip(range(self.steps), self.samples, self.data):
			uids, labels = sample[:2]
			counts = data[1].numpy()
			hist = data[3].numpy()
			layer = data[4].numpy()
			if layer == 0:
				total_points = counts.sum()
				bits = int(total_points).bit_length()
				self.buffer.open('data/test_{}.bin'.format(i), mode='wb')
				self.buffer.write(bits, 5, soft_flush=True)
				self.buffer.write(total_points, bits, soft_flush=True)
				bits_per_node = [bits]
			probs = self.model.predict_on_batch(uids)
			probs = probs.reshape(-1, self.meta.output_size)
			nodes = np.where(probs[:,0]>probs[:,1], hist[:,0], hist[:,1])
			assert(len(nodes) == len(bits_per_node))
			for bits, points in zip(bits_per_node, nodes):
				self.buffer.write(points, bits, soft_flush=True)
			bits_per_node = [int(h).bit_length() for h in hist[hist>0]]

			if layer == self.meta.tree_depth-1:
				bpp = len(self.buffer) / float(total_points)
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
		
		self.buffer.close()
		metrics = dict(
			bpp = bpp_sum / self.meta.num_of_files,
			bpp_min = bpp_min,
			bpp_max = bpp_max
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