
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from .. import bitops


class TestCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, info, meta,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		range_encode=True
		):
		"""
		"""
		super(TestCallback, self).__init__(**{w:self for w in when})
		self.samples = samples
		self.info = info
		self.meta = meta
		self.steps = steps or meta.num_of_samples
		self.freq = freq
		self.writer = writer
		self.range_encode = range_encode
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, step = args[:2]
		if step % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		self.model.reset_metrics()

		for i, sample, info in zip(range(self.steps), self.samples, self.info):
			uids, labels = sample
			counts = info[3].numpy()
			flags = info[4]
			layer = info[5].numpy()
			encode = self.range_encode and layer == self.meta.tree_depth-1
			if layer == 0:
				total_points = int(counts)
				probs = tf.zeros((0, self.meta.bins), dtype=self.meta.dtype)
				acc_flags = flags
			else:
				acc_flags = tf.concat([acc_flags, flags], axis=-1)
			metrics = self.model.test_on_batch(uids, labels, reset_metrics=False, return_dict=True)
			probs, code = self.model.predict_on_batch((uids, probs, acc_flags, encode))
			code = code[0]

			if encode:
				bpp = len(code) * 8 / total_points
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp

		if self.range_encode and code:
			metrics['bpp'] = bpp_sum / self.meta.num_of_files
			metrics['bpp_min'] = bpp_min
			metrics['bpp_max'] = bpp_max
		
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