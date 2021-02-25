
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from .. import range_coder


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
		self.encoder = encoder
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
		
		for i, sample, data in zip(range(self.steps), self.samples, self.data):
			uids, labels = sample[:2]
			layer = data[3].numpy()
			num_points = len(data[4])
			tree_end = layer == self.meta.tree_depth-1
			encode = self.encoder is None and tree_end
			if layer == 0:
				self.encoder and self.encoder.reset()
				probs = np.zeros((0, self.meta.bins), dtype=self.meta.dtype)
				acc_labels = labels
			else:
				acc_labels = tf.concat([acc_labels, labels], axis=1)
			metrics = self.model.test_on_batch(uids, labels, reset_metrics=False, return_dict=True)
			probs, code = self.model.predict_on_batch((uids, probs, acc_labels, encode))
			code = code[0]
			
			if self.encoder:
				labels = labels.numpy().reshape(-1, self.meta.bins)
				labels = np.nonzero()[-1]
				cdfs = range_coder.cdf(probs, precision=16, floor=self.model.floor)
				code = self.encoder.updates(labels, cdfs)

			if tree_end:
				bpp = float(len(code) * 8 / num_points)
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
		
		if code:
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

	def on_epoch_end(self, epoch, log):
		self.msg = "Epoch {}: ".format(epoch+1) + ", ".join(['{} = {}'.format(k,v) for k,v in log.items()])
	
	def on_epoch_begin(self, epoch, log):
		if self.msg:
			self.logger.info(self.msg)
	
	def on_train_end(self, log):
		if self.msg:
			self.logger.info(self.msg)