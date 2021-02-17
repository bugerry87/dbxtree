
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from .. import range_coder


class TestCallback(LambdaCallback):
	"""
	"""
	def __init__(self, tester, tester_args, test_meta,
		test_freq=1,
		test_steps=0,
		when=['on_epoch_end'],
		writer=None
		):
		"""
		"""
		super(TestCallback, self).__init__(**{w:self for w in when})
		self.tester = tester
		self.tester_args = tester_args
		self.test_meta = test_meta
		self.test_steps = test_steps if test_steps else test_meta.num_of_samples
		self.test_freq = test_freq
		self.writer = writer

		self.compiled_metrics = None
		self.encoder = range_coder.RangeEncoder()
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, step = args[:2]
		if step % self.test_freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		self.model.reset_metrics()
		
		print('\n')
		for i, sample, args in zip(range(self.test_steps), self.tester, self.tester_args):
			uids, labels, weights = sample
			layer = args[1].numpy()
			encode = layer == self.test_meta.tree_depth-1
			if layer == 0:
				self.encoder.reset()
				probs = np.zeros((0, self.test_meta.bins), dtype=self.test_meta.dtype)
			metrics = self.model.test_on_batch(uids, labels, weights, reset_metrics=False, return_dict=True)
			probs, code = self.model.predict_on_batch((encode, uids, probs, labels))
			code = code[0]
			
			if not self.model.tensorflow_compression:
				labels = np.nonzero(labels.numpy())[-1]
				cdfs = range_coder.cdf(probs, precision=16, floor=0.01)
				code = self.encoder.updates(labels, cdfs)
				print('.', end='', flush=True)

			if encode:
				bpp = len(code) * 8 / len(uids)
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
		
		if code:
			metrics['bpp'] = bpp_sum / self.test_meta.num_of_files
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