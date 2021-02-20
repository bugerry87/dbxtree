
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from .. import range_coder


class TestCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, info, meta,
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
		self.info = info
		self.meta = meta
		self.steps = steps or meta.num_of_samples
		self.freq = freq
		self.writer = writer

		self.gt_flag_map = np.zeros((1, meta.tree_depth, meta.output_size, 1))
		self.pred_flag_map = np.zeros((1, meta.tree_depth, meta.output_size, 1))
		self.compiled_metrics = None
		self.encoder = encoder or range_coder.RangeEncoder()
		pass

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, step = args[:2]
		if step % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		self.gt_flag_map[:] = 0
		self.pred_flag_map[:] = 0
		self.model.reset_metrics()

		print('\n')
		for i, sample, info in zip(range(self.steps), self.samples, self.info):
			uids, labels, weights = sample
			gt_flags = info[1].numpy()
			layer = info[2].numpy()
			X = info[3].numpy()
			encode = layer == self.meta.tree_depth-1
			if layer == 0:
				self.encoder.reset()
				probs = np.zeros((0, self.meta.output_size), dtype=self.meta.dtype)
				acc_flags = gt_flags
			else:
				acc_flags = np.concatenate([acc_flags, gt_flags])
			metrics = self.model.test_on_batch(uids, labels, weights, reset_metrics=False, return_dict=True)
			probs, code = self.model.predict_on_batch((encode, uids, probs, acc_flags))
			code = code[0]
			pred_flags = np.argmax(probs[-len(gt_flags):], axis=-1)
			self.pred_flag_map[:, layer, pred_flags, :] += 1
			self.gt_flag_map[:, layer, gt_flags, :] += 1
			print(layer, end=' ', flush=True)

			if not self.model.tensorflow_compression:
				cdfs = range_coder.cdf(probs[:,1:], precision=32, floor=0.01)
				code = self.encoder.updates(gt_flags-1, cdfs)

			if encode:
				bpp = len(code) * 8 / len(X)
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
			self.gt_flag_map /= self.gt_flag_map.max()
			self.pred_flag_map /= self.pred_flag_map.max()
			with self.writer.as_default():
				for name, metric in metrics.items():
					name = 'epoch_' + name
					tf.summary.scalar(name, metric, step)
				tf.summary.image('gt_flag_map', self.gt_flag_map, step)
				tf.summary.image('pred_flag_map', self.pred_flag_map, step)
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