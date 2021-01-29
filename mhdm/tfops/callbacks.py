
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback


class TestCallback(LambdaCallback):
	"""
	"""
	def __init__(self, tester, tester_args, test_meta,
		test_freq=1,
		test_steps=0,
		when='on_epoch_end',
		writer=None
		):
		"""
		"""
		super(TestCallback, self).__init__(**{when:self.run})
		self.tester = tester
		self.tester_args = tester_args
		self.test_meta = test_meta
		self.test_steps = test_steps if test_steps else test_meta.num_of_samples
		self.test_freq = test_freq
		self.writer = writer

		self.gt_flag_map = np.zeros((1, test_meta.tree_depth, test_meta.output_size, 1))
		self.pred_flag_map = np.zeros((1, test_meta.tree_depth, test_meta.output_size, 1))
		self.compiled_metrics = None
		pass

	def run(self, *args):
		args = (*args[::-1], 0)
		log, step = args[:2]
		if step % self.test_freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		self.gt_flag_map[:] = 0
		self.pred_flag_map[:] = 0
		self.model.reset_metrics()

		for i, sample, args in zip(range(self.test_steps), self.tester, self.tester_args):
			uids, labels, weights = sample
			gt_flags = args[0].numpy()
			layer = args[1].numpy()
			encode = layer == self.test_meta.tree_depth-1
			if layer == 0:
				probs = np.zeros((0, self.test_meta.output_size), dtype=self.test_meta.dtype)
				acc_flags = gt_flags
			else:
				acc_flags = np.concatenate([acc_flags, gt_flags])
			metrics = self.model.test_on_batch(uids, labels, weights, reset_metrics=False, return_dict=True)
			probs, code = self.model.predict_on_batch((encode, labels, probs, acc_flags))
			code = code[0]
			pred_flags = np.argmax(probs[-len(gt_flags):], axis=-1)
			self.pred_flag_map[:, layer, pred_flags, :] += 1
			self.gt_flag_map[:, layer, gt_flags, :] += 1
			if encode and len(code):
				bpp = len(code) * 8 / len(gt_flags)
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