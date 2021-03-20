
## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from .. import bitops


class NbitTreeCallback(LambdaCallback):
	"""
	"""
	def __init__(self, samples, info, meta,
		freq=1,
		steps=0,
		when=['on_epoch_end'],
		writer=None,
		range_encode=True,
		binary=False,
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
		self.binary = binary

		if self.meta.dim > 0:
			self.mode = flag_mode
		else:
			self.mode = counter_mode
		pass

	def flag_mode(self, step, sample, info):
		feature = sample[0]
		flags = info[2]
		layer = info[4].numpy()
		encode = self.range_encode and layer == self.meta.tree_depth-1

		if layer == 0:
			probs = tf.zeros((0, self.meta.bins), dtype=self.meta.dtype)
			acc_flags = flags	
		else:
			acc_flags = tf.concat([acc_flags, flags], axis=-1)

		probs, code = self.model.predict_on_batch((feature, probs, acc_flags, encode))
		code = code[0]
		bits = int(len(code)*8)
		return probs, code, bits

	def counter_mode(self, step, sample, info):
		feature = sample[0]
		hist = info[3].numpy()
		layer = info[4].numpy()
		counts = hist.sum(axis=-1, keepdims=True)
		bits = np.max(np.floor(np.log2(counts+1)), 1).astype(hist.dtype)
		mask = (1<<bits) - 1

		probs = self.model.predict_on_batch(feature)
		pred_minor = np.argmin(probs, axis=-1)[...,None]
		symbol = np.take_along_axis(hist, pred_minor, axis=-1)
		symbol = np.min(symbol, mask)

		gt = np.argmin(hist, axis=-1)[...,None]
		gt_minor = np.take_along_axis(hist, gt, axis=-1)
		sym_overflow = (hist[...,0] == hist[...,1]) & (counts > 1)
		pred_overflow = (symbol == mask) & (counts > 1)
		first_overflow = pred_overflow | sym_overflow
		second_overflow = (gt_minor >= mask) & ~sym_overflow & (counts > 1)
		gt_minor = np.min(gt_minor, mask)

		symbol[first_overflow] = mask[first_overflow] << bits[first_overflow] + gt_minor[first_overflow]
		symbol[second_overflow] <<= 1
		symbol[second_overflow] |= gt
		bits[first_overflow] *= 2
		bits[second_overflow] += 1

		bits = int(bits.sum())
		return probs, code, bits

	def __call__(self, *args):
		args = (*args[::-1], 0)
		log, epoch = args[:2]
		if epoch % self.freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = (1<<32)-1
		bpp_max = 0
		self.model.reset_metrics()

		for step, sample, info in zip(range(self.steps), self.samples, self.info):
			feature, labels = sample[:2]
			metrics = self.model.test_on_batch(feature, labels, reset_metrics=False, return_dict=True)
			probs, code, bits = self.mode(step, sample, info)

			if bits:
				points = int(info[3].numpy().sum())
				bpp = bits / points
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp

		if bits:
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