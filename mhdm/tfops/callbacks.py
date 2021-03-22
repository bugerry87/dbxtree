
## Build In
import os.path as path

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
from ..bitops import BitBuffer


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

		if self.meta.mode > 0:
			self.mode = self.flag_mode
		elif self.meta.mode == 0:
			self.mode = self.counter_mode
		else:
			self.mode = self.overflow_mode
		pass

	def flag_mode(self, step, sample, info, tree_start, tree_end):
		feature = sample[0]
		flags = info[2]
		encode = tree_end and self.range_encode

		if tree_start:
			probs = tf.zeros((0, self.meta.bins), dtype=self.meta.dtype)
			acc_flags = flags	
		else:
			acc_flags = tf.concat([acc_flags, flags], axis=-1)

		probs, code = self.model.predict_on_batch((feature, probs, acc_flags, encode))
		code = code[0]
		bits = int(len(code)*8)
		return probs, code, bits

	def overflow_mode(self, step, sample, info, tree_start, tree_end):
		feature = sample[0]
		hist = info[3].numpy()
		layer = info[4].numpy()
		counts = hist.sum(axis=-1)
		bits = np.maximum(np.floor(np.log2(counts+1)), 1).astype(hist.dtype)
		mask = (1<<bits) - 1

		probs = self.model.predict_on_batch(feature)
		pred_minor = np.argmin(probs, axis=-1)[...,None]
		code = np.take_along_axis(hist, pred_minor, axis=-1).flatten()
		code = np.minimum(code, mask)

		gt = np.argmin(hist, axis=-1)
		gt_minor = np.take_along_axis(hist, gt[...,None], axis=-1).flatten()
		more = counts > 1
		sym_overflow = (hist[...,0] == hist[...,1]) & more
		pred_overflow = (code == mask) & more
		first_overflow = pred_overflow | sym_overflow
		second_overflow = (gt_minor >= mask) & ~sym_overflow & more
		gt_minor = np.minimum(gt_minor, mask)

		code[first_overflow] = mask[first_overflow] << bits[first_overflow] + gt_minor[first_overflow]
		code[second_overflow] <<= 1
		code[second_overflow] |= gt[second_overflow]
		bits[first_overflow] *= 2
		bits[second_overflow] += 1
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
			filename = str(info[5].numpy())
			tree_start = step % self.meta.tree_depth == 0
			tree_end = (step+1) % self.meta.tree_depth == 0
			metrics = self.model.test_on_batch(feature, labels, reset_metrics=False, return_dict=True)
			probs, code, bits = self.mode(step, sample, info, tree_start, tree_end)

			if tree_start:
				bit_count = 0
				if self.buffer:
					filename = path.join(self.output, path.splitext(path.basename(filename))[0] + '.nbit.bin')
					self.buffer.open(filename, 'wb')
			
			if self.buffer:
				for c, b in zip(code, bits):
					self.buffer.write(c, b, soft_flush=True)
			bit_count += int(sum(bits))
			
			if tree_end:
				points = int(info[3].numpy().sum())
				bpp = bit_count / points
				bpp_min = min(bpp_min, bpp)
				bpp_max = max(bpp_max, bpp)
				bpp_sum += bpp
				if self.buffer:
					self.buffer.close()

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