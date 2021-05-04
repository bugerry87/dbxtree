
## Build In
import os.path as path

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, LambdaCallback

## Local
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

		if self.meta.mode > 0:
			self.mode = self.flag_mode
		elif self.meta.mode == 0:
			self.mode = self.overflow_mode
		else:
			self.mode = self.overflow_regression
		pass

	def flag_mode(self, step, sample, info, tree_start, tree_end):
		feature = sample[0]
		encode = tree_end and self.range_encode
		flags = info[2]
		layer = info[5]
		if self.meta.payload:
			mask = info[-1].numpy()

		if tree_start:
			self.probs = tf.zeros((0, self.meta.bins), dtype=self.meta.dtype)
			self.flags = flags
			if self.meta.payload:
				counts = info[3].numpy().sum()
				self.bits = np.zeros(counts, int)
		else:
			if self.meta.payload:
				self.bits[mask] = self.meta.word_length - (layer.numpy()+1) * self.meta.dim
			self.flags = tf.concat([self.flags, flags], axis=-1)

		self.probs, code = self.model.predict_on_batch((feature, layer, self.probs, self.flags, encode))
		code = code[0]
		if self.meta.payload and tree_end:
			payload = info[-2].numpy()
			bits = self.bits
		else:
			payload = []
			bits = []
		return self.probs, code, payload, bits

	def overflow_mode(self, step, sample, info, tree_start, tree_end):
		feature = sample[0]
		hist = info[4].numpy()
		layer = info[5]
		counts = hist.sum(axis=-1)
		bits = np.maximum(np.floor(np.log2(counts+1)), 1).astype(hist.dtype)
		mask = (1<<bits) - 1

		probs = self.model.predict_on_batch(feature, layer)
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

		code[first_overflow] = mask[first_overflow] << bits[first_overflow] | gt_minor[first_overflow]
		code[second_overflow] <<= 1
		code[second_overflow] |= gt[second_overflow]
		bits[first_overflow] *= 2
		bits[second_overflow] += 1
		return probs, [], code, bits
	
	def overflow_regression(self, step, sample, info, tree_start, tree_end):
		feature, labels = sample[:2]
		labels = labels[0]
		counts = info[3].numpy()
		hist = info[4].numpy()
		layer = info[5]
		probs = self.model.predict_on_batch(feature, layer)
		probs /= np.linalg.norm(probs, ord=1)
		pred = np.argmin(probs, axis=-1)[...,None]
		payload = np.take_along_axis(hist, pred, axis=-1).flatten()
		pred = np.take_along_axis(probs, pred, axis=-1).flatten() * counts
		overflow_bits = np.ceil(np.log2(counts + 1.0)).astype(payload.dtype)
		bits = np.ceil(np.log2(pred + 1.0)).astype(payload.dtype)
		mask = (1<<bits) - 1
		overflow = (counts > 1) & (payload >= mask)
		payload[overflow] = mask[overflow] << overflow_bits[overflow] | payload[overflow]
		bits[overflow] += overflow_bits[overflow]
		code = []
		return probs, code, payload, bits

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
			feature, labels = sample[:2]
			filename = str(info[6].numpy())
			tree_start = step % self.meta.tree_depth == 0
			tree_end = (step+1) % self.meta.tree_depth == 0
			metrics = self.model.test_on_batch(feature, labels, reset_metrics=False, return_dict=True)
			probs, code, payload, bits = self.mode(step, sample, info, tree_start, tree_end)

			if tree_start:
				points = int(info[3].numpy().sum())
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
				for c in code:
					self.buffer.write(c, 8, soft_flush=True)

				for p, b in zip(payload, bits):
					self.buffer.write(p, b, soft_flush=True)
			bit_count += len(code)*8 + int(sum(bits))
			
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