#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import Callback, TensorBoard, ModelCheckpoint, EarlyStopping

## Local
from mhdm.tfops.models import range_encoder
from mhdm.tfops.models import NbitTreeProbEncoder
from mhdm.tfops.metrics import FlatTopKAccuracy


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="NbitTreeTrainer",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--train_index', '-X',
		metavar='PATH',
		nargs='*',
		help='A index file to training data'
		)
	
	main_args.add_argument(
		'--val_index', '-Y',
		metavar='PATH',
		nargs='*',
		default=None,
		help='A index file to validation data'
		)
	
	main_args.add_argument(
		'--test_index', '-T',
		metavar='PATH',
		nargs='*',
		default=None,
		help='A index file to test data'
		)
	
	main_args.add_argument(
		'--epochs', '-e',
		metavar='INT',
		type=int,
		default=1,
		help='Num of epochs'
		)
	
	main_args.add_argument(
		'--stop_patience',
		metavar='INT',
		type=int,
		default=-1,
		help='The early stopping patience (deactivate = -1)'
		)
	
	main_args.add_argument(
		'--steps_per_epoch',
		metavar='INT',
		type=int,
		default=0,
		help='Define to train on a subset'
		)
	
	main_args.add_argument(
		'--validation_freq',
		metavar='INT',
		type=int,
		default=1,
		help="Validation frequency"
		)
	
	main_args.add_argument(
		'--validation_steps',
		metavar='INT',
		type=int,
		default=0,
		help='Define to validate on a subset'
		)
	
	main_args.add_argument(
		'--test_freq',
		metavar='INT',
		type=int,
		default=1,
		help="Test frequency (default=1)"
		)
	
	main_args.add_argument(
		'--test_steps',
		metavar='INT',
		type=int,
		default=0,
		help='Define for test on a subset'
		)
	
	main_args.add_argument(
		'--dim', '-d',
		metavar='INT',
		type=int,
		default=2,
		help='Dimensionality of the tree'
		)
	
	main_args.add_argument(
		'--bits_per_dim', '-b',
		metavar='INT',
		nargs='+',
		type=int,
		default=[16,16,16,0],
		help='Bit quantization per input dim'
		)
	
	main_args.add_argument(
		'--kernel', '-k',
		metavar='INT',
		type=int,
		default=16,
		help='kernel size'
		)
	
	main_args.add_argument(
		'--transformers', '-t',
		metavar='INT',
		type=int,
		default=4,
		help='Number of transformers'
		)
	
	main_args.add_argument(
		'--convolutions', '-c',
		metavar='INT',
		type=int,
		default=0,
		help='Number of convolutions'
		)
	
	main_args.add_argument(
		'--normalize', '-n',
		action='store_true',
		help="Whether to normalize the transformer or (default) not"
		)
	
	main_args.add_argument(
		'--topk',
		metavar='INT',
		type=int,
		default=5,
		help="Considering top k (default=5)"
		)
	
	main_args.add_argument(
		'--log_dir',
		metavar='PATH',
		default='logs',
		help="Model type (default=logs)"
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		metavar='INT',
		type=int,
		default=1,
		help="verbose level (see tensorflow)"
		)
	
	main_args.add_argument(
		'--cpu',
		action='store_true',
		help="Whether to allow cpu or (default) force gpu execution"
		)
	return main_args


class TestCallback(Callback):
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
		super(TestCallback, self).__init__()
		self.tester = tester
		self.tester_args = tester_args
		self.test_meta = test_meta
		self.test_steps = test_steps if test_steps else test_meta.num_of_samples
		self.test_freq = test_freq
		self.when = when
		self.writer = writer

		self.codes = list()
		self.gt_flag_map = np.zeros((1, test_meta.tree_depth, test_meta.output_size, 1))
		self.pred_flag_map = np.zeros((1, test_meta.tree_depth, test_meta.output_size, 1))
		self.compiled_metrics = None
		pass

	def run(self, step, log):
		if step % self.test_freq != 0:
			return
		
		bpp_sum = 0
		bpp_min = 0
		bpp_max = 0
		self.gt_flag_map[:] = 0
		self.pred_flag_map[:] = 0
		self.model.reset_metrics()

		for _, sample, args in zip(range(self.test_steps), self.tester, self.tester_args):
			uids, labels, weights = sample
			gt_flags = args[0].numpy()
			layer = args[1].numpy()
			if layer == 0:
				probs = np.zeros((0, self.test_meta.output_size), dtype=self.test_meta.dtype)
				acc_flags = gt_flags
			else:
				acc_flags = np.concatenate([acc_flags, gt_flags])
			metrics = self.model.test_on_batch(uids, labels, weights, reset_metrics=False, return_dict=True)
			probs, code = self.model.predict_on_batch((uids, probs, acc_flags))
			code = np.squeeze(code)
			pred_flags = np.argmax(probs, axis=-1)
			self.pred_flag_map[:, layer, pred_flags, :] += 1
			self.gt_flag_map[:, layer, gt_flags, :] += 1
			if layer == self.test_meta.tree_depth-1 and code:
				bpp = len(code) * 8 / len(gt_flags)
				bpp_min = min(bpp_min, bpp)
				bpp_max = min(bpp_max, bpp)
				bpp_sum += bpp
		
		if code:
			metrics['bpp'] = bpp_sum / self.test_steps
			metrics['bpp_min'] = bpp_min
			metrics['bpp_max'] = bpp_max
		
		for name, metric in metrics.items():
			name = 'test_' + name
			log[name] = metric
		
		if self.writer is not None:
			self.flag_map /= self.flag_map.max()
			with self.writer.as_default():
				for name, metric in metrics.items():
					name = 'epoch_' + name
					tf.summary.scalar(name, metric, step)
				tf.summary.image('gt_flag_map', self.gt_flag_map, step)
				tf.summary.image('pred_flag_map', self.pred_flag_map, step)
			self.writer.flush()
		pass

	def on_epoch_end(self, epoch, log):
		if self.when == 'on_epoch_end':
			self.run(epoch, log)
	
	def on_test_end(self, log):
		if self.when == 'on_test_end':
			self.run(0, log)


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
	
	def on_train_end(self, epoch, log):
		if self.msg:
			self.logger.info(self.msg)


def main(
	train_index=None,
	val_index=None,
	test_index=None,
	epochs=1,
	stop_patience=-1,
	steps_per_epoch=0,
	validation_freq=1,
	validation_steps=0,
	test_freq=1,
	test_steps=0,
	dim=2,
	bits_per_dim=[16,16,16,0],
	kernel=16,
	transformers=4,
	convolutions=2,
	normalize=False,
	topk=5,
	log_dir='logs',
	verbose=2,
	cpu=False,
	name=None,
	params={},
	**kwargs
	):
	"""
	"""
	if not cpu:
		assert len(tf.config.list_physical_devices('GPU')) > 0
		assert tf.test.is_built_with_cuda()

	tf.summary.trace_on(graph=True, profiler=False)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(log_dir, timestamp)
	log_model = os.path.join(log_dir, "ckpts", "nbittree_{}".format(timestamp))
	log_output = os.path.join(log_dir, timestamp + '.log')
	os.makedirs(log_dir, exist_ok=True)
	train_index = train_index[0] if train_index and len(train_index) == 1 else train_index
	val_index = val_index[0] if val_index and len(val_index) == 1 else val_index
	test_index = test_index[0] if test_index and len(test_index) == 1 else test_index

	tflog = tf.get_logger()
	tflog.setLevel(logging.DEBUG)
	fh = logging.FileHandler(log_output)
	tflog.addHandler(fh)
	tflog.info("Main Args:\n".join(['\t {} = {}'.format(k,v) for k,v in params.items()]))
	
	model = NbitTreeProbEncoder(
		dim=dim,
		k=kernel,
		transformers=transformers,
		convolutions=convolutions,
		normalize=normalize,
		name=name,
		**kwargs
		)
	
	trainer, train_args, train_meta = model.trainer(train_index, bits_per_dim) if train_index else (None, None, None)
	validator, val_args, val_meta = model.validator(val_index, bits_per_dim) if val_index else (None, None, None)
	tester, tester_args, test_meta = model.tester(test_index, bits_per_dim) if test_index else (None, None, None)
	master_meta = train_meta or val_meta or test_meta

	if master_meta is None:
		msg = "Main: No index file was set!"
		tflog.error(msg)
		raise ValueError(msg)

	if train_meta is None:
		pass
	elif steps_per_epoch:
		steps_per_epoch *= train_meta.tree_depth
		trainer = trainer.take(steps_per_epoch)
	else:
		steps_per_epoch = train_meta.num_of_samples
	
	if validation_steps is not None:
		validation_steps *= val_meta.tree_depth
		validator = validator.take(validation_steps)
	elif val_meta is not None:
		validation_steps = val_meta.num_of_samples
	else:
		validation_steps = 0
	
	if test_steps is not None:
		test_steps *= test_meta.tree_depth
		pass
	elif test_meta is not None:
		test_steps = test_meta.num_of_samples
	else:
		test_steps = 0

	topk = FlatTopKAccuracy(topk, classes=master_meta.output_size, name='top{}'.format(topk))
	model.compile(
		optimizer='adam', 
		loss='categorical_crossentropy',
		metrics=['accuracy', topk],
		sample_weight_mode='temporal'
		)
	model.build(tf.TensorShape([1, None, master_meta.word_length]))
	model.summary(print_fn=tflog.info)
	tflog.info("Samples for Train: {}, Validation: {}, Test: {}".format(steps_per_epoch, validation_steps, test_steps))
	
	tensorboard = TensorBoard(log_dir=log_dir)
	callbacks = [
		tensorboard,
		ModelCheckpoint(
			log_model,
			save_best_only=True,
			monitor='val_accuracy' if validator is not None else 'accuracy'
			)
		]
	
	if stop_patience >= 0:
		callbacks.append(EarlyStopping(
			monitor='val_accuracy' if validator is not None else 'accuracy',
			patience=stop_patience
			))
	
	if tester is not None:
		writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
		method = 'on_test_end' if trainer is None else 'on_epoch_end'
		test_callback = TestCallback(tester, tester_args, test_meta, test_freq, test_steps, method, writer)
		callbacks.append(test_callback)
		if range_encoder is None:
			tflog.warning(
				"Model has no range_encoder and will only return raw probabilities." \
				"Please install 'tensorflow-compression' to obtain encoded bit-streams."
				)
	
	callbacks.append(LogCallback(tflog))

	if trainer is not None:
		history = model.fit(
			trainer.repeat(epochs),
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			callbacks=callbacks,
			validation_freq=validation_freq,
			validation_data=validator.repeat(epochs),
			validation_steps=validation_steps,
			verbose=verbose
			)
	elif validator is not None:
		history = model.evaluate(
			validator,
			steps=validation_steps,
			callbacks=callbacks,
			verbose=verbose,
			return_dict=True
			)
	elif tester is not None:
		history = dict()
		test_callback.run(history)
	else:
		raise RuntimeError("Unexpected Error!")
	return history

if __name__ == '__main__':
	main_args = init_main_args().parse_args()
	main(params=main_args.__dict__, **main_args.__dict__)