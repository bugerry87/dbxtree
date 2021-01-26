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
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

## Local
from mhdm.tfops.models import NbitTreeProbEncoder
from mhdm.tfops.metrics import FlatTopKAccuracy
from mhdm.tfops.callbacks import TestCallback, LogCallback


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
		'--monitor',
		metavar='STR',
		default=None,
		help='Choose the metric to be monitored for checkpoints and early stopping (default=automatic)'
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
		'--bits_per_dim', '-B',
		metavar='INT',
		nargs='+',
		type=int,
		default=[16,16,16,0],
		help='Bit quantization per input dim'
		)
	
	main_args.add_argument(
		'--sort_bits',
		metavar='STR',
		nargs='*',
		choices=['absolute', 'reverse'],
		default=None,
		help='Sort the bits according their probabilities (default=None)'
		)

	main_args.add_argument(
		'--permute',
		metavar='INT',
		nargs='+',
		type=int,
		default=None,
		help='Fixed bit permutation (overrides sort_bits!)'
		)
	
	main_args.add_argument(
		'--offset',
		metavar='INT',
		nargs='+',
		type=float,
		default=None,
		help='Bit permutation'
		)
	
	main_args.add_argument(
		'--scale',
		metavar='INT',
		nargs='+',
		type=float,
		default=None,
		help='Bit permutation'
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
		'--kernel', '-k',
		metavar='INT',
		type=int,
		default=16,
		help='kernel size'
		)
	
	main_args.add_argument(
		'--normalize', '-n',
		action='store_true',
		help="Whether to normalize the transformer or (default) not"
		)
	
	main_args.add_argument(
		'--smoothing',
		metavar='FLOAT',
		type=float,
		default=0,
		help="Set a value from [0..1) for label smoothing (default=0)"
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


def main(
	train_index=None,
	val_index=None,
	test_index=None,
	epochs=1,
	monitor=None,
	stop_patience=-1,
	steps_per_epoch=0,
	validation_freq=1,
	validation_steps=0,
	test_freq=1,
	test_steps=0,
	dim=2,
	bits_per_dim=[16,16,16,0],
	sort_bits=None,
	permute=None,
	offset=None,
	scale=None,
	kernel=16,
	transformers=4,
	convolutions=2,
	normalize=False,
	smoothing=0,
	topk=5,
	log_dir='logs',
	verbose=2,
	cpu=False,
	name=None,
	log_params={},
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
	tflog.info("Main Args:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in log_params.items()]))
	if kwargs:
		tflog.warn("Unrecognized Kwargs:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in kwargs.items()]))
	
	model = NbitTreeProbEncoder(
		dim=dim,
		k=kernel,
		transformers=transformers,
		convolutions=convolutions,
		normalize=normalize,
		name=name,
		**kwargs
		)
	
	quant_args = dict(bits_per_dim=bits_per_dim, sort_bits=sort_bits, permute=permute, offset=offset, scale=scale)
	trainer, train_args, train_meta = model.trainer(train_index, **quant_args, smoothing=smoothing) if train_index else (None, None, None)
	validator, val_args, val_meta = model.validator(val_index, **quant_args) if val_index else (None, None, None)
	tester, tester_args, test_meta = model.tester(test_index, **quant_args) if test_index else (None, None, None)
	master_meta = train_meta or val_meta or test_meta

	if master_meta is None:
		msg = "Main: No index file was set!"
		tflog.error(msg)
		raise ValueError(msg)

	if train_meta is None:
		pass
	elif steps_per_epoch:
		train_meta.num_of_files = steps_per_epoch
		steps_per_epoch *= train_meta.tree_depth
		trainer = trainer.take(steps_per_epoch)
	else:
		steps_per_epoch = train_meta.num_of_samples
	
	if validation_steps:
		val_meta.num_of_files = validation_steps
		validation_steps *= val_meta.tree_depth
		validator = validator.take(validation_steps)
	elif val_meta is not None:
		validation_steps = val_meta.num_of_samples
	else:
		validation_steps = 0
	
	if test_steps:
		test_meta.num_of_files = test_steps
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
	
	monitor = monitor or 'val_accuracy' if validator else 'accuracy'
	tensorboard = TensorBoard(log_dir=log_dir)
	callbacks = [
		tensorboard,
		ModelCheckpoint(
			log_model,
			save_best_only=True,
			monitor=monitor
			)
		]
	
	if stop_patience >= 0:
		callbacks.append(EarlyStopping(
			monitor=monitor,
			patience=stop_patience
			))
	
	if tester is not None:
		writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
		method = 'on_test_end' if trainer is None else 'on_epoch_end'
		test_callback = TestCallback(tester, tester_args, test_meta, test_freq, test_steps, method, writer)
		callbacks.append(test_callback)
	
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
	main(log_params=main_args.__dict__, **main_args.__dict__)