#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging

## Installed
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN

## Local
from mhdm.tfops.models import EntropyMapper
from mhdm.tfops.callbacks import LogCallback


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="EntropyMapTrainer",
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
		'--fix_subset',
		action='store_true',
		help='Whether the subset should be fixed to the first few samples or (default) not'
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
		'--bits_per_dim', '-B',
		metavar='INT',
		nargs='+',
		type=int,
		default=[16,16,16,0],
		help='Bit quantization per input dim'
		)
	
	main_args.add_argument(
		'--offset', '-O',
		metavar='FLOAT',
		nargs='+',
		type=float,
		default=None,
		help='Bit permutation'
		)
	
	main_args.add_argument(
		'--scale', '-S',
		metavar='FLOAT',
		nargs='+',
		type=float,
		default=None,
		help='Bit permutation'
		)
	
	main_args.add_argument(
		'--kernels', '-k',
		metavar='INT',
		type=int,
		default=16,
		help='num of kernel units'
		)
	
	main_args.add_argument(
		'--kernel_size', '-K',
		metavar='INT',
		type=int,
		default=3,
		help='size of the kernels'
		)
	
	main_args.add_argument(
		'--strides', '-s',
		metavar='INT',
		type=int,
		default=3,
		help='step size of the kernels'
		)
	
	main_args.add_argument(
		'--layers', '-L',
		metavar='INT',
		type=int,
		default=0,
		help='Number of layers'
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
	epochs=1,
	monitor=None,
	stop_patience=-1,
	steps_per_epoch=0,
	fix_subset=False,
	validation_freq=1,
	validation_steps=0,
	bits_per_dim=[16,16,16,0],
	offset=None,
	scale=None,
	kernels=16,
	kernel_size=3,
	strides=2,
	layers=2,
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
	
	strategy = tf.distribute.MirroredStrategy()
	tf.summary.trace_on(graph=True, profiler=False)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(log_dir, timestamp)
	log_model = os.path.join(log_dir, "ckpts", "entropymap_{}".format(timestamp))
	log_output = os.path.join(log_dir, timestamp + '.log')
	os.makedirs(log_dir, exist_ok=True)
	train_index = train_index[0] if train_index and len(train_index) == 1 else train_index
	val_index = val_index[0] if val_index and len(val_index) == 1 else val_index

	tflog = tf.get_logger()
	tflog.setLevel(logging.DEBUG)
	fh = logging.FileHandler(log_output)
	tflog.addHandler(fh)
	tflog.info("Main Args:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in log_params.items()]))
	if kwargs:
		tflog.warn("Unrecognized Kwargs:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in kwargs.items()]))
	
	tflog.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
	
	with strategy.scope():
		model = EntropyMapper(
			bins=sum(bits_per_dim),
			kernels=kernels,
			kernel_size=kernel_size,
			strides=strides,
			layers=layers,
			name=name,
			**kwargs
			)
	
	meta_args = dict(
		bits_per_dim=bits_per_dim,
		offset=offset,
		scale=scale
		)
	
	trainer, train_meta = model.trainer(train_index, **meta_args) if train_index else (None, None, None)
	validator, val_meta = model.validator(val_index, **meta_args) if val_index else (None, None, None)
	master_meta = train_meta or val_meta

	if master_meta is None:
		msg = "Main: No index file was set!"
		tflog.error(msg)
		raise ValueError(msg)

	if train_meta is None:
		pass
	elif steps_per_epoch:
		train_meta.num_of_samples = steps_per_epoch
		if fix_subset:
			trainer = trainer.take(steps_per_epoch)
	else:
		steps_per_epoch = train_meta.num_of_samples
	
	if validation_steps:
		val_meta.num_of_samples = validation_steps
		if fix_subset:
			validator = validator.take(validation_steps)
	elif val_meta is not None:
		validation_steps = val_meta.num_of_samples
	else:
		validation_steps = 0
	
	with strategy.scope():
		model.compile(
			optimizer='adam',
			loss='mse',
			sample_weight_mode='temporal'
			)
		model.build()
	model.summary(print_fn=tflog.info)
	tflog.info("Samples for Train: {}, Validation: {}".format(steps_per_epoch, validation_steps))
	
	monitor = monitor or 'val_loss' if validator else 'loss'
	tensorboard = TensorBoard(log_dir=log_dir)
	callbacks = [
		tensorboard,
		ModelCheckpoint(
			log_model,
			save_best_only=True,
			monitor=monitor
			),
		TerminateOnNaN()
		]
	
	if stop_patience >= 0:
		callbacks.append(EarlyStopping(
			monitor=monitor,
			patience=stop_patience
			))
	
	log_callback = LogCallback(tflog)
	callbacks.append(log_callback)

	if trainer is not None:
		history = model.fit(
			trainer.repeat(),
			epochs=epochs,
			steps_per_epoch=steps_per_epoch,
			callbacks=callbacks,
			validation_freq=validation_freq,
			validation_data=validator,
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
	else:
		raise RuntimeError("Unexpected Error!")
	tflog.info('Done!')
	return history


if __name__ == '__main__':
	main_args = init_main_args().parse_args()
	main(log_params=main_args.__dict__, **main_args.__dict__)