#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN

## Local
from mhdm.tfops.models import NbitTree
from mhdm.tfops.metrics import RegularizedCrossentropy, RegularizedCosine
from mhdm.tfops.callbacks import NbitTreeCallback, LogCallback


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
		'--learning_rate',
		metavar='Float',
		type=float,
		default=0.001,
		help="Learning rate for the Adam optimizer (default=0.001)"
		)
	
	main_args.add_argument(
		'--monitor',
		metavar='STR',
		default=None,
		help='Choose the metric to be monitored for checkpoints and early stopping (default=automatic)'
		)

	main_args.add_argument(
		'--save_best_only',
		action='store_true',
		help="Whether to save only best model or (default) not"
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
		metavar='FLOAT',
		nargs='+',
		type=float,
		default=None,
		help='Bit permutation'
		)
	
	main_args.add_argument(
		'--scale',
		metavar='FLOAT',
		nargs='+',
		type=float,
		default=None,
		help='Bit permutation'
		)
	
	main_args.add_argument(
		'--payload',
		action='store_true',
		help="Whether to prune the tree and separate point details as payload (default=False)"
		)
	
	main_args.add_argument(
		'--spherical',
		action='store_true',
		help="Whether to transform the point-clouds to polar coordinates (default=False)"
		)
	
	main_args.add_argument(
		'--keypoints',
		metavar='Float',
		type=float,
		default=0.0,
		help="A threshold for the keypoint detector (default=disabled)"
		)
	
	main_args.add_argument(
		'--convolutions', '-C',
		metavar='INT',
		type=int,
		default=0,
		help='Number of convolutions'
		)
	
	main_args.add_argument(
		'--branches',
		metavar='STR',
		nargs='+',
		choices=('uids', 'pos', 'pivots', 'meta'),
		default=('uids', 'pos', 'pivots', 'meta'),
		help='Sort the bits according their probabilities (default=None)'
		)
	
	main_args.add_argument(
		'--dense', '-D',
		metavar='INT',
		type=int,
		default=0,
		help='Number of dense layers'
		)
	
	main_args.add_argument(
		'--kernels', '-k',
		metavar='INT',
		type=int,
		default=16,
		help='num of kernel units'
		)
	
	main_args.add_argument(
		'--activation',
		metavar='STR',
		type=str,
		default='softmax',
		help="The final activation function"
		)
	
	main_args.add_argument(
		'--loss',
		metavar='STR',
		type=str,
		default='regularized_crossentropy',
		help="The final activation function"
		)
	
	main_args.add_argument(
		'--floor',
		metavar='FLOAT',
		type=float,
		default=0.0,
		help='Probability floor, added to the estimated probabilities'
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
	
	main_args.add_argument(
		'--checkpoint',
		metavar='PATH',
		help='Resume from a checkpoint'
		)
	return main_args


def main(
	train_index=None,
	val_index=None,
	test_index=None,
	epochs=1,
	learning_rate=0.001,
	monitor=None,
	save_best_only=False,
	stop_patience=-1,
	steps_per_epoch=0,
	fix_subset=False,
	validation_freq=1,
	validation_steps=0,
	test_freq=1,
	test_steps=0,
	dim=2,
	bits_per_dim=[16,16,16,0],
	sort_bits=None,
	permute=None,
	payload=False,
	spherical=False,
	keypoints=False,
	offset=None,
	scale=None,
	kernels=16,
	convolutions=2,
	branches=('uids', 'pos', 'pivots', 'meta'),
	dense=0,
	loss='regularized_cosine',
	activation='softmax',
	floor=0.0,
	log_dir='logs',
	verbose=2,
	cpu=False,
	checkpoint=None,
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
	log_model = os.path.join(log_dir, "ckpts", "nbittree_{epoch:04d}-{loss:.3f}.hdf5")
	log_model_start = os.path.join(log_dir, "ckpts", 'nbittree_start.hdf5')
	log_output = os.path.join(log_dir, timestamp + '.log')
	log_data = os.path.join(log_dir, 'test')
	os.makedirs(os.path.join(log_dir, "ckpts"), exist_ok=True)
	train_index = train_index[0] if train_index and len(train_index) == 1 else train_index
	val_index = val_index[0] if val_index and len(val_index) == 1 else val_index
	test_index = test_index[0] if test_index and len(test_index) == 1 else test_index
	heads = int(np.ceil(sum(bits_per_dim) / dim))

	tflog = tf.get_logger()
	tflog.setLevel(logging.DEBUG)
	fh = logging.FileHandler(log_output)
	tflog.addHandler(fh)
	tflog.info("Main Args:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in log_params.items()]))
	if kwargs:
		tflog.warn("Unrecognized Kwargs:\n" + "\n".join(['\t{} = {}'.format(k,v) for k,v in kwargs.items()]))
	
	model = NbitTree(
		dim=dim,
		kernels=kernels,
		heads=heads,
		convolutions=convolutions,
		branches=branches,
		dense=dense,
		activation=activation,
		floor=floor,
		name=name,
		**kwargs
		)
	
	meta_args = dict(
		bits_per_dim=bits_per_dim,
		sort_bits=sort_bits,
		permute=permute,
		payload=payload,
		spherical=spherical,
		keypoints=keypoints,
		offset=offset,
		scale=scale
		)
	
	trainer, train_encoder, train_meta = model.trainer(train_index, **meta_args) if train_index else (None, None, None)
	validator, val_encoder, val_meta = model.validator(val_index, **meta_args) if val_index else (None, None, None)
	tester, test_encoder, test_meta = model.tester(test_index, **meta_args) if test_index else (None, None, None)
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
		if fix_subset:
			trainer = trainer.take(steps_per_epoch)
	else:
		steps_per_epoch = train_meta.num_of_samples
	
	if validation_steps:
		val_meta.num_of_files = validation_steps
		validation_steps *= val_meta.tree_depth
		if fix_subset:
			validator = validator.take(validation_steps)
	elif val_meta is not None:
		validation_steps = val_meta.num_of_samples
	else:
		validation_steps = 0
	
	if test_steps:
		test_meta.num_of_files = test_steps
		test_steps *= test_meta.tree_depth
		if fix_subset:
			tester = tester.take(test_steps)
	elif test_meta is not None:
		test_steps = test_meta.num_of_samples
	else:
		test_steps = 0

	if loss == 'regularized_cosine':
		loss = RegularizedCosine(msle_smoothing=0.01)
	elif loss == 'regularized_crossentropy':
		loss = RegularizedCrossentropy(msle_smoothing=0.01)
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

	model.compile(
		optimizer=optimizer,
		loss=loss,
		metrics=['accuracy'],
		sample_weight_mode='temporal'
		)
	model.build(meta=master_meta)
	model.summary(print_fn=tflog.info)
	if checkpoint:
		model.load_weights(checkpoint, by_name=checkpoint.endswith('.hdf5'), skip_mismatch=checkpoint.endswith('.hdf5'))
		print(model.branches['uids'].dense.weights)
	model.save_weights(log_model_start)
	tflog.info("Samples for Train: {}, Validation: {}, Test: {}".format(steps_per_epoch, validation_steps, test_steps))
	
	monitor = monitor or 'val_accuracy' if validator else 'accuracy'
	tensorboard = TensorBoard(log_dir=log_dir)
	callbacks = [
		tensorboard,
		ModelCheckpoint(
			log_model,
			save_best_only=save_best_only,
			monitor=monitor
			),
		TerminateOnNaN()
		]
	
	if stop_patience >= 0:
		callbacks.append(EarlyStopping(
			monitor=monitor,
			patience=stop_patience
		))
	
	if test_encoder is not None:
		writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
		when = ['on_test_end' if trainer is None else 'on_epoch_end']
		test_callback = NbitTreeCallback(
			samples=tester,
			info=test_encoder,
			meta=test_meta,
			freq=test_freq,
			steps=test_steps,
			when=when,
			writer=writer,
			range_encode=True,
			output=log_data
			)
		callbacks.append(test_callback)
	
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
	elif tester is not None:
		history = dict()
		test_callback.model = model
		test_callback(history)
		log_callback(history)
	else:
		raise RuntimeError("Unexpected Error!")
	
	tflog.info('Done!')
	return history


if __name__ == '__main__':
	main_args = init_main_args().parse_args()
	main(log_params=main_args.__dict__, **main_args.__dict__)