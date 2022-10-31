#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging
import pickle

## Installed
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN
from tensorflow.keras.losses import MeanSquaredLogarithmicError as MSLE, BinaryCrossentropy as BC
from tensorflow.keras.metrics import BinaryAccuracy, Accuracy, RootMeanSquaredError

## Local
from mhdm.tfops.models import DynamicTree2 as DynamicTree
from mhdm.tfops.metrics import FocalLoss, CombinedLoss, SlicedMetric
from mhdm.tfops.callbacks import DynamicTree2Callback, SaveOptimizerCallback, LogCallback
from mhdm import utils


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="DynamicTreeTrainer",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--train_index', '-X',
		metavar='PATH',
		nargs='*',
		default='./data/train_index.txt',
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
		'--xshape',
		metavar='SHAPE',
		type=int,
		nargs='+',
		default=(-1, 4),
		help='Shape of the input data'
		)
	
	main_args.add_argument(
		'--xtype',
		metavar='TYPE',
		default='float32',
		help='Type of the input data'
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
		'--max_layers',
		metavar='INT',
		type=int,
		default=17,
		help='Max layers to encode before early stopping'
		)
	
	main_args.add_argument(
		'--shuffle',
		metavar='INT',
		type=int,
		default=0,
		help="Size of the shuffle buffer"
		)
	
	main_args.add_argument(
		'--radius', '-r',
		metavar='Float',
		type=float,
		default=0.1,
		help="Radius of voxel precision"
		)
	
	main_args.add_argument(
		'--convolutions', '-C',
		metavar='INT',
		type=int,
		default=1,
		help='Number of convolutions'
		)
	
	main_args.add_argument(
		'--features',
		metavar='STR',
		nargs='+',
		choices=('org', 'pos', 'uids'),
		default=('org', 'pos', 'uids'),
		help='Sort the bits according their probabilities (default=None)'
		)
	
	main_args.add_argument(
		'--dense', '-D',
		metavar='INT',
		type=int,
		default=1,
		help='Number of dense layers'
		)
	
	main_args.add_argument(
		'--kernels', '-k',
		metavar='INT',
		type=int,
		default=4,
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
		'--range_encoder',
		metavar='CODER',
		default='tfc',
		choices=('tfc', 'python', None),
		help='Chose range coder implementation'
		)
	
	main_args.add_argument(
		'--floor',
		metavar='FLOAT',
		type=float,
		default=0.001,
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
		help='Load from checkpoint'
		)
	
	main_args.add_argument(
		'--training_state',
		metavar='PATH',
		help='Resume from a training_state'
		)
	return main_args


def main(
	train_index=None,
	val_index=None,
	test_index=None,
	xshape=(-1,4),
	xtype='float32',
	epochs=1,
	learning_rate=0.001,
	monitor=None,
	save_best_only=False,
	stop_patience=-1,
	steps_per_epoch=0,
	validation_freq=1,
	validation_steps=0,
	test_freq=1,
	test_steps=0,
	max_layers=17,
	shuffle=0,
	radius=0.003,
	kernels=4,
	convolutions=2,
	features=('org', 'pos', 'uids'),
	dense=0,
	activation='softmax',
	floor=0.0,
	log_dir='logs',
	verbose=2,
	cpu=False,
	range_encoder='tfc',
	checkpoint=None,
	training_state=None,
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
	log_model = os.path.join(log_dir, "ckpts", "dbx_tree_{epoch:04d}-{loss:.3f}.hdf5")
	log_model_start = os.path.join(log_dir, "ckpts", 'dbx_tree_start.hdf5')
	log_optimizer = os.path.join(log_dir, "ckpts", "dbx_tree_{epoch:04d}-{loss:.3f}.train.pkl")
	log_output = os.path.join(log_dir, timestamp + '.log')
	log_data = os.path.join(log_dir, 'test')
	os.makedirs(os.path.join(log_dir, "ckpts"), exist_ok=True)
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
	
	model = DynamicTree(
		features=features,
		kernels=kernels,
		pre_conv=convolutions,
		post_conv=convolutions,
		dense=dense,
		name=name,
		**kwargs
		)
	
	meta_args = dict(
		radius=radius,
		xshape=xshape,
		xtype=xtype
		)
	
	trainer, train_meta = model.trainer(train_index, 
		take=steps_per_epoch,
		shuffle=shuffle,
		augmentation=True,
		**meta_args) if train_index else (None, None)
	validator, val_meta = model.validator(val_index,
		take=validation_steps,
		**meta_args) if val_index else (None, None)
	tester, test_meta = model.tester(test_index,
		take=test_steps,
		max_layers=max_layers,
		**meta_args) if test_index else (None, None)
	master_meta = train_meta or val_meta or test_meta
	steps_per_epoch = steps_per_epoch if steps_per_epoch * max_layers else master_meta.num_of_files * max_layers

	if master_meta is None:
		msg = "Main: No index file was set!"
		tflog.error(msg)
		raise ValueError(msg)
	
	loss = CombinedLoss([
		utils.Prototype(
			loss=BC(),
			slices=slice(0, model.flag_size)
		),
		utils.Prototype(
			loss = FocalLoss(),
			slices = slice(model.flag_size, model.flag_size + model.bins)
		),
		utils.Prototype(
			loss = MSLE(),
			slices = slice(model.flag_size + model.bins, model.flag_size + model.bins + model.meta.dim)
		)
	])

	metrics = [
		SlicedMetric(BinaryAccuracy('flag_acc'), slice(0, model.flag_size)),
		SlicedMetric(Accuracy('symp_acc'), slice(model.flag_size, model.flag_size + model.bins)),
		SlicedMetric(RootMeanSquaredError('mean_err'), slice(model.flag_size + model.bins, model.flag_size + model.bins + model.meta.dim)),
	]
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	model.compile(
		optimizer=optimizer,
		loss=loss,
		metrics=metrics,
		sample_weight_mode='temporal'
		)
	model.build()
	model.summary(print_fn=tflog.info)
	
	if checkpoint:
		model.load_weights(checkpoint, by_name=checkpoint.endswith('.hdf5'), skip_mismatch=checkpoint.endswith('.hdf5'))
	model.save_weights(log_model_start)

	if training_state:
		with tf.name_scope(optimizer._name):
			with tf.init_scope():
				optimizer._create_all_weights(model.trainable_variables)
		with open(training_state, 'rb') as f:
			model.optimizer.set_weights(pickle.load(f))
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
		SaveOptimizerCallback(
			model.optimizer,
			log_optimizer,
			monitor=monitor,
			save_best_only=save_best_only,
			mode='max'
			),
		TerminateOnNaN()
		]
	
	if stop_patience >= 0:
		callbacks.append(EarlyStopping(
			monitor=monitor,
			patience=stop_patience
		))
	
	if tester is not None:
		writer = tf.summary.create_file_writer(os.path.join(log_dir, 'test'))
		when = ['on_test_end' if trainer is None else 'on_epoch_end']
		test_callback = DynamicTree2Callback(
			samples=tester,
			meta=test_meta,
			freq=test_freq,
			steps=test_steps,
			when=when,
			writer=writer,
			range_encoder=range_encoder,
			floor=floor,
			output=log_data
			)
		callbacks.append(test_callback)
	
	log_callback = LogCallback(tflog)
	callbacks.append(log_callback)

	if trainer is not None:
		history = model.fit(
			trainer.repeat(),
			epochs=epochs,
			steps_per_epoch = steps_per_epoch,
			callbacks=callbacks,
			validation_freq=validation_freq,
			validation_data=validator,
			verbose=verbose
			)
	elif validator is not None:
		history = model.evaluate(
			validator,
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