#!/usr/bin/env python3

## Build In
import os
from datetime import datetime
from argparse import ArgumentParser
import logging

## Installed
import numpy as np
import tensorflow as tf

## Local
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
		nargs='+',
		required=True,
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
		help="Test frequency (default=9)"
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
		help="Whether to normalize the transformer or not (default)"
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
	return main_args


def main(
	train_index,
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
	name=None,
	params={},
	**kwargs
	):
	"""
	"""
	tf.summary.trace_on(graph=True, profiler=False)
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	log_dir = os.path.join(log_dir, timestamp)
	log_model = os.path.join(log_dir, "ckpts", "nbittree_{}".format(timestamp))
	log_output = os.path.join(log_dir, timestamp + '.log')
	os.makedirs(log_dir, exist_ok=True)
	train_index = train_index[0] if len(train_index) == 1 else train_index
	val_index = val_index[0] if len(val_index) == 1 else val_index
	test_index = test_index[0] if len(test_index) == 1 else test_index

	tflog = tf.get_logger()
	tflog.setLevel(logging.DEBUG)
	fh = logging.FileHandler(log_output)
	tflog.addHandler(fh)

	tflog.info("Main Args:")
	tflog.info("\n".join(['\t {} = {}'.format(k,v) for k,v in params.items()]))
	
	model = NbitTreeProbEncoder(
		dim=dim,
		k=kernel,
		transformers=transformers,
		convolutions=convolutions,
		normalize=normalize,
		name=name,
		**kwargs
		)
	
	trainer, train_args, train_meta = model.trainer(train_index, bits_per_dim)
	validator, val_args, val_meta = model.validator(val_index, bits_per_dim) if val_index else (None, None, None)
	tester, tester_args, test_meta = model.tester(test_index, bits_per_dim) if test_index else (None, None, None)
	topk = FlatTopKAccuracy(topk, classes=train_meta.output_size, name='top5')

	if steps_per_epoch:
		steps_per_epoch *= train_meta.tree_depth
		trainer = trainer.take(steps_per_epoch)
	else:
		steps_per_epoch = train_meta.num_of_samples
	
	if validation_steps:
		validation_steps *= val_meta.tree_depth
		validator = validator.take(validation_steps)
	elif val_meta is not None:
		validation_steps = val_meta.num_of_samples
	
	if test_steps:
		test_steps *= test_meta.tree_depth
		pass
	elif test_meta is not None:
		test_steps = test_meta.num_of_samples

	model.compile(
		optimizer='adam', 
		loss='categorical_crossentropy',
		metrics=['accuracy', topk],
		sample_weight_mode='temporal'
		)
	model.build(tf.TensorShape([1, None, train_meta.word_length]))
	model.summary(print_fn=tflog.info)
	tflog.info("Samples for Train: {}, Validation: {}, Test: {}".format(steps_per_epoch, validation_steps, test_steps))
	
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
	def run_test(epoch, *args):
		if epoch % test_freq != 0:
			return
		writer = tensorboard._writers['train']
		flag_map = np.zeros((1, train_meta.tree_depth, train_meta.output_size, 1))
		for i, sample, args in zip(range(test_steps), tester, tester_args):
			uids = sample[0]
			layer = args[1]
			pred = model.predict_on_batch(uids)
			flags = np.argmax(pred, axis=-1)
			flag_map[:,layer, flags,:] += 1
		flag_map /= flag_map.max()
		with writer.as_default():
			tf.summary.image('flag_prediction', flag_map, epoch)
		writer.flush()
		pass

	def log(epoch, log):
		tflog.info("Epoch {}: ".format(epoch) + " ".join(['{} = {}'.format(k,v) for k,v in log.items()]))
	
	callbacks = [
		tensorboard,
		tf.keras.callbacks.ModelCheckpoint(
			log_model,
			save_best_only=True,
			monitor='val_accuracy' if validator is not None else 'accuracy'
			),
		tf.keras.callbacks.LambdaCallback(on_epoch_end=log)
		]
	
	if stop_patience >= 0:
		callbacks.append(tf.keras.callbacks.EarlyStopping(
			monitor='val_accuracy' if validator is not None else 'accuracy',
			patience=stop_patience
			))
	
	if tester is not None:
		callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=run_test))

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
	return 0

if __name__ == '__main__':
	main_args = init_main_args().parse_known_args()[0]
	main(params=main_args.__dict__, **main_args.__dict__)