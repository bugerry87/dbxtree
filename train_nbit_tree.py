#!/usr/bin/env python3

## Build In
import os.path as path
from datetime import datetime

## Installed
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Local
from mhdm.tfops.models import NbitTreeProbEncoder
from mhdm.tfops.metrics import FlatTopKAccuracy


if __name__ == '__main__':
	#tf.compat.v1.disable_eager_execution()
	tf.summary.trace_on(graph=True, profiler=False)
	index_txt = 'data/index.txt'
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = path.join("logs", "scalars", timestamp)
	quali_inverval = 9
	k = 64
	
	loss = tf.keras.metrics.CategoricalCrossentropy(label_smoothing=0.2)
	topk = FlatTopKAccuracy(classes=16, name='top5')
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
	model = NbitTreeProbEncoder(2, k, transformers=2, normalize=False)
	model.compile(
		optimizer=optimizer, 
		loss=loss,
		metrics=['accuracy', topk],
		sample_weight_mode='temporal'
		)
	model.build(tf.TensorShape([1,None,48]))
	model.summary()
	
	encoder, meta = model.encoder(index_txt, [16,16,16,0])
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)
	
	def monitor_leanring_rate(epoch, logs):
		writer = tensorboard._writers['train']
		lr = model.optimizer._decayed_lr('float32')
		logs['lr'] = lr
		with writer.as_default():
			tf.summary.scalar('learning_rate', lr, epoch)
		writer.flush()
		pass
	
	def qualitative_validation(epoch, *args):
		if epoch % quali_inverval:
			return
		writer = tensorboard._writers['train']
		flag_map = np.zeros((1, meta.tree_depth+1, meta.output_size, 1))
		args = val_args.as_numpy_iterator()
		val_iter = iter(validator)
		for uids, arg in zip(val_iter, args):
			layer = arg[1]
			pred = model.predict_on_batch(uids)
			flags = np.argmax(pred, axis=-1)
			flag_map[:,layer, flags,:] += 1
		flag_map /= flag_map.max()
		with writer.as_default():
			tf.summary.image('flag_prediction', flag_map, epoch)
		writer.flush()
		pass
	
	trainer = model.trainer(encoder=encoder)
	validator, val_args = model.validator(encoder=encoder)
	callbacks = [
		tensorboard,
		tf.keras.callbacks.LambdaCallback(on_epoch_end=monitor_leanring_rate),
		tf.keras.callbacks.ModelCheckpoint(
			'logs/models/transformer-{}'.format(timestamp),
			save_best_only=True,
			monitor='accuracy'
			),
		tf.keras.callbacks.LambdaCallback(on_epoch_end=qualitative_validation)
		]
	
	history = model.fit(
		trainer,
		epochs=100,
		callbacks=callbacks
		)