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


if __name__ == '__main__':
	#tf.compat.v1.disable_eager_execution()
	index_txt = 'data/index.txt'
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = path.join("logs", "scalars", timestamp)
	tf.summary.trace_on(graph=True, profiler=False)
	k = 16
	
	optimizer = tf.keras.optimizers.Adam(learning_rate=0.5)
	model = NbitTreeProbEncoder(2, k, transformers=4, normalize=False)
	model.compile(
		optimizer=optimizer, 
		loss='categorical_crossentropy',
		metrics=['accuracy']
		)
	model.build(tf.TensorShape([1,None,48]))
	model.summary()
	
	encoder, meta = model.encoder(index_txt, [16,16,16,0])
	tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logdir)
	
	def monitor_leanring_rate(epoch, *args):
		writer = tensorboard._writers['train']
		with writer.as_default():
			tf.summary.scalar('learning_rate', model.optimizer._decayed_lr('float32'), epoch)
		writer.flush()
		pass
	
	trainer = model.trainer(encoder=encoder)
	callbacks = [
		tensorboard,
		tf.keras.callbacks.LambdaCallback(on_epoch_end=monitor_leanring_rate),
		tf.keras.callbacks.ModelCheckpoint(
			'logs/models/transformer-{}'.format(timestamp),
			save_best_only=True,
			monitor='accuracy'
			)
		]
	
	history = model.fit(
		trainer,
		epochs=100,
		callbacks=callbacks
		)
	
	#sample = dataset.take(3)
	#for input, label in sample.as_numpy_iterator():
	#	pred = model.predict(input)
	#	print(pred, label)
	
	
	#data = encoder.as_numpy_iterator()
	#flags = np.zeros((meta.tree_depth+1, meta.output_size))
	#for sample in data:
	#	uids, flag, layer, offset, scale = sample[0]
	#	flags[layer, flag] += 1
	#
	#print(flags)
	#flags = flags.T[1:]
	#flags -= flags.min(axis=0)
	#flags /= flags.max(axis=0)
	#
	#y = range(meta.output_size-1)
	#labels = ["{:0>4}".format(bin(i+1)[2:]) for i in y]
	#fig, ax = plt.subplots()
	#ax.set_title("Layer-wise Distribution (Norm)")
	#ax.set_ylabel('flags')
	#ax.set_xlabel('layers')
	#ax.imshow(flags)
	#ax.set_yticks(y)
	#ax.set_yticklabels(labels)
	#plt.show()
	#exit()