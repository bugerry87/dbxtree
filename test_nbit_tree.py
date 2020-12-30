#!/usr/bin/env python3

## Build In
import os.path as path
from datetime import datetime

## Installed
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Local
from tfops.models import NbitTreeProbEncoder

if __name__ == '__main__':
	#tf.compat.v1.disable_eager_execution()
	index_txt = 'data/index.txt'
	timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
	logdir = path.join("logs", "scalars", timestamp)
	k = 16
	
	model = NbitTreeProbEncoder(2, k, transformers=1, normalize=False)
	#model.compile(
	#	optimizer='adam', 
	#	loss='categorical_crossentropy', 
	#	metrics=['accuracy']
	#	)
	#model.build(tf.TensorShape([1,None,48]))
	#model.summary()
	
	encoder, meta = model.encoder(index_txt, [16,16,16,0])
	
	data = encoder.as_numpy_iterator()
	flags = np.zeros((meta.tree_depth+1, meta.output_size))
	for sample in data:
		uids, flag, layer, offset, scale = sample[0]
		flags[layer, flag] += 1
	
	flags = flags.T
	flags /= flags.max(axis=0)
	
	y = range(meta.output_size)
	labels = ["{:0>4}".format(bin(i)[2:]) for i in y]
	fig, ax = plt.subplots()
	ax.set_title("Layer-wise Distribution (Norm)")
	ax.set_ylabel('flags')
	ax.set_xlabel('layers')
	ax.imshow(flags)
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	plt.show()
	exit()
	
	encoder = encoder.batch(1)
	callbacks = [
		tf.keras.callbacks.TensorBoard(log_dir=logdir),
		tf.keras.callbacks.ModelCheckpoint(
			'logs/models/transformer-{}'.format(timestamp),
			save_best_only=True,
			monitor='accuracy'
			),
		]
	
	history = model.fit(
		encoder,
		epochs=100,
		callbacks=callbacks
		)
	
	#sample = dataset.take(3)
	#for input, label in sample.as_numpy_iterator():
	#	pred = model.predict(input)
	#	print(pred, label)