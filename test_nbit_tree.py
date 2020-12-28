#!/usr/bin/env python3

## Build In
import os.path as path

## Installed
import tensorflow as tf

## Local
from tfops.models import NbitTreeProbEncoder

if __name__ == '__main__':
	tf.compat.v1.disable_eager_execution()
	index_txt = 'data/index.txt'
	
	model = NbitTreeProbEncoder(2, 128, normalize=False)
	model.compile(
		optimizer='adam', 
		loss='categorical_crossentropy', 
		metrics=['accuracy']
		)
	model.build(tf.TensorShape([1,None,48]))
	model.summary()
	
	dataset = model.gen_train_data(index_txt, [16,16,16,0])
	dataset = dataset.batch(1)
	
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path.join("logs"))
	history = model.fit(
		dataset,
		epochs=10,
		callbacks=[tensorboard_callback]
		)
	
	sample = dataset.take(3)
	for input, label in sample.as_numpy_iterator():
		pred = model.predict(input)
		print(pred, label)