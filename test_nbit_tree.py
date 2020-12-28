#!/usr/bin/env python3

## Installed
import tensorflow as tf

## Local
from tfops.models import NbitTreeProbEncoder

if __name__ == '__main__':
	#tf.compat.v1.disable_eager_execution()
	index_txt = 'data/index.txt'
	model = NbitTreeProbEncoder(2)
	model.compile(
		optimizer='adam', 
		loss='sparse_categorical_crossentropy', 
		metrics=['accuracy']
		)
	
	model.build(tf.TensorShape([1,None,48]))
	model.summary()
	
	dataset = model.gen_train_data(index_txt, [16,16,16,0])
	dataset = dataset.batch(1)
	
	history = model.fit(
		dataset,
		epochs=10
		)