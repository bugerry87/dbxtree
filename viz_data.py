#!/usr/bin/env python3

## Build In
import os.path as path

## Installed
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

## Local
from mhdm.tfops.models import NbitTreeProbEncoder
from mhdm import viz


if __name__ == '__main__':
	index_txt = 'data/index.txt'
	model_name = 'logs/models/transformer-20210105-162817'
	k = 16
	animate = False
	
	model = NbitTreeProbEncoder(2, k, transformers=4, normalize=False)
	model.load_weights(model_name)
	model.build(tf.TensorShape([1,None,48]))
	model.summary()
	
	encoder, meta = model.encoder(index_txt, [16,16,16,0])
	flag_map = np.zeros((meta.tree_depth+1, meta.output_size))
	
	if animate:
		data = encoder.as_numpy_iterator()
		fig = viz.create_figure()
		
		@viz.mlab.animate(delay=10)
		def animation():
			plot = None
			for sample in data:
				uids, flags, layer, offset, scale = sample
				flag_map[layer, flags] += 1
				if layer == 0:
					buffer = tf.constant([0], dtype=tf.int64)
				
				buffer = NbitTreeProbEncoder.decode(flags, meta, buffer)
				X = NbitTreeProbEncoder.finalize(buffer, meta, offset, scale)
				x = X.numpy()
				
				if layer < meta.tree_depth:
					plot = viz.vertices(x, x[:,2], fig, plot)
				fig.render()
				yield

		animator = animation()
		viz.mlab.show()
	
	encoder = encoder.batch(1)
	data = encoder.as_numpy_iterator()
	
	for sample in data:
		uids, flags, layer, offset, scale = sample
		pred = model.predict(uids)
		flags = np.argmax(pred, axis=-1)
		flags = np.squeeze(flags)
		flag_map[layer, flags] += 1
	
	flag_map = flag_map.T[1:]
	flag_map -= flag_map.min(axis=0)
	flag_map /= flag_map.max(axis=0)
	
	y = range(meta.output_size-1)
	labels = ["{:0>4}".format(bin(i+1)[2:]) for i in y]
	fig, ax = plt.subplots()
	ax.set_title("Layer-wise Distribution (Norm)")
	ax.set_ylabel('flags')
	ax.set_xlabel('layers')
	ax.imshow(flag_map)
	ax.set_yticks(y)
	ax.set_yticklabels(labels)
	plt.show()
	