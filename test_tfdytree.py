

if __name__ == '__main__':
	import numpy as np
	import tensorflow as tf
	import mhdm.tfops.dynamictree as dynamictree
	from mhdm.bitops import BitBuffer 

	buffer = BitBuffer('data/tf_dyntree.bin', 'wb')

	X = np.fromfile('data/0000000000.bin', np.float32).reshape(-1,4)[...,:3]
	bbox = np.abs(X).max(axis=0).astype(np.float32)
	i = np.argsort(bbox)[...,::-1]
	bbox = tf.constant(np.repeat(bbox[None,...,i], len(X), axis=0))
	X = tf.constant(X[...,i])
	nodes = tf.constant(np.ones(len(X), dtype=np.int64))
	radius = 0.0015
	while len(X.numpy()):
		X, nodes, bbox, flags, shifts = dynamictree.encode(X, nodes, bbox, radius)
		print(len(X.numpy()))
		for flag, shift in zip(flags.numpy(), shifts.numpy()):
			buffer.write(flag, shift, soft_flush=True)
	buffer.close()