

if __name__ == '__main__':
	import numpy as np
	import tensorflow as tf
	import mhdm.tfops.dynamictree as dynamictree
	from mhdm.bitops import BitBuffer
	from mhdm.utils import time_delta

	radius = 0.0015
	buffer = BitBuffer('data/tf_dyntree.bin', 'wb')
	X = np.fromfile('data/kitti_20110926_0005/0000000136.bin', np.float32).reshape(-1,4)[...,:3]
	#itest = iter(np.fromfile('data/flags.bin', np.uint8))

	bbox = np.abs(X).max(axis=0).astype(np.float32)
	buffer.write(int.from_bytes(np.array(radius).astype(np.float32).tobytes(), 'big'), 32, soft_flush=True)
	#buffer.write(bbox.shape[-1] * 32, 8, soft_flush=True)
	buffer.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)

	i = np.argsort(bbox)[...,::-1]
	bbox = tf.constant(bbox[i])
	X = tf.constant(X[...,i])
	pos = tf.zeros_like(bbox)[None, ...]
	nodes = tf.constant(np.ones(len(X), dtype=np.int64))
	P = []
	
	delta = time_delta()
	next(delta)
	dim = 3
	while dim:
		X, nodes, pivots, pos, bbox, flags, uids, dim = dynamictree.encode(X, nodes, pos, bbox, radius)
		dim = dim.numpy()
		#input(pivots.numpy())
		P.append(pos.numpy())
		if dim:
			for flag in flags.numpy():
				buffer.write(flag, 1<<dim, soft_flush=True)
	print(next(delta))
	buffer.close()
	np.vstack(P).astype(np.float32).tofile('data/pos')