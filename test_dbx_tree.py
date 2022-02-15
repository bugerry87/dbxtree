import numpy as np
import tensorflow as tf
from scipy.spatial import cKDTree
import mhdm.tfops.dynamictree as dynamictree
from mhdm.bitops import BitBuffer
from mhdm.utils import time_delta
from mhdm.lidar import psnr

if __name__ == '__main__':
	radius = 0.003
	X = np.fromfile('data/0000000000.bin', np.float32).reshape(-1,4)[...,:3]
	bbox = np.abs(X).max(axis=0).astype(np.float32)

	buffer = BitBuffer('data/test_dbx_tree.flg.bin', 'wb')
	buffer.write(int.from_bytes(np.array(radius).astype(np.float32).tobytes(), 'big'), 32, soft_flush=True)
	buffer.write(bbox.shape[-1] * 32, 8, soft_flush=True)
	buffer.write(int.from_bytes(bbox.tobytes(), 'big'), bbox.shape[-1] * 32, soft_flush=True)

	i = np.argsort(bbox)[...,::-1]
	BB = bbox = tf.constant(bbox[i])
	X = x = tf.constant(X[...,i])
	pos = tf.zeros_like(bbox)[None, ...]
	nodes = tf.constant(np.ones(len(X), dtype=np.int64))
	F = []
	
	print("Encoding...")
	delta = time_delta()
	next(delta)
	dims = 3
	while dims:
		x, nodes, pivots, pos, bbox, flags, uids, dims = dynamictree.encode(x, nodes, pos, bbox, radius)
		dims = dims.numpy()
		F.append(flags)
		if dims:
			for f in flags.numpy():
				buffer.write(f, 1<<dims, soft_flush=True)
	buffer.close()
	print("Inference Time:", next(delta))
	
	print("Decoding...")
	bbox = BB
	Y = tf.zeros([1,3], dtype=tf.float32)
	keep = tf.zeros([0,3], dtype=tf.float32)
	for flags in F:
		Y, keep, bbox = dynamictree.decode(flags, bbox, radius, Y, keep)
	print("Inference Time:", next(delta))

	print("Evaluation...")
	Xtree = cKDTree(X)
	Ytree = cKDTree(Y)
	XYdelta, XYnn = Xtree.query(Y)
	YXdelta, YXnn = Ytree.query(X)
	print("PSNR XY:", psnr(np.mean(XYdelta**2)))
	print("PSNR YX:", psnr(np.mean(YXdelta**2)))
	Y.numpy().tofile('data/test_dbx_tree.bin')