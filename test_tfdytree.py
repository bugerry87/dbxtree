

if __name__ == '__main__':
	import numpy as np
	import mhdm.tfops.dynamictree as dynamictree

	X = np.fromfile('data/0000000000.bin', np.float32).reshape(-1,4)[...,:3]
	bbox = np.abs(X).max(axis=0).astype(np.float32)
	bbox = np.repeat(bbox[None,...], len(X), axis=0)
	nodes = np.ones(len(X), dtype=int)
	radius = 0.0015
	X, nodes, bbox, flags = dynamictree.encode(X, nodes, bbox, radius)
	print(flags)