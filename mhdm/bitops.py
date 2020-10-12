
## Installed
import numpy as np


def serialize(X, bits):
	pass


def sort_bits(X, reverse=False):
	shifts = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	p = np.array([np.sum(X>>i&1) for i in range(shifts)])
	p = np.max((p, len(Y)-p), axis=0)
	p = np.argsort(p)
	if reverse:
		p = p[::-1]
	
	for i in range(shifts):
		Y |= (X>>p[i]&1)<<i
	return Y.reshape(shape), p.astype(np.uint8)


def unsort_bits(X, p):
	shifts = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	
	for i in range(shifts):
		Y |= (X>>i&1)<<p[i]
	return Y.reshape(shape)


def reverse_bits(X):
	shifts = np.iinfo(X.dtype).bits
	Y = np.zeros_like(X)
	
	for low, high in zip(range(shifts//2), range(shifts-1, shifts//2 - 1, -1)):
		Y |= (X & 1<<low) << high | (X & 1<<high) >> high-low
	return Y
