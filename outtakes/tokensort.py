
## Installed
import numpy as np

## Local
from . import bitops


def encode(X, bits_per_dim,
	qtype=object,
	offset=0.0,
	scale=1.0,
	reverse=False,
	sort=False,
	iterations=0,
	**kwargs
	):
	"""
	"""
	permutes = []
	bits = sum(bits_per_dim)
	X, offset, scale = bitops.serialize(X, bits_per_dim, qtype, offset, scale)
	X, p = bitops.sort(X, reverse)
	if sort:
		X.sort()
		X = np.diff(X, prepend=0).astype(qtype)
	zero_padding = -len(X) % 8	
	X = np.hstack((np.zeros(zero_padding, dtype=qtype), X))
	X = X.reshape(-1, 8)
	X = bitops.transpose(X, bits, qtype)
	permutes.append(p)
	
	for i in range(iterations):
		X, p = bitops.sort(X, reverse)
		X = X.reshape(-1, 8)
		X = bitops.transpose(X)
		permutes.append(p)
	return X, offset, scale, permutes, zero_padding


def decode(X, bits_per_dim, permutes, zero_padding,
	qtype=object,
	offset=0.0,
	scale=1.0,
	reverse=False,
	cumulative=False,
	**kwargs
	):
	"""
	"""
	bits = sum(bits_per_dim)
	for p in permutes[-1:0:-1]:
		X = bitops.transpose(X, bits, qtype)
		X = bitops.permute(X, p)
	p = permutes[0]
	X = bitops.transpose(X, bits, qtype)
	if cumulative:
		X = np.cumsum(X)
	X = bitops.permute(X, p)
	X = bitops.deserialize(X, bits_per_dim, qtype)
	X = bitops.realization(X, offset, scale)
	return X[zero_padding:]
