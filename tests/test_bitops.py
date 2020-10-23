
import numpy as np
import mhdm.bitops as bitops


def test_serialization():
	bits_per_dim = [16,16,16]
	X = np.arange(30).reshape(-1,3)
	Y, offset, scale = bitops.serialize(X, bits_per_dim, qtype=np.uint64)
	Y = bitops.deserialize(Y, bits_per_dim, qtype=np.uint64)
	Y = bitops.realization(Y, offset, scale)
	Y = np.round(Y).astype(int)
	result = np.all(X == Y)
	assert(result)


def test_sort_n_permute():
	X = np.arange(256).astype(np.uint8)
	Y, p = bitops.sort(X)
	Y = bitops.permute(Y, p)
	result = np.all(X == Y)
	assert(result)
	Y, p = bitops.sort(X, reverse=True)
	Y = bitops.permute(Y, p)
	result = np.all(X == Y)
	assert(result)
	Y, p = bitops.sort(X, reverse=True, absp=True)
	Y = bitops.permute(Y, p)
	result = np.all(X == Y)
	assert(result)
	Y, p = bitops.sort(X, reverse=False, absp=True)
	Y = bitops.permute(Y, p)
	result = np.all(X == Y)
	assert(result)


def test_transpose():
	X = np.arange(8000, dtype=np.uint64).reshape(-1,8)
	Y = bitops.transpose(X)
	assert(Y.shape == (1000*64,))
	Y = bitops.transpose(Y, dtype=np.uint64)
	assert(X.shape == Y.shape)
	result = np.all(X == Y)
	assert(result)
	assert(Y.dtype == np.uint64)
	