
import numpy as np
import mhdm.bitops as bitops

def test_serialization():
	bits_per_dim = [16,16,16]
	X = np.arange(30).reshape(-1,3)
	S, offset, scale = bitops.serialize(X, bits_per_dim, qtype=np.uint64)
	D = bitops.deserialize(S, bits_per_dim, qtype=np.uint64)
	R = bitops.realization(D, offset, scale)
	Y = np.round(R).astype(int)
	result = np.all(X == Y)
	assert(result)
