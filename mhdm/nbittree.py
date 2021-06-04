
## Build in
import pickle
import os.path as path

## Installed
import numpy as np

## Local
from . import bitops
from .utils import Prototype
from .bitops import BitBuffer


def save_header(header_file, **kwargs):
	with open(header_file, 'wb') as fid:
		pickle.dump(kwargs, fid)
	return header_file, Prototype(**kwargs)


def load_header(header_file):
	with open(header_file, 'rb') as fid:
		header = pickle.load(fid)
	return Prototype(**header)


def yield_dims(dims, word_length):
	if len(dims):
		prev = 0
		for dim in dims:
			if dim > 6:
				raise ValueError("Tree dimension greater than 6 is not allowed!")
			if prev > 0 and dim <= 0:
				raise ValueError("Tree dimension of '0' cannot be followed after higher dimensions!")
			prev = dim
			if word_length > 0:
				word_length -= max(dim, 1)
				yield dim
		while word_length > 0:
			word_length -= max(dim, 1)
			yield dim
	else:
		while word_length > 0:
			word_length -= 1
			yield 0


def decode(header_file, payload=True):
	"""
	"""
	header = load_header(header_file)
	flags = BitBuffer(path.join(path.dirname(header_file), header.flags), 'rb')
	payload = header.payload and BitBuffer(path.join(path.dirname(header_file), header.payload), 'rb')
	word_length = sum(header.bits_per_dim)
	dim_seq = yield_dims(header.dims, word_length)

	X = np.zeros([1], dtype=header.qtype)
	tails = np.full(1, word_length) if payload else None
	for dim in dim_seq:
		if dim:
			nodes = np.array([flags.read(1<<dim) for i in range(len(X))], dtype=header.qtype)
		else:
			counts = [header.inp_points]
			nodes = np.array([flags.read(int(c).bit_length()) for c in counts], dtype=header.qtype)
		X, counts, tails = bitops.decode(nodes, dim, X, tails)
	
	if payload:
		payload = [payload.read(bits) for bits in tails]
		X = X << tails | payload
	
	if header.permute is True:
		X = bitops.reverse(X, word_length)
	elif header.permute:
		X = bitops.permute(X, header.permute)
	
	X = bitops.deserialize(X, header.bits_per_dim, header.qtype)
	X = bitops.realize(X, header.bits_per_dim, header.offset, header.scale, header.xtype)
	return X, header