
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
				raise ValueError("Tree dimension of '0' or lower cannot be followed after higher dimensions!")
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


def minor_major(flags, total):
	minor = flags.read(total.bit_length())
	return minor, total - minor


def overflow(flags, total):
	bits = total.bit_length() - 1
	overflow = (1 << bits) - 1
	minor = flags.read(bits)
	if minor < overflow:
		return minor, total - minor
	
	major = flags.read(bits)
	if major < overflow:
		return total - major, major
	elif total & 1:
		odd = flags.read(1)
		return overflow + odd, overflow + (odd^1)
	else:
		return overflow, overflow


def permutation(X, bits):
	if bits is None:
		bits = np.iinfo(X.dtype).bits
	idx = np.zeros_like(X, int)
	p = bitops.argmax(X, bits, True, idx, [len(X)])
	yield p + 1, 1
	bits -= 1

	for bits in range(bits, 0, -1):
		flags = X >> p[idx] & 1
		mask = (1<<p[idx]) - 1
		X = (X>>1 & ~mask) | (X & mask)
		nodes = np.vstack((idx, flags.astype(idx.dtype))).T
		args = np.argsort(nodes, axis=0)
		nodes[args]
		X[args]
		ravel = np.ravel_multi_index(nodes.T, (len(p), 2))
		flags = np.zeros(len(p) * 2, X.dtype)
		idx, counts = np.unique(nodes, return_inverse=True, return_counts=True, axis=0)[1:]
		p = bitops.argmax(X, bits, True, idx, counts[...,None])
		np.add.at(flags, ravel, 1)
		flags = flags > 0
		nodes = flags.astype(X.dtype)
		nodes[flags] *= p + 1
		yield nodes.flatten(), 1


def depermutation(bitstream, word_length, perm, dtype=np.uint64):
	bits = word_length.bit_length()
	X = np.zeros([1], dtype)
	perm = np.array(perm, dtype)[None,...]
	p = np.array([bitstream.read(bits) for i in range(len(X))], dtype)
	for word_length in range(word_length-1, 0, -1):
		bits = word_length.bit_length()
		nodes = np.array([bitstream.read(bits) for i in range(len(X)*2)], dtype)
		flags = nodes > 0

		p = np.repeat(p, 2)[flags] - 1
		perm = np.repeat(perm, 2, axis=0)[flags]

		i, x = np.where(nodes.reshape(-1,2))
		x = x.astype(X.dtype)
		x <<= perm[range(len(perm)),p]
		X = x + X[i]

		mask = p[...,None] != range(word_length+1)
		perm = perm[mask].reshape(-1, word_length)
		p = nodes[flags]
	return X


def encode(X, dims, word_length,
	differential=False,
	tree=None,
	payload=False,
	pattern=0,
	yielding=False,
	):
	"""
	"""
	tree = tree if isinstance(tree, BitBuffer) else BitBuffer(tree, 'wb')
	payload = payload and (payload if isinstance(payload, BitBuffer) else BitBuffer(payload, 'wb'))
	differential = differential and dict()

	if -2 in dims:
		for flags, bits in permutation(X, word_length):
			for flag in flags:
				tree.write(flag, bits, soft_flush=True)
				#bitops.transpose(flags, bits, buffer=tree)
			if yielding:
				yield tree, payload
	else:
		dim_seq = [dim for dim in yield_dims(dims, word_length)]
		layers = bitops.tokenize(X, dim_seq)
		mask = np.ones(len(X), bool)
		tail = np.full(len(X), word_length)

		for i, (X0, X1, dim) in enumerate(zip(layers[:-1], layers[1:], dim_seq)):
			uids, idx, counts = np.unique(X0[mask], return_inverse=True, return_counts=True)
			flags, hist = bitops.encode(X1[mask], idx, max(dim,1))

			if differential is not False:
				diff = differential.get(i, None)
				if diff:
					flags = [flag ^ diff.get(uid, 0) for uid, flag in zip(uids, flags)]
					del diff
				differential[i] = {uid:flag for uid, flag in zip(uids, flags)}
			
			if dim < 0:
				major = pattern >> (word_length - i - 1) & 1
				minor = major ^ 1
				bits = np.array([int(c).bit_length() - 1 for c in counts], np.uint64)
				overflow = (1 << bits) - 1
				minor, major = hist[:,minor].astype(bits.dtype), hist[:,major].astype(bits.dtype)
				minor_overflow = minor >= overflow
				odd_overflow = minor_overflow & (major >= overflow) & (counts & 1 > 0)

				symbols = minor.copy()
				symbols[minor_overflow] = overflow[minor_overflow] << bits[minor_overflow] | np.minimum(major[minor_overflow], overflow[minor_overflow])
				symbols[odd_overflow] <<= 1
				symbols[odd_overflow] |= minor[odd_overflow] > major[odd_overflow]

				bits[minor_overflow] *= 2
				bits[odd_overflow] += 1
				for s, b in zip(symbols, bits):
					tree.write(s, b, soft_flush=True)
			elif dim:
				for flag in flags:
					tree.write(flag, 1<<dim, soft_flush=True)
			else:
				minor = pattern >> (word_length - i - 1) & 1 ^ 1
				minor = hist[:,minor]	
				bits = np.array([int(c).bit_length() for c in counts])
				for minor, bits in zip(minor, bits):
					tree.write(minor, bits, soft_flush=True)
			
			if payload:
				mask[mask] = (counts > 1)[idx]
				tail[mask] -= dim
				if not np.any(mask):
					break

			if yielding:
				yield tree, payload

	if payload:
		for x, bits in zip(X, tail):
			payload.write(x, bits, soft_flush=True)
	if not yielding:
		return tree, payload


def decode(header_file, payload=True):
	"""
	"""
	header = load_header(header_file)
	flags = BitBuffer(path.join(path.dirname(header_file), header.flags), 'rb')
	payload = header.payload and BitBuffer(path.join(path.dirname(header_file), header.payload), 'rb')
	word_length = sum(header.bits_per_dim)

	if -2 in header.dims:
		X = depermutation(flags, word_length, np.arange(word_length), header.qtype)
	else:
		dim_seq = yield_dims(header.dims, word_length)
		X = np.zeros([1], dtype=header.qtype)
		tails = np.full(1, word_length) if payload else None
		counts = [header.inp_points]
		for i, dim in enumerate(dim_seq):
			if dim < 0:
				nodes = np.array([overflow(flags, int(c)) for c in counts], dtype=header.qtype)
				if not header.pattern >> (word_length - i - 1) & 1:
					nodes = nodes[:,::-1]
			elif dim:
				nodes = np.array([flags.read(1<<dim) for i in range(len(X))], dtype=header.qtype)
			else:
				nodes = np.array([minor_major(flags, int(c)) for c in counts], dtype=header.qtype)
				if not header.pattern >> (word_length - i - 1) & 1:
					nodes = nodes[:,::-1]
			X, counts, tails = bitops.decode(nodes, dim, X, tails)
	
	if payload:
		payload = [payload.read(bits) for bits in tails]
		X = X << tails | payload
	
	if header.permute is True:
		X = bitops.reverse(X, word_length)
	elif header.permute:
		X = bitops.permute(X, header.permute)
	
	X = bitops.realize(X, header.bits_per_dim, header.offset, header.scale, header.xtype)
	return X, header