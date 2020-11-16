#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np

## Local
from .bitops import BitBuffer
from .utils import Prototype, log

	
def encode(X,
	dims=[],
	tree_depth=None,
	output=None,
	breadth_first=False,
	payload=False,
	pattern=0,
	**kwargs
	):
	"""
	"""
	assert(X.ndim == 1)
	tree_depth = tree_depth if tree_depth else np.iinfo(X.dtype).bits
	flags = BitBuffer(output + '.flg.bin', 'wb')
	stack_size = 0
	local = Prototype(points = 0)
	msg = "Layer: {:>2}, Flag: {:>16}, Stack: {:>8}, Points: {:>8}"
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin', 'wb') if output else BitBuffer()
	elif not isinstance(payload, BitBuffer):
		payload = False

	def expand(X, layer, tail, t=0):
		dim = dims[layer] if layer < len(dims) else dims[-1]
		mask = (1<<dim)-1
		fbit = 1<<dim
		flag = 0
		
		if len(X) == 0:
			pass
		elif dim == 0:
			fbit = len(X).bit_length()
			if tail > 1:
				m = X & 1 == t & 1
				flag = np.sum(m)
				if len(X) != flag:
					yield expand(X[~m]>>1, layer+1, max(tail-1, 0), t>>1)
				if flag:
					yield expand(X[m]>>1, layer+1, max(tail-1, 0), t>>1)
			else:
				flag = np.sum((X & 1).astype(bool))
				local.points += len(X)
		elif payload and len(X) == 1:
			payload.write(int(X), tail, soft_flush=True)
			local.points += 1
		else:
			for t in range(fbit):
				m = (X & mask) == t
				if np.any(m):
					flag |= 1<<t
					if tail > dim:
						yield expand(X[m]>>dim, layer+1, max(tail - dim, 0))
					else:
						local.points += 1
		
		flags.write(flag, fbit, soft_flush=True)
		if log.verbose:
			flag = hex(flag)[2:] if dim else flag
			log(msg.format(layer, flag, stack_size, local.points), end='\r', flush=True)
		pass
	
	nodes = deque(expand(X, 0, tree_depth, pattern))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
		stack_size = len(nodes)
	
	flags.close()
	if payload:
		payload.close()
	return flags, payload


def decode(Y, num_points,
	dims=[],
	tree_depth=None,
	payload=None,
	breadth_first=False,
	qtype=np.uint64,
	pattern=0,
	**kwargs
	):
	"""
	"""
	if isinstance(payload, str):
		payload = BitBuffer(payload, 'rb')
	elif isinstance(payload, BitBuffer):
		payload.open(payload.name, 'rb')
	else:
		payload = None

	tree_depth = tree_depth if tree_depth else np.iinfo(qtype).bits
	X = np.zeros(num_points, dtype=qtype)
	local = Prototype(points = 0)
	msg = "Layer: {:>2}, Flag: {:>16}, Points: {:>8}, Done: {:>3.2f}%"
	
	def expand(x, layer, pos, n=0):
		tail = max(tree_depth - pos, 0)
		dim = dims[layer] if layer < len(dims) else dims[-1]
		fbit = 1<<dim if dim else n.bit_length()
		flag = Y.read(fbit)
		
		if dim == 0:
			right = n - flag
			t = pattern & 1<<pos
			if tail > 1:
				if right > 0:
					yield expand(x | (t^1<<pos), layer+1, pos+1, right)
				if flag > 0:
					yield expand(x | t, layer+1, pos+1, flag)
			else:
				X[local.points] = x | t
				local.points += 1
		elif flag == 0:
			if payload:
				x |= payload.read(tail) << pos
			X[local.points] = x
			local.points += 1
		else:
			for t in range(fbit):
				if flag & 1<<t:
					if tail > dim:
						yield expand(x | t<<pos, layer+1, pos+dim)
					else:
						X[local.points] = x | t<<pos
						local.points += 1
			pass
		
		if log.verbose:
			progress = 100.0 * Y.tell() / len(Y)
			flag = hex(flag)[2:] if dim else flag
			log(msg.format(layer, flag, local.points, progress), end='\r', flush=True)
		pass
	
	nodes = deque(expand(np.zeros(1, dtype=qtype), 0, 0, num_points))
	while nodes:
		nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
	return X
