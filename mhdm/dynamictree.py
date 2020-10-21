#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np

## Local
from . import bitops
from .bitops import BitBuffer
from .utils import Prototype, log

	
def encode(X, dims=[], tree_depth=None, output=None, breadth_first=False, payload=False, **kwargs):
	"""
	"""
	assert(X.ndim == 1)
	tree_depth = tree_depth if tree_depth else np.iinfo(X.dtype).bits
	flags = BitBuffer(output + '.flg.bin', 'wb')
	stack_size = 0
	msg = "Layer: {:>2}, BranchFlag: {:>16}, StackSize: {:>10}"
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin', 'wb') if output else BitBuffer()

	def expand(X, layer, tail):
		flag = 0
		dim = dims[layer] if layer < len(dims) else 1
		fbit = 1<<dim
		mask = (1<<dim)-1
		
		if len(X) == 0 or dim == 0:
			pass
		elif payload is not False and len(X) == 1:
			payload.write(int(X), tail, soft_flush=True)
		else:
			for t in range(fbit):
				m = X & mask == t
				if np.any(m):
					yield expand(X[m]>>dim, layer+1, tail-dim)
					flag |= 1<<t
		if log.verbose:
			log(msg.format(layer, hex(flag)[2:], stack_size), end='\r', flush=True)
		flags.write(flag, fbit, soft_flush=True)
		pass
	
	nodes = deque(expand(X, 0, tree_depth))
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
	msg = "Layer: {:>2}, BranchFlag: {:>16}, Points: {:>10}, Done: {:>3.2f}%"
	X = np.zeros(num_points, dtype=qtype)
	Xi = iter(range(len(X)))
	local = Prototype(
		points = 0
		)
	
	def expand(x, layer, pos):
		dim = dims[layer] if layer < len(dims) else 1
		fbit = 1<<dim
		flag = Y.read(fbit) if layer < tree_depth else 0
		
		if flag == 0:
			if payload:
				x |= payload.read(tree_depth-pos) << pos
			
			xi = next(Xi, None)
			if xi is not None:
				X[xi] = x
			local.points = xi+1
		else:
			for token in range(fbit):
				if flag & 1<<token:
					yield expand(x | token<<pos, layer+1, pos+dim)
		
		if log.verbose:
			progress = 100.0 * local.points / len(X)
			log(msg.format(layer, hex(flag)[2:], local.points, progress), end='\r', flush=True)
		pass
		
	nodes = deque(expand(np.zeros(1, dtype=qtype), 0, 0))
	while nodes:
		nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
	return X