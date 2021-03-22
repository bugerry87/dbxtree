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
	output=None,
	flags=True,
	payload=False,
	tree_depth=None,
	breadth_first=False,
	callback=None,
	**kwargs
	):
	"""
	"""
	assert(X.ndim == 1)
	tree_depth = tree_depth or np.iinfo(X.dtype).bits
	stack_size = 0
	local = Prototype(points=0, overflows=0)
	msg = "\rLayer: {:>2}, Flag: {:>16}, Stack: {:>8}, Points: {:>8}, Overflow Bits: {:>8}"
	
	if flags is True:
		flags = BitBuffer(output + '.flg.bin', 'wb') if output else BitBuffer()
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin', 'wb') if output else BitBuffer()

	def expand(X, layer, tail):
		assert(len(X))

		if dims:
			dim = dims[layer] if layer < len(dims) else dims[-1]
		else:
			dim = 0
		
		if dim > 0:
			fbit = 1<<dim
		else:
			fbit = len(X).bit_length()
		flag = 0

		if dim == -1:
			fbit = max(fbit-1, 1)
			m = (X & 1).astype(bool)
			right = np.sum(m)
			left = len(X) - right
			mask = (1<<fbit)-1

			if len(X) == 1 or right < mask:
				minor = right
				flag = minor
			elif right == left:
				minor = right
				flag = mask<<fbit | minor
				fbit *= 2
				local.overflows += fbit
			elif left < mask:
				minor = left
				m[:] = ~m
				flag = mask<<fbit | minor
				fbit *= 2
				local.overflows += fbit
			elif left < right:
				minor = left
				m[:] = ~m
				flag = mask<<fbit+1 | mask<<1 | 0
				fbit = fbit*2 + 1
				local.overflows += fbit + 1
			else:
				minor = right
				flag = mask<<fbit+1 | mask<<1 | 1
				fbit = fbit*2 + 1
				local.overflows += fbit + 1

			if tail > 1:
				if minor < len(X):
					yield expand(X[~m]>>1, layer+1, max(tail-1, 1))
				if minor:
					yield expand(X[m]>>1, layer+1, max(tail-1, 1))
			else:
				local.points += 1
		elif dim == 0:
			m = (X & 1).astype(bool)
			flag = right = np.sum(m)
			left = len(X) - right
			
			if tail > 1:
				if left:
					yield expand(X[~m]>>1, layer+1, max(tail-1, 1))
				if right:
					yield expand(X[m]>>1, layer+1, max(tail-1, 1))
			else:
				local.points += len(X)
			
			if payload:
				payload.write(flag>>1, max(fbit-1, 1), soft_flush=True)
				flag &= 1
				fbit = 1
		elif payload and len(X) == 1:
			payload.write(int(X), tail, soft_flush=True)
			local.points += 1
		else:
			mask = (1<<dim)-1
			for t in range(fbit):
				m = (X & mask) == t
				if np.any(m):
					flag |= 1<<t
					if tail > dim:
						yield expand(X[m]>>dim, layer+1, max(tail - dim, 1))
					else:
						local.points += 1
		
		if flags:
			flags.write(flag, fbit, soft_flush=True)
		if callback:
			callback.update(flag, fbit)
		if log.verbose:
			flag = hex(flag)[2:] if dim else flag
			log(msg.format(layer, flag, stack_size, local.points, local.overflows), end='', flush=True)
		pass
	
	nodes = deque(expand(X, 0, tree_depth))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
		stack_size = len(nodes)
	
	if flags:
		flags.close()
	if payload:
		payload.close()
	return flags, payload, kwargs


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

	tree_depth = tree_depth or np.iinfo(qtype).bits
	X = np.zeros(num_points, dtype=qtype)
	local = Prototype(points=0, overflows=0)
	msg = "\rLayer: {:>2}, Flag: {:>16}, Points: {:>8}, Overflows: {:>8}, Done: {:>6.2f}%"
	
	def expand(x, layer, pos, remains=0):
		tail = max(tree_depth - pos, 0)
		dim = dims[layer] if layer < len(dims) else dims[-1]
		if dim > 0:
			fbit = 1<<dim
		elif dim == 0:
			fbit = remains.bit_length()
		else:
			fbit = max(remains.bit_length() - 1, 1)
		flag = Y.read(fbit)
		
		if dim == -1:
			assert(remains)
			mask = (1<<fbit) - 1
			if remains != 1 and flag == mask:
				flag = Y.read(fbit)
				if flag*2 == remains:
					minor = 1
				elif flag == mask:
					minor = Y.read(1)
					local.overflows += 1
				else:
					minor = 0
				local.overflows += fbit*2
			else:
				minor = 1

			if tail > 1:
				if flag < remains:
					yield expand(x | (1^minor)<<pos, layer+1, pos+1, remains - flag)
				if flag:
					yield expand(x | minor<<pos, layer+1, pos+1, flag)
			else:
				X[local.points] = x | bool(flag)<<pos
				local.points += 1
		elif dim == 0:
			right = flag
			left = remains - right
			if tail > 1:
				if left > 0:
					yield expand(x.copy(), layer+1, pos+1, left)
				if right > 0:
					yield expand(x | 1<<pos, layer+1, pos+1, right)
			else:
				X[local.points] = x | bool(right)<<pos
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
			progress = 800.0 * Y.tell() / len(Y)
			flag = hex(flag)[2:] if dim else flag
			log(msg.format(layer, flag, local.points, local.overflows, progress), end='', flush=True)
		pass
	
	nodes = deque(expand(np.zeros(1, dtype=qtype), 0, 0, num_points))
	while nodes:
		nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
	return X
