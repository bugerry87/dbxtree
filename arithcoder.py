
## Build In
from collections import deque

## Installed
import numpy as np

## Local
import mhdm.bitops as bitops


def get_range_index(X):
	"""
	"""
	symbols, args = np.unique(X, return_counts=True)
	args = np.argsort(args)
	symbols = symbols[args]
	n = len(symbols)
	offset = 0.5 * (n%2) / n
	ranges = [1.0]
	codec = [1]
	
	def expand(pos, scalar, code):
		ranges.append(pos)
		codec.append(code)
		yield expand(pos - scalar, scalar/2, code<<1 | 1)
		yield expand(pos + scalar, scalar/2, code<<1)
	
	nodes = deque(expand(0.5, 0.25, 0b1))
	while len(ranges) < n:
		node = nodes.popleft()
		nodes.extend(node)
	ranges = np.array(ranges)[:n]
	ranges -= offset
	args = np.argsort(ranges)
	index = dict(zip(symbols[args], ranges[args]))
	
	return index 


def encode(X, output=None):
	if not isinstance(output, bitops.BitBuffer):
		output = bitops.BitBuffer(output, 'wb')

	
	
	
	pass
