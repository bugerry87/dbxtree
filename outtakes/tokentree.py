
## Build in
from collections import deque

## Installed
import numpy as np

## Local
from .bitops import BitBuffer
from .utils import Prototype, log


def encode(X, output=None, breadth_first=False, payload=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits
	fbits = 1<<token_dim
	token = np.arange(1<<token_dim, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_dim:]
	flags = BitBuffer(output + '.flg.bin', 'wb')
	msg = "Layer: {:>2}, {:0>" + str(fbits) + "}, StackSize: {:>10}"
	stack_size = 0
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin', 'wb') if output else BitBuffer()

	def expand(X, bits):
		flag = 0
		
		if len(X) == 0 or bits == 0:
			pass
		elif payload is not False and len(X) == 1:
			for d in range(token_dim):
				payload.write(int(X[:,d]), bits)
			payload.flush()
		else:
			for i, t in enumerate(token):
				m = np.all(X & 1 == t, axis=-1)
				if np.any(m):
					if bits > 1:
						yield expand(X[m] >> 1, bits-1)
					flag |= 1<<i
		if log.verbose:
			log(msg.format(tree_depth-bits, bin(flag)[2:], stack_size), end='\r', flush=True)
		flags.write(flag, fbits, soft_flush=True)
		pass
	
	nodes = deque(expand(X, tree_depth))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
		stack_size = len(nodes)
	
	flags.close()
	if payload:
		payload.close()
	return flags, payload


def decode(Y, payload=None, qtype=np.uint16, breadth_first=False, **kwargs):
	if isinstance(payload, str):
		payload = BitBuffer(payload, 'rb')
	elif isinstance(payload, BitBuffer):
		payload.open(payload.name, 'rb')
	else:
		payload = None

	qtype = np.iinfo(qtype)
	fbits = np.iinfo(Y.dtype).bits
	dim = int(np.log2(fbits))
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(qtype)
	msg = "Layer: {:0>2}, {:0>" + str(fbits) + "}, Points: {:>10}, Done: {:>3.2f}%"
	local = Prototype(
		X = np.zeros((np.sum(Y==0), dim), dtype=qtype),
		points = 0,
		done = 0
		)
	Xi = iter(range(len(local.X)))
	ptr = iter(Y)
	
	def expand(layer, x):
		flag = next(ptr, 0) if layer < qtype.bits else 0
		
		if flag == 0:
			if payload:
				x |= payload.read(qtype.bits - layer) << layer
			
			xi = next(Xi, None)
			if xi is not None:
				local.X[xi] = x
			else:
				local.X = np.vstack((local.X, x))
			local.points += 1
			
		else:
			for bit in range(fbits):
				if flag & 1<<bit == 0:
					continue
				yield expand(layer+1, x | token[bit]<<layer)
		
		if log.verbose:
			local.done += 1
			progress = 100.0 * local.done / len(Y)
			log(msg.format(layer, bin(flag)[2:], points, progress), end='\r', flush=True)
		pass
		
	nodes = deque(expand(0, np.zeros(dim, dtype=qtype)))
	while nodes:
		nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
	return X
	