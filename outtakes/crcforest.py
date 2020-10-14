#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np
from crcmod import mkCrcFun

## Local
from mhdm.utils import BitBuffer, log


CRC32 = mkCrcFun(0x104c11db7, initCrc=0, xorOut=0xFFFFFFFF)


def find_ftype(dim):
	if dim <= 3:
		dtype = np.uint8
	elif dim == 4:
		dtype = np.uint16
	elif dim == 5:
		dtype = np.uint32
	else:
		raise ValueError("Data of only up to 5 dimensions are supported")
	return dtype


def reverse_bits(X):
	if X.dtype == np.uint64:
		Y = X.astype(object)
	else:
		Y = X.copy()
	
	if X.dtype == object:
		shifts = 64
	else:
		shifts = np.iinfo(X.dtype).bits
	
	for low, high in zip(range(shifts//2), range(shifts-1, shifts//2 - 1, -1)):
		Y |= (X & 1<<low) << high | (X & 1<<high) >> high-low
	return Y


def create_checksum(X, seed_length=8, **kwargs):
	xtype = X.dtype
	bits = 64 if xtype == object else np.iinfo(xtype).bits
	xor = X.sum(axis=-1) & ((1<<bits)-1)
	cs = np.array([CRC32(x) for x in X], dtype=xtype) & ((1<<seed_length)-1)
	
	for low, high in zip(range(bits-seed_length), range(seed_length, bits)):
		cs |= (cs>>low & 1)<<high ^ (xor & 1<<high)
	return np.vstack((X.T, cs)).astype(xtype).T


def check_checksum(X, at, seed_length=8, **kwargs):
	xtype = X.dtype
	bits = 64 if xtype == object else np.iinfo(xtype).bits
	
	if at >= bits:
		cs = np.array([CRC32(x) for x in X[:,:-1]], dtype=xtype) & ((1<<seed_length)-1)
		m = X[:,-1] & ((1<<seed_length)-1) == cs
	else:
		xor = X[:,:-1].sum(axis=-1) & ((1<<at)-1)
		cs = X[:,-1] & ((1<<min(at, seed_length))-1)
		
		for low, high in zip(range(at-seed_length), range(seed_length, at)):
			cs |= (cs>>low & 1)<<high ^ (xor & 1<<high)
		m = X[:,-1] & ((1<<at)-1) == cs
	return m


def decode(Y, output=None, dim=2, dtype=np.uint32, breadth_first=False, seed_length=8, **kwargs):
	dim += 1
	dtype = np.iinfo(dtype)
	fbits = 1 << dim
	depth = dtype.bits
	xtype = object if depth > 32 else dtype
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(xtype)
	subtrees = {0}
	ptr = iter(Y)
	F = {}
	X = []
	
	def expand(sub, layer, x):
		flag = next(ptr, 0) if layer < depth else 0
		if flag == 0:
			x = np.hstack((x, np.full((len(x),1), layer, dtype=x.dtype)))
			subtrees.add(layer)
			
			if sub in F:
				F[sub].append(x)
			else:
				F[sub] = [x]
			
			if log.verbose:
				points[:] += len(x)
		else:
			for bit in range(fbits):
				if flag & 1<<bit == 0:
					continue
				yield expand(sub, layer+1, x | token[bit]<<layer)
		
		if log.verbose:
			done[:] += 1
			progress = 100.0 * float(done) / len(Y)
			log(msg.format(sub, layer, bin(flag)[2:], int(points),  progress), end='\r', flush=True)
		pass
	
	def merge(x, sub):
		if sub == depth:
			m = check_checksum(x, sub, seed_length)
			X.append(x[m])
			return
		
		if sub > seed_length:
			m = check_checksum(x, sub, seed_length)
			if log.verbose:
				points[:] += m.sum() - np.sum(~m)
			
			if np.any(m):
				x = x[m]
			else:
				return
		else:
			points[:] += len(x)
	
		f, layers = F[sub][:,:-1], F[sub][:,-1]
		for tx in x:
			tx = tx.reshape(1,-1) | f
			for layer in np.unique(layers):
				yield merge(tx[layer==layers], layer)
	
		if log.verbose:
			done [:] += 1
			progress = 100.0 * float(done) / float(total)
			merges = np.sum([len(v) for v in X])
			log(msg.format(sub, int(points), int(merges), progress))#, end='\r', flush=True)
		pass
	
	log("\nUnpack:")
	if log.verbose:
		msg = "SubTree:{:0>2}, Layer:{:0>2}, {:0>" + str(fbits) + "}, Points:{:>8}, Done:{:>3.2f}%"
		done = np.zeros(1)
		points = np.zeros(1)
	
	for sub in range(depth):
		if sub not in subtrees:
			continue
		
		nodes = deque(expand(sub, sub, np.zeros((1,dim), dtype=xtype)))
		while nodes:
			nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
		F[sub] = np.vstack(F[sub])
	
	log("\nMerge:")
	if log.verbose:
		msg = "SubTree:{:0>2}, Points:{:>10}, Merges:{:>10}, Done:{:>3.2f}%"
		done[:] = 0
		points[:] = 0
		total = np.sum([len(v) for v in F.values()])
	
	nodes = deque(merge(np.zeros((1,dim), dtype=xtype), 0))
	while nodes:
		nodes.extend(nodes.pop())
	
	return np.vstack(X)	

	
def encode(X, output=None, breadth_first=False, **kwargs):
	dim = X.shape[-1]
	fbits = 1 << dim
	depth = 64 if X.dtype == object else np.iinfo(X.dtype).bits
	shifts = np.zeros(len(X), dtype=np.int8)
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(X.dtype)
	flags = BitBuffer(output)
	
	if log.verbose:
		stack_size = 0
		msg = "SubTree:{:>2}, Flag:{:0>" + str(fbits) + "}, Node:{:>8}, Stack:{:>8}, Done:{:>3.2f}%"
		done = len(X) * (depth-1)

	def expand(Xi, leaf_size):
		flag = 0
		
		if len(Xi) > leaf_size:
			x = X[Xi] >> shifts[Xi].reshape(-1, 1)
			for i, t in enumerate(token):
				m = np.all(x & 1 == t, axis=-1)
				if not np.any(m):
					continue
				flag |= 1<<i
					
				m &= shifts[Xi] != (depth-1)
				if not np.any(m):
					continue
					
				xi = Xi[m]
				shifts[xi] += 1
				yield expand(xi, leaf_size)
				
		if log.verbose:
			progress = np.sum(shifts, dtype=float) / done * 100
			log(msg.format(layer, bin(flag)[2:], len(Xi), stack_size, progress), end='\r', flush=True)
		flags.write(flag, fbits, soft_flush=True)
		pass
	
	for layer in range(depth):
		m = shifts == layer
		if not np.any(m):
			continue
		
		leaf_size = depth - layer
		if m.sum() <= leaf_size:
			nodes = deque(expand(np.arange(len(X))[m], 1))
		else:
			nodes = deque(expand(np.arange(len(X))[m], leaf_size))
				
		while nodes:
			nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
			stack_size = len(nodes)
	
	flags.close()
	return flags


if __name__ == '__main__':
	from argparse import ArgumentParser
	
	def init_argparse(parents=[]):
		''' init_argparse(parents=[]) -> parser
		Initialize an ArgumentParser for this module.
		
		Args:
			parents: A list of ArgumentParsers of other scripts, if there are any.
			
		Returns:
			parser: The ArgumentParsers.
		'''
		parser = ArgumentParser(
			description="Demo of TokenTree",
			parents=parents
			)
		
		parser.add_argument(
			'--compress', '-X',
			metavar='PATH'
			)
		
		parser.add_argument(
			'--decompress', '-Y',
			metavar='PATH'
			)
		
		parser.add_argument(
			'--dtype', '-t',
			metavar='TYPE',
			default='uint32'
			)
		
		parser.add_argument(
			'--dim', '-d',
			type=int,
			metavar='INT',
			default=2
			)
		
		parser.add_argument(
			'--leaf_factor', '-l',
			type=int,
			metavar='INT',
			default=1
			)
		
		parser.add_argument(
			'--seed_length', '-s',
			type=int,
			metavar='INT',
			default=8
			)
		
		parser.add_argument(
			'--output', '-o',
			metavar='PATH',
			default='crcforest.bin'
			)
		
		parser.add_argument(
			'--breadth_first', '-b',
			metavar='FLAG',
			nargs='?',
			type=bool,
			default=False,
			const=True
			)
		
		parser.add_argument(
			'--reverse', '-r',
			metavar='FLAG',
			nargs='?',
			type=bool,
			default=False,
			const=True
			)
		
		parser.add_argument(
			'--visualize', '-V',
			metavar='FLAG',
			nargs='?',
			type=bool,
			default=False,
			const=True
			)
		
		parser.add_argument(
			'--verbose', '-v',
			metavar='FLAG',
			nargs='?',
			type=bool,
			default=False,
			const=True
			)
		
		return parser
	
	args, _ = init_argparse().parse_known_args()
	log.verbose = args.verbose
	
	if args.compress:
		log("\n---CRCForest Encoding---\n")
		X = np.fromfile(args.compress, dtype=args.dtype)
		X = X.reshape(-1, args.dim)
		if args.reverse:
			X = reverse_bits(X)
		X = create_checksum(X, **args.__dict__)
		
		u = np.unique(X[:,-1])
		log("Checksum uniqueness {:>3.2f}%".format(100.0 * len(u)/len(X)))
		log("Data:", X.shape, "\n", X)
		
		Y = encode(X, **args.__dict__)
		log("\nFlags safed to:", Y.fid.name)
	elif args.decompress:
		log("\n---CRCForest Decoding---\n")
		Y = np.fromfile(args.decompress, dtype=find_ftype(args.dim+1))
		log("Flags:", Y.shape, "\n", Y, "\n")
		Xs, Xl = decode(Y, **args.__dict__)
		
		if args.reverse:
			X = reverse_bits(X).astype(args.dtype)
		log("\n Final Data:", X.shape, "\n", X, "\n")
	else:
		raise ValueError("Choose a file to either compress (-X) or decompress (-Y)!")

