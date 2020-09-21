#!/usr/bin/env python3

## Build in
from collections import deque
from copy import copy

## Installed
import numpy as np

## Local
from mhdm.utils import BitBuffer, log


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
	seed = np.round(np.random.rand(len(xor)) * ((1<<seed_length)-1)).astype(xtype)
	
	for low, high in zip(range(bits-seed_length), range(seed_length, bits)):
		seed |= (seed>>low & 1)<<high ^ (xor & 1<<high)
	return np.vstack((X.T, seed)).astype(xtype).T


def check_checksum(X, at, seed_length=8, **kwargs):
	xtype = X.dtype
	bits = 64 if xtype == object else np.iinfo(xtype).bits
	xor = X[:,:-1].sum(axis=-1) & ((1<<bits)-1)
	seed = X[:,-1]
	high = at
	low = at - seed_length
	c = (seed>>low & 1)<<at ^ (xor & 1<<high)
	return c > 0


def decode(Y, output=None, dim=2, dtype=np.uint32, breadth_first=False, seed_length=8, **kwargs):
	dim += 1
	dtype = np.iinfo(dtype)
	fbits = 1 << dim
	depth = dtype.bits
	xtype = object if depth > 32 else dtype
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(xtype)
	X = {0:np.zeros((1,dim), dtype=xtype)}
	ptr = iter(Y)
	
	if log.verbose:
		msg = "SubTree{:0>2}, Layer{:0>2}, {:0>" + str(fbits) + "}, Pnts:{:>10}, Rej:{:>10}, {:>3.2f}%"
		done = np.zeros(1)
		points = np.zeros(1)
		rejected = np.zeros(1)
	
	def expand(layer, x):
		flag = next(ptr, 0)
		if flag == 0:
			if layer in X:
				X[layer] = np.vstack((X[layer], x))
			else:
				X[layer] = x
			if log.verbose:
				points[:] = np.sum([len(v) for v in X.values()])
		else:
			for bit in range(fbits):
				if flag & 1<<bit == 0:
					continue
					
				tx = x | token[bit]<<layer
				if sub and layer >= seed_length:
					m = check_checksum(tx, layer, seed_length).flatten()
					if log.verbose:
						rejected[:] += np.sum(~m)
					if not np.any(m):
						raise RuntimeError("Invalid parser state!")
					elif layer == depth-1:
						X[layer] = tx[m]
					else:
						yield expand(layer+1, tx[m])
				else:
					yield expand(layer+1, tx)
		
		if log.verbose:
			done[:] += 1
			progress = 100.0 * float(done) / len(Y)
			log(msg.format(sub, layer, bin(flag)[2:], int(points), int(rejected),  progress)) #, end='\r', flush=True)
		pass
	
	for sub in range(depth):
		if not sub in X:
			continue
		nodes = deque(expand(sub, X[sub]))
		while nodes:
			nodes.extend(nodes.popleft() if breadth_first else nodes.pop())
			stack_size = len(nodes)
		if not sub == depth-1:
			del X[sub]
	
	return X[sub].astype(dtype)

	
def encode(X, output=None, leaf_size=0, breadth_first=False, **kwargs):
	dim = X.shape[-1]
	fbits = 1 << dim
	depth = 64 if X.dtype == object else np.iinfo(X.dtype).bits
	leaf_size = dim if leaf_size <= 0 else leaf_size
	shifts = np.zeros(len(X), dtype=np.int8)
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(X.dtype)
	flags = BitBuffer(output)
	
	stack_size = 0
	msg = "SubTree: {:>2}, Flag: {:0>" + str(fbits) + "}, StackSize: {:>10}, Done: {:>3.2f}%"
	done = len(X) * (depth-1)

	def expand(Xi):
		flag = 0
		
		if len(Xi) > leaf_size:
			x = X[Xi] >> shifts[Xi].reshape(-1, 1)
			for i, t in enumerate(token):
				m = np.all(x & 1 == t, axis=-1)
				if not np.any(m):
					continue
					
				m &= shifts[Xi] != (depth-1)
				if not np.any(m):
					continue
					
				xi = Xi[m]
				shifts[xi] += 1
				yield expand(xi)
				flag |= 1<<i
				
		if log.verbose:
			progress = np.sum(shifts, dtype=float) / done * 100
			log(msg.format(layer, bin(flag)[2:], stack_size, progress), end='\r', flush=True)
		flags.write(flag, fbits, soft_flush=True)
		pass
	
	for layer in range(depth):
		m = shifts == layer
		if not np.any(m):
			continue
	
		nodes = deque(expand(np.arange(len(X))[m]))
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
			'--leaf_size', '-l',
			type=int,
			metavar='INT',
			default=0
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
		log("Data:", X.shape, "\n", X, "\n")
		
		Y = encode(X, **args.__dict__)
		log("\nFlags safed to:", Y.fid.name)
	elif args.decompress:
		log("\n---CRCForest Decoding---\n")
		Y = np.fromfile(args.decompress, dtype=find_ftype(args.dim))
		log("Flags:", Y.shape, "\n", Y, "\n")
		X = decode(Y, **args.__dict__)
		if args.reverse:
			X = reverse_bits(X).astype(args.dtype)
		log("\n Final Data:", X.shape, "\n", X, "\n")
	else:
		raise ValueError("Choose a file to either compress (-X) or decompress (-Y)!")

