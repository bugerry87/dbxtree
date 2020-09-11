#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np

## Local
from mhdm.utils import BitBuffer, log

def find_ftype(token_dim):
	if token_dim <= 3:
		return np.iinfo(np.uint8)
	elif token_dim == 4:
		return np.iinfo(np.uint16)
	elif token_dim == 5:
		return np.iinfo(np.uint32)
	elif token_dim == 6:
		return np.iinfo(np.uint64)
	else:
		raise ValueError("Only token sizes upto 6 are supported, but {} is given.".format(token_dim))

	
def encode(X, filename=None, breadth_first=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits
	ftype = find_ftype(token_dim)
	X = X.astype(object)
	shifts = np.full(len(X), tree_depth-1, dtype=np.uint8)
	token = np.arange(ftype.bits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_dim:]
	flags = BitBuffer(filename)
	stack_size = 0
	msg = "{}: {:0>" + str(ftype.bits) + "}, StackSize: {:>10}, Done: {:>3.2f}%"
	done = np.sum(shifts, dtype=float)

	def expand(Xi, root=False):
		flag = 0
		
		if not root and len(Xi) == 1:
			pass
		else:
			x = X[Xi] >> shifts[Xi].reshape(-1, 1)
			for i, t in enumerate(token):
				m = np.all(x & 1 == t, axis=-1)
				if np.any(m):
					m &= shifts[Xi] != 0
					if np.any(m):
						xi = Xi[m]
						shifts[xi] -= 1
						yield expand(xi, root)
					flag |= 1<<i
				
		if log.verbose:
			progress = 100.0 - np.sum(shifts, dtype=float) / done * 100
			step = "Root" if root else "Tree"
			log(msg.format(step, bin(flag)[2:], stack_size, progress))
		flags.write(flag, ftype.bits, soft_flush=True)
		pass
	
	nodes = deque(expand(np.arange(len(X)), False))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
		stack_size = len(nodes)
	
	nodes = deque(expand(np.arange(len(X))[shifts != 0], True))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
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
			'--input_file', '-X',
			required=True,
			metavar='PATH'
			)
		
		parser.add_argument(
			'--dtype', '-t',
			metavar='TYPE',
			default='uint16'
			)
		
		parser.add_argument(
			'--dim', '-d',
			type=int,
			metavar='INT',
			default=3
			)
		
		parser.add_argument(
			'--filename', '-Y',
			metavar='PATH',
			default='tokentree.bin'
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
	
	X = np.fromfile(args.input_file, dtype=args.dtype)
	X = X.reshape(-1, args.dim)
	
	log("\nData:\n", X)
	log("\n---Encoding---\n")
	flags = encode(X, **args.__dict__)
	log("Flags safed to:", flags.fid.name)

