#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np

## Local
from mhdm.utils import BitBuffer, log

	
def encode(X, filename=None, breadth_first=False, big_first=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits - 1
	fbits = 1 << token_dim
	X = X.astype(object)
	shifts = np.full(len(X), tree_depth if big_first else 0, dtype=np.int16)
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_dim:]
	flags = BitBuffer(filename)
	stack_size = 0
	msg = "Layer: {:>2}, {:0>" + str(fbits) + "}, StackSize: {:>10}, Done: {:>3.2f}%"
	done = len(X) * tree_depth
	stop_bit = 0 if big_first else tree_depth
	increment = -1 if big_first else 1

	def expand(Xi):
		flag = 0
		
		if len(Xi) == 1:
			pass
		else:
			x = X[Xi] >> shifts[Xi].reshape(-1, 1)
			for i, t in enumerate(token):
				m = np.all(x & 1 == t, axis=-1)
				if np.any(m):
					m &= shifts[Xi] != stop_bit
					if np.any(m):
						xi = Xi[m]
						shifts[xi] += increment
						yield expand(xi)
					flag |= 1<<i
				
		if log.verbose:
			progress = np.sum(shifts, dtype=float) / done * 100
			if big_first:
				progress = 100.0 - progress
			log(msg.format(layer, bin(flag)[2:], stack_size, progress))
		flags.write(flag, fbits, soft_flush=True)
		pass
	
	report = {}
	layers = range(tree_depth, -1, -1) if big_first else range(tree_depth+1)
	for layer in layers:
		m = shifts == layer
		if not np.any(m):
			continue
	
		nodes = deque(expand(np.arange(len(X))[m]))
		while nodes:
			node = nodes.popleft() if breadth_first else nodes.pop()
			nodes.extend(node)
			stack_size = len(nodes)
		report[layer] = m.sum()
	
	log(report)
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
			'--big_first', '-B',
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
	X = np.unique(X, axis=0)
	
	log("\nData:", X.shape)
	log(X)
	log("\n---Encoding---\n")
	flags = encode(X, **args.__dict__)
	log("Flags safed to:", flags.fid.name)

