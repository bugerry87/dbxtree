#!/usr/bin/env python3

## Installed
import numpy as np

## Local
from mhdm.utils import BitBuffer, log

	
def encode(x, filename=None, dtype=np.uint8, **kwargs):
	dtype = np.iinfo(dtype)
	bits = [1<<i for i in range(dtype.bits)]
	x = x.reshape(-1, 1) & bits > 0
	probs = np.argsort(np.sum(x, axis=0))
	flags = BitBuffer(filename)
	
	for x in x:
		try:
			flag = bool(np.any(x))
			if not flag:
				continue
			
			flag <<= 1
			flag |= bool(np.all(x))
			if flag & 1:
				continue
			
			for c in x:
				flag <<= 1
				flag |= bool(c)
			continue
			
			n = x.sum()
			flag <<= 1
			flag |= bool(n > dtype.bits//2)
			if flag & 1:
				x = ~x[probs]
				n = x.sum()
			else:
				x = x[probs[::-1]]
			
			flag <<= 1
			flag |= bool(n==1)
			
			if n == 2:
				flag <<= 1
				flag |= 1
			
			N = 8 - n
			for i, c in enumerate(x):
				flag <<= 1
				flag |= bool(c)
				
				if c:
					n -= 1
					N += 1
					if not n:
						break
				if N == i:
					break
		finally:
			n = flag.bit_length() if flag else 1
			flags.write(flag, n, soft_flush=True)
			if log.verbose:
				log(x.astype(int), "-->", bin(flag))
			
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
			description="Demo of Probability Flags",
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
			default='uint8'
			)
		
		parser.add_argument(
			'--filename', '-Y',
			metavar='PATH',
			default='probflag.bin'
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
	
	log("\nData:\n", X)
	log("\n---Encoding---\n")
	flags = encode(X, **args.__dict__)
	
	log("ProbFlags safed to:", flags.fid.name)

