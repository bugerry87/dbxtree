#!/usr/bin/env python3

## Installed
import numpy as np


def tile_merge(L, R):
	left = np.bitwise_and(L, 0xF0F0F0F0) + np.right_shift(np.bitwise_and(R, 0xF0F0F0F0), 4)
	right = np.left_shift(np.bitwise_and(L, 0x0F0F0F0F), 4) + np.bitwise_and(R, 0x0F0F0F0F)
	return left, right


def encode(X):
	assert(X.dtype == np.uint16)
	assert(len(X.shape) == 2)
	assert(X.shape[-1] == 4)
	
	## Pack to 64bit
	Y = np.zeros(len(X), dtype=np.uint64)
	for bit in range(16):
		for dim in range(4):
			Y += (np.bitwise_and(np.right_shift(X[:,dim], bit), 0b1)*2).astype(np.uint64)**((bit+1)*dim + dim)
	Y.sort()
	
	#for b in Y:
	#	print("{:0>64}".format(bin(b)[2:]))
	
	## To 16*8bit 2Point Pack
	N = len(Y) // 2
	Y = np.ndarray((N,2,2), dtype=np.uint32, buffer=Y)
	Y = np.array([tile_merge(L, R) for L, R in zip(Y[:,0].T, Y[:,1].T)]).reshape(4,-1)
	Y = [np.ndarray(N*4, dtype=np.uint8, buffer=y) for y in Y]
	Y = np.hstack((np.vstack((Y[0], Y[1])).T.reshape(-1,8), np.vstack((Y[2], Y[3])).T.reshape(-1,8)))
	return Y.T


if __name__ == '__main__':
	from argparse import ArgumentParser
	import matplotlib.pyplot as plt
	import matplotlib.ticker as ticker
	
	def init_argparse(parents=[]):
		''' init_argparse(parents=[]) -> parser
		Initialize an ArgumentParser for this module.
		
		Args:
			parents: A list of ArgumentParsers of other scripts, if there are any.
			
		Returns:
			parser: The ArgumentParsers.
		'''
		parser = ArgumentParser(
			description="Demo of TranSort",
			parents=parents
			)
		
		parser.add_argument(
			'--input_size', '-X',
			metavar='INT',
			type=int,
			nargs=2,
			default=(100,4)
			)
		
		parser.add_argument(
			'--input_type', '-t',
			metavar='STRING',
			default='uint16'
			)
		
		parser.add_argument(
			'--filename', '-Y',
			metavar='PATH',
			default='transort.bin'
			)
		
		parser.add_argument(
			'--generator', '-g',
			metavar='STRING',
			choices=('rand','randn'),
			default='rand'
			)
		
		parser.add_argument(
			'--seed', '-s',
			metavar='INT',
			type=int,
			default=0
			)
		
		return parser
	
	args, _ = init_argparse().parse_known_args()
	np.random.seed(args.seed)
	
	X = np.random.__dict__[args.generator](*args.input_size)
	X -= X.min(axis=0)
	X /= X.max(axis=0)
	X *= np.iinfo(args.input_type).max
	X = np.round(X).astype(args.input_type)
	X.tofile('org.bin')
	
	print("\n---Encoding---\n")
	Y = encode(X)
	Y.tofile(args.filename)
	
	print("\nData:\n", X)
	print("\n Encoded:\n", Y.T)
	
	if len(X) <= 1000000:
		@ticker.FuncFormatter
		def major_formatter(i, pos):
			return "{:0>8}".format(bin(int(i))[2:])
	
		Y = Y.flatten()
		ax = plt.subplot(111)
		ax.set_ylim(-7, 263)
		ax.yaxis.set_major_formatter(major_formatter)
		ax.scatter(range(len(Y)), Y, 0.5, marker='.')
		plt.show()
