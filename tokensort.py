#!/usr/bin/env python3

## Installed
import numpy as np

## Local
from mhdm.spatial import *


def pack_64(X,
	shifts=(0, 16, 36, 32, 48),
	masks=(0xFFFF, 0xFFFF, 0xFFF0, 0xF, 0xFFFF),
	outtype=np.uint16
	):
	"""
	"""
	outtype = np.iinfo(outtype)
	Y = np.zeros((len(X), 64//outtype.bits), dtype=outtype)
	
	for dim, (shift, mask) in enumerate(zip(shifts, masks)):
		s = shift % outtype.bits
		Y[:,shift//outtype.bits] |= np.bitwise_and((X[:,dim] << s), mask, dtype=outtype)
	return Y


def tile_merge(L, R):
	high = np.bitwise_and(L, 0xF0F0F0F0) + np.right_shift(np.bitwise_and(R, 0xF0F0F0F0), 4)
	low = np.left_shift(np.bitwise_and(L, 0x0F0F0F0F), 4) + np.bitwise_and(R, 0x0F0F0F0F)
	return low, high


def encode(X):
	assert(X.dtype == np.uint16)
	assert(len(X.shape) == 2)
	assert(X.shape[-1] == 4)
	
	## Pack to 64bit
	Y = np.zeros(len(X), dtype=np.uint64)
	for bit in range(16):
		for dim in range(4):
			Y += (np.bitwise_and(np.right_shift(X[:,dim], bit), 0b1)*2).astype(np.uint64)**(bit*4 + dim)
	Y.sort()
	
	#for y in Y:
	#	print("{:0>64}".format(bin(y)[2:]))
	
	## To 16*8bit 2Point Pack
	N = len(Y) // 2
	Y = np.ndarray((N,2,2), dtype=np.uint32, buffer=Y)
	Y = np.array([tile_merge(L, R) for L, R in zip(Y[:,0].T, Y[:,1].T)]).reshape(4,-1)
	Y = [np.ndarray(N*4, dtype=np.uint8, buffer=y) for y in Y]
	Y = np.hstack((np.vstack((Y[0], Y[1])).T.reshape(-1,8), np.vstack((Y[2], Y[3])).T.reshape(-1,8)))
	return Y.T


def decode(Y):
	Y = Y.reshape(16, -1).T
	N = len(Y)
	X = np.zeros((N,2,4), dtype=np.uint16)
	
	for i in range(8*16):
		p = (i%8)//4
		d = i%4
		B = i//8
		b = 0b1<<(i%8)
		X[:,p,d] += np.left_shift((np.bitwise_and(Y[:,B], b) == b).astype(np.uint16), B)
	return X.reshape(-1,4)


## Test
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
			description="Demo of TranSort",
			parents=parents
			)
		
		parser.add_argument(
			'--data', '-x',
			metavar='STRING',
			default=None
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
			default='tokensort.bin'
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
		
		parser.add_argument(
			'--visualize', '-V',
			metavar='STRING',
			nargs='*',
			default=[],
			choices=('cloud', 'dist')
			)
		
		return parser
	
	args, _ = init_argparse().parse_known_args()
	np.random.seed(args.seed)
	
	if args.data:
		import pykitti
		from mhdm.utils import *
		
		print("\nLoad data: {}".format(args.data))
		files = ifile(args.data)	
		frames = pykitti.utils.yield_velo_scans(files)
		X = np.vstack([np.hstack((f, np.full((len(f),1), i))) for i, f in enumerate(frames)])
		X[:,3] /= X[:,3].max()
		X[:,3] *= 0xF
		X[:,:3] *= 100
		X[:,2] += 2**11
		X[:,:2] += np.iinfo(args.input_type).max * 0.5
	else:
		X = np.random.__dict__[args.generator](*args.input_size)
		X -= X.min(axis=0)
		X /= X.max(axis=0)
		X *= np.iinfo(args.input_type).max
	
	X = np.round(X).astype(args.input_type)
	X = pack_64(X)
	X.tofile('org.bin')
	
	print("\nData: {}\n".format(X.shape), X)
	print("\n---Encoding---")
	
	Y = encode(X)
	Y.tofile(args.filename)
	
	print("\nEncoded:\n", Y.T)
	print("\n---Decoding---")
	
	X = decode(Y)
	print("\nDecoded:\n", X)

	if 'cloud' in args.visualize:
		import mhdm.viz as viz
		fig = viz.create_figure()
		I = X[:,2] & 0xF
		X[:,2] = X[:,2] >> 4
		viz.vertices(X, I, fig, None)
		viz.show_figure()
	
	if 'dist' in args.visualize:
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
	
		@ticker.FuncFormatter
		def major_formatter(i, pos):
			return "{:0>8}".format(bin(int(i))[2:])
		
		Y = Y.flatten()
		ax = plt.subplot(111)
		ax.set_ylim(-7, 263)
		ax.yaxis.set_major_formatter(major_formatter)
		ax.scatter(range(len(Y)), Y, s=0.2, marker='.')
		plt.show()