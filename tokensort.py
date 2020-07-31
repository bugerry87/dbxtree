#!/usr/bin/env python3

## Installed
import numpy as np

## Local
from mhdm.spatial import *


def XYZI_to_UVDI(XYZI, precision=100, dtype=np.uint16, HDL64=False):
	if HDL64:
		UVDI = cone_uvd(XYZI[:,:3], z_off=0.2)
		mask = UVD[:,1] < -0.16
		UVDI[mask] = cone_uvd(XYZI[:,:3][mask], z_off=0.13, r_off=-0.03)
	else:
		UVDI = sphere_uvd(XYZI[:,:3])
	UVDI = np.hstack((UVD, XYZI[:,-1]))
	UVDI[:,(0,1,3)] -= UVDI[:,(0,1,3)].min(axis=0)
	UVDI[:,(0,1,3)] /= UVDI[:,(0,1,3)].max(axis=0)
	UVDI[:,2] *= precision
	return UVDI.astype()


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
			choices=('cloud', 'disp')
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
		X = np.vstack([f for f in frames])
		X *= 100
		X += np.iinfo(args.input_type).max * 0.5
	else:
		X = np.random.__dict__[args.generator](*args.input_size)
		X -= X.min(axis=0)
		X /= X.max(axis=0)
		X *= np.iinfo(args.input_type).max
	
	X = np.round(X).astype(args.input_type)
	X.tofile('org.bin')
	
	print("\nData: {}\n".format(X.shape), X)
	print("\n---Encoding---")
	Y = encode(X)
	Y.tofile(args.filename)
	
	print("\nEncoded:")
	for i in range(50):
		print(Y[:,i])
	print('...')
	
	print("\n---Decoding---")
	
	X = decode(Y)
	print("\nDecoded:\n", X)

	if 'cloud' in args.visualize:
		from mhdm.viz import *
		fig = create_figure()
		vertices(X, X[:,3], fig, None)
		mlab.show()
	
	if 'disp' in args.visualize:
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