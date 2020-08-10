#!/usr/bin/env python3

## Installed
import numpy as np

## Local
from utils import ifile


UINT8_TO_TOKEN32 = np.array([int(bin(i).replace('b','x'),16) for i in range(256)], dtype=np.uint32)
TOKEN32_TO_UINT8 = dict([(token, byte) for byte, token in enumerate(UINT8_TO_TOKEN32)])


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


def featurize(X):
	N = len(X)
	X = np.ndarray((N,4,2), dtype=np.uint8, buffer=X)
	Y = np.empty((N,2), dtype=np.uint32)
	shifts = np.arange(4)
	
	Y[:,0] = np.sum(UINT8_TO_TOKEN32[X[:,:,0]] << shifts, axis=-1)
	Y[:,1] = np.sum(UINT8_TO_TOKEN32[X[:,:,1]] << shifts, axis=-1)
	return np.ndarray(N, dtype=np.uint64, buffer=Y)


def realize(Y):
	N = len(Y)
	Y = np.ndarray((N,2), dtype=np.uint32, buffer=Y)
	X = np.empty((N,4,2), dtype=np.uint8)
	m = 0x11111111
	
	for shift in range(4):
		if shift:
			Y = Y >> 1
		X[:,shift,0] = [TOKEN32_TO_UINT8[i] for i in (Y[:,0] & m)]
		X[:,shift,1] = [TOKEN32_TO_UINT8[i] for i in (Y[:,1] & m)]
	return np.ndarray((N,4), dtype=np.uint16, buffer=X)


def numeric_delta(X, offset=0):
	return np.concatenate((X[:offset+1], np.diff(X[offset:], axis=0)))


def pack_8x64(X):
	shape = (len(X)//8,8,8)
	Y = np.zeros(shape, dtype=np.uint8)
	X = np.ndarray(shape, dtype=np.uint8, buffer=X)
	
	for byte in range(8):
		for token in range(8):
			for bit in range(8):
				Y[:,byte,token] += ((X[:,bit,byte] >> token) & 0b1) << bit
	
	return Y.reshape(-1, 64)


def unpack_8x64(Y):
	assert(Y.dtype == np.uint8)
	
	Y = Y.reshape(64, -1).T
	N = len(Y)
	X = np.zeros((N,8), dtype=np.uint64)
	for i in range(8*64):
		p = i%8
		B = i//8
		b = 0b1<<p
		X[:,p] += ((2 if B else 1)*((Y[:,B] & b) == b).astype(np.uint64))**(B if B else 1)
	return X.flatten()


def encode(X):
	Y = featurize(X)
	Y.sort()
	Y = numeric_delta(Y)
	#for y in Y[:50]:
	#	print("{:0>64}".format(bin(y)[2:]))
	return pack_8x64(Y).T


def decode(Y):
	X = unpack_8x64(Y)
	X = np.cumsum(X)
	#for x in X[:50]:
	#	print("{:0>64}".format(bin(x)[2:]))
	return realize(X)


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
			description="Demo of TokenSort",
			parents=parents
			)
		
		parser.add_argument(
			'--kitti',
			metavar='STRING',
			default=None
			)
		
		parser.add_argument(
			'--decode', '-y',
			metavar='STRING',
			nargs='*',
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
	
	if args.decode:
		files = ifile(args.decode)
		Y = np.hstack([np.fromfile(f, dtype=np.uint8) for f in files])
		X = None
	elif args.kitti:
		import pykitti
		from mhdm.utils import *
		
		print("\nLoad data: {}".format(args.kitti))
		files = ifile(args.kitti)	
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
	
	if X is not None:
		X = np.round(X).astype(args.input_type)
		X = pack_64(X)
		X.tofile('org.bin')
		
		print("\nData: {}\n".format(X.shape), X)
		print("\n---Encoding---")
		Y = encode(X)
		Y.tofile(args.filename)
		print("\nEncoded:\n", Y[-16:].T)
	
	print("\n---Decoding---")
	X = decode(Y)
	print("\nDecoded:\n", X)

	if 'cloud' in args.visualize:
		import viz
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
		
		Y = Y.flatten()[::10]
		ax = plt.subplot(111)
		ax.set_ylim(-7, 263)
		ax.yaxis.set_major_formatter(major_formatter)
		ax.scatter(range(len(Y)), Y, s=0.2, marker='.')
		plt.show()
