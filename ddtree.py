#!/usr/bin/env python3

## Build in
from argparse import ArgumentParser
from collections import deque

## Installed
import numpy as np

## Local
from mhdm.utils import BitBuffer, log


def init_argparse(parents=[]):
	''' init_argparse(parents=[]) -> parser
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	'''
	parser = ArgumentParser(
		description="Demo of DDTree",
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
		default='uint16'
		)
	
	parser.add_argument(
		'--dims', '-d',
		nargs='*',
		type=int,
		metavar='INT',
		default=None
		)
	
	parser.add_argument(
		'--output', '-o',
		metavar='PATH',
		default='ddtree.bin'
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
		'--sort_bits', '-B',
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
		'--payload', '-p',
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


def find_ftype(dim):
	if dim <= 3:
		dtype = np.uint8
	elif dim == 4:
		dtype = np.uint16
	elif dim == 5:
		dtype = np.uint32
	elif dim == 6:
		dtype = np.uint64
	else:
		raise ValueError("Data of only up to 6 dimensions are supported")
	return dtype


def sort_bits(X, reverse=False):
	shifts = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	p = np.array([np.sum(X>>i&1) for i in range(shifts)])
	p = np.max((p, len(Y)-p), axis=0)
	p = np.argsort(p)
	if reverse:
		p = p[::-1]
	
	for i in range(shifts):
		Y |= (X>>p[i]&1)<<i
	return Y.reshape(shape), p.astype(np.uint8)


def unsort_bits(X, p):
	shifts = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	
	for i in range(shifts):
		Y |= (X>>i&1)<<p[i]
	return Y.reshape(shape)


def reverse_bits(X):
	shifts = np.iinfo(X.dtype).bits
	Y = np.zeros_like(X)
	
	for low, high in zip(range(shifts//2), range(shifts-1, shifts//2 - 1, -1)):
		Y |= (X & 1<<low) << high | (X & 1<<high) >> high-low
	return Y

	
def encode(X, dims=None, output=None, breadth_first=False, payload=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits
	flags = BitBuffer(output + '.flg.bin')
	stack_size = 0
	msg = "Layer: {:>2}, BranchFlag: {:>16}, StackSize: {:>10}"
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin') if output else BitBuffer()

	def expand(X, layer, tail):
		flag = 0
		if dims:
			dim = dims[layer] if layer < len(dims) else 1
		else:
			dim = int(np.clip(np.log(len(X)+1), 1, 6))
		fbit = 1<<dim
		mask = (1<<dim)-1
		
		if len(X) == 0 or dim == 0:
			pass
		elif payload is not False and len(X) == 1:
			payload.write(int(X), tail, soft_flush=True)
		else:
			for t in range(fbit):
				m = X & mask == t
				if np.any(m):
					yield expand(X[m]>>dim, layer+1, tail-dim)
					flag |= 1<<t
		if log.verbose:
			log(msg.format(layer, hex(flag)[2:], stack_size))
		flags.write(flag, fbit, soft_flush=True)
		pass
	
	nodes = deque(expand(X, 0, tree_depth))
	while nodes:
		node = nodes.popleft() if breadth_first else nodes.pop()
		nodes.extend(node)
		stack_size = len(nodes)
	
	flags.close()
	if payload:
		payload.close()
	return flags, payload


def main(args):
	log.verbose = args.verbose
	if args.output:
		args.output = args.output.replace('.bin', '')
	header = None
	
	X = np.fromfile(args.compress, dtype=args.dtype)
	X = np.unique(X)
	
	if args.sort_bits:
		X, p = sort_bits(X, args.reverse)
		log("\nBit order:\n", p)
		if args.output:
			header = args.output + '.hdr.bin'
			p.tofile(header)			
	elif args.reverse:
		X = reverse_bits(X)
	
	if log.verbose:
		log("\nData:", X.shape)
		log(X)
		log("\n---Encoding---\n")
		input('Press Enter to continue!')
	flags, payload = encode(X, **args.__dict__)
	
	log("Header safed to:", header)
	log("Flags safed to:", flags.fid.name)
	log("Payload safed to:", payload.fid.name)
	exit()
	np.concatenate((flags, payload), axis=None).tofile(args.output_file)
	
	log("\n---Decoding---\n")
	Y = Decoder(X.shape[-1]).expand(flags, payload.reshape(-1,X.shape[-1])).decode(X.dtype)
	log(Y)
	
	if args.visualize:
		import matplotlib.pyplot as plt
		import matplotlib.ticker as ticker
		from mpl_toolkits.mplot3d import Axes3D
	
		@ticker.FuncFormatter
		def major_formatter(i, pos):
			return "{:0>8}".format(bin(int(i))[2:])
		
		fig = plt.figure()
		ax = fig.add_subplot((111), projection='3d')
		ax.scatter(*X[:,:3].T, c=X.sum(axis=-1), s=0.5, alpha=0.5, marker='.')
		plt.show()
		
		ax = plt.subplot(111)
		ax.scatter(range(len(flags)), flags, 0.5, marker='.')
		ax.set_ylim(-7, 263)
		ax.yaxis.set_major_formatter(major_formatter)
		plt.show()


if __name__ == '__main__':
	args, _ = init_argparse().parse_known_args()
	main(args)
