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
		default='uint16'
		)
	
	parser.add_argument(
		'--dim', '-d',
		type=int,
		metavar='INT',
		default=3
		)
	
	parser.add_argument(
		'--output', '-o',
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


def decode(Y, output=None, dim=2, dtype=np.uint32, breadth_first=False, seed_length=8, **kwargs):
	dim += 1
	dtype = np.iinfo(dtype)
	fbits = 1 << dim
	depth = dtype.bits
	token = np.arange(fbits, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-dim:].astype(dtype)
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

	
def encode(X, output=None, breadth_first=False, payload=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits
	fbits = 1<<token_dim
	token = np.arange(1<<token_dim, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_dim:]
	flags = BitBuffer(output + '.flg.bin')
	stack_size = 0
	msg = "Layer: {:>2}, BranchFlag: {:0>" + str(fbits) + "}, StackSize: {:>10}"
	
	if payload is True:
		payload = BitBuffer(output + '.pyl.bin') if output else BitBuffer()

	def expand(X, bits):
		flag = 0
		
		if len(X) == 0:
			pass
		elif payload is not False and len(X) == 1:
			for d in range(token_dim):
				payload.write(int(X[:,d]), bits)
			payload.flush()
		else:
			for i, t in enumerate(token):
				m = np.all(X & 1 == t, axis=-1)
				if np.any(m):
					if bits > 1:
						yield expand(X[m] >> 1, bits-1)
					flag |= 1<<i
		if log.verbose:
			log(msg.format(tree_depth-bits, bin(flag)[2:], stack_size))
		flags.write(flag, fbits, soft_flush=True)
		pass
	
	nodes = deque(expand(X, tree_depth))
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
	X = X[:(len(X)//args.dim)*args.dim].reshape(-1,args.dim)
	if args.sort_bits:
		X, p = sort_bits(X, args.reverse)
		log("\nBit order:", p)
		if args.output:
			header = args.output + '.hdr.bin'
			p.tofile(header)			
	elif args.reverse:
		X = reverse_bits(X)
	X = np.unique(X, axis=0)
	
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
