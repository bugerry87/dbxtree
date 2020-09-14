#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np

## Local
from mhdm.utils import BitBuffer, log


class Node():
	def __init__(self, token=0):
		self.token = token
		self.payload = None
		self.nodes = []
	
	def __iter__(self):
		return iter(self.nodes)
	
	def expand(self, tree, flag, payload):
		if flag:
			for bit in range(tree.ftype.bits):
				if 1<<bit & flag:
					self.nodes.append(Node(tree.token[bit]))
		else:
			self.payload = next(payload)
		return self.nodes
	
	def decode(self, token_pos, x=0):
		token = np.left_shift(self.token, token_pos, dtype=np.int32)
		if self.payload is not None:
			yield np.bitwise_or(x + token, self.payload)
		elif token_pos:
			for node in self:
				for payload in node.decode(token_pos-1, x + token):
					yield payload
		else:
			yield x + token
		pass


class Decoder():
	def __init__(self, token_dim):
		self.token = np.arange(1<<token_dim, dtype=np.uint8).reshape(-1,1)
		self.token = np.unpackbits(self.token, axis=-1)[:,-token_dim:]
	
	def expand(self, flags, payload):
		self.ftype = np.iinfo(flags.dtype)
		self.root = Node()
		
		flag_iter = iter(flags)
		payload_iter = iter(payload)
		nodes = deque(self.root.expand(self, next(flag_iter), payload_iter))
		for flag in flag_iter:
			node = nodes.popleft()
			nodes.extend(node.expand(self, flag, payload_iter))
		return self
	
	def decode(self, dtype):
		dtype = np.iinfo(dtype)
		nodes = list(self.root.decode(dtype.bits))
		return np.array(nodes, dtype=dtype)

	
def encode(X, filename=None, breadth_first=False, big_first=False, payload=False, **kwargs):
	token_dim = X.shape[-1]
	tree_depth = np.iinfo(X.dtype).bits
	fbits = 1<<token_dim
	X = X.astype(object)
	token = np.arange(1<<token_dim, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_dim:]
	flags = BitBuffer(filename)
	stack_size = 0
	msg = "Layer: {:>2}, BranchFlag: {:0>" + str(fbits) + "}, StackSize: {:>10}"
	
	if payload is True:
		payload = BitBuffer(filename + '~') if filename else BitBuffer()

	def expand(X, bits):
		flag = 0
		
		if len(X) == 0:
			pass
		elif payload is not False and len(X) == 1:
			for d in range(token_dim):
				payload.write(int(X[:,d]), bits)
			payload.flush()
		else:
			x = X >> bits-1 if big_first else X
			for i, t in enumerate(token):
				m = np.all(x & 1 == t, axis=-1)
				if np.any(m):
					if bits > 1:
						yield expand(X[m] >> int(not big_first), bits-1)
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
	
	args, _ = init_argparse().parse_known_args()
	log.verbose = args.verbose
	
	X = np.fromfile(args.input_file, dtype=args.dtype)
	X = X[len(X)%args.dim:].reshape(-1,args.dim)
	X = np.unique(X, axis=0)
	
	log("\nData:", X.shape)
	log(X)
	log("\n---Encoding---\n")
	flags, payload = encode(X, **args.__dict__)
	
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
