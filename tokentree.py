#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np


def find_ftype(token_size):
	if token_size <= 3:
		return np.uint8
	elif token_size == 4:
		return np.uint16
	elif token_size == 5:
		return np.uint32
	elif token_size == 6:
		return np.uint64
	else:
		raise ValueError("Only token sizes upto 6 are supported, but {} is given.".format(token_size))


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
	def __init__(self, token_size):
		self.token = np.arange(1<<token_size, dtype=np.uint8).reshape(-1,1)
		self.token = np.unpackbits(self.token, axis=-1)[:,-token_size:]
	
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

	
def encode(X, ptype=np.uint8):
	ptype = np.iinfo(ptype)
	payload = []
	flags = []
	
	token_size = X.shape[-1]
	token_depth = np.iinfo(X.dtype).bits
	token = np.arange(1<<token_size, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_size:]

	def expand(token_pos, X):
		token_bit = np.left_shift(token, token_pos, dtype=np.int32)
		flag = 0
		
		if token_pos < ptype.bits and token_pos >= token_size and len(X) == 1:
			payload.extend(X.astype(ptype).flatten())
		else:
			for i, t in enumerate(token_bit):
				mask = np.all(np.bitwise_and(X, 1<<token_pos) == t, axis=-1)
				if np.any(mask):
					if token_pos:
						yield expand(token_pos-1, X[mask])
					flag |= 1<<i
		flags.append(flag)
		pass
	
	nodes = deque(expand(token_depth-1, X))
	while nodes:
		node = nodes.popleft()
		nodes.extend(node)
	
	ftype = find_ftype(token_size)
	flags = np.array(flags, dtype=ftype)
	payload = np.array(payload, dtype=ptype)
	return flags, payload


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
			description="Demo of TokenTree",
			parents=parents
			)
		
		parser.add_argument(
			'--input_size', '-X',
			metavar='INT',
			type=int,
			nargs=2,
			default=(1000000,3)
			)
		
		parser.add_argument(
			'--input_type', '-t',
			metavar='STRING',
			default='uint16'
			)
		
		parser.add_argument(
			'--filename', '-Y',
			metavar='PATH',
			default='tokentree.bin'
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
			metavar='FLAG',
			nargs='?',
			type=bool,
			default=False,
			const=True
			)
		
		return parser
	
	args, _ = init_argparse().parse_known_args()
	np.random.seed(args.seed)
	
	X = np.random.__dict__[args.generator](*args.input_size)
	X -= X.min(axis=0)
	X /= X.max(axis=0)
	X *= np.iinfo(args.input_type).max
	X = np.round(X).astype(args.input_type)
	
	print("\n---Encoding---\n")
	flags, payload = encode(X)
	
	print("\nData:\n", X)
	print("Flags:", flags.shape)
	print("Payload:", payload.shape)

	np.concatenate((flags, payload), axis=None).tofile(args.filename)
	
	print("\n---Decoding---\n")
	Y = Decoder(X.shape[-1]).expand(flags, payload.reshape(-1,X.shape[-1])).decode(X.dtype)
	print(Y)
	
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