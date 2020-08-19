#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np


def find_ftype(token_size):
	if token_size <= 3:
		return np.iinfo(np.uint8)
	elif token_size == 4:
		return np.iinfo(np.uint16)
	elif token_size == 5:
		return np.iinfo(np.uint32)
	elif token_size == 6:
		return np.iinfo(np.uint64)
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

	
def encode(X):
	token_size = X.shape[-1]
	token_depth = np.iinfo(X.dtype).bits
	ftype = find_ftype(token_size)
	X = X.astype(object)
	token = np.arange(1<<token_size, dtype=np.uint8).reshape(-1,1)
	token = np.unpackbits(token, axis=-1)[:,-token_size:]
	payload = np.zeros(token_size, dtype=object)
	flags = 0

	def expand(X, token_pos):
		flag = 0
		
		if len(X) == 1:
			payload <<= token_pos
			payload |= X.flatten()
		else:
			for i, t in enumerate(token):
				m = np.all(X & 1 == t, axis=-1)
				if np.any(m):
					if token_pos > 1:
						yield expand(token_pos-1, X[m] >> 1)
					flag |= 1<<i
		flags <<= ftype.bit
		flags |= flag
		pass
	
	nodes = deque(expand(X, token_depth))
	while nodes:
		node = nodes.popleft()
		nodes.extend(node)
	
	n = np.ceil(flags.bit_length/8.0).astype(int)
	flags = np.ndarray(n, dtype=np.uint8, buffer=flags.to_bytes(n, 'big'))
	n = np.ceil(payload[0].bit_length/8.0).astype(int)
	payload = np.vstack([np.ndarray(n, dtype=np.uint8, buffer=p.to_bytes(n, 'big')) for p in payload]).T
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
			'--input_file', '-X',
			nargs=1,
			metavar='PATH'
			)
		
		parser.add_argument(
			'--dtype', '-t',
			metavar='TYPE',
			default='uint64'
			)
		
		parser.add_argument(
			'--output_file', '-Y',
			metavar='PATH',
			default='tokentree.bin'
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
	X = np.fromfile(args.filename, dtype=args.dtype)
	
	print("\nData:\n", X)
	print("\n---Encoding---\n")
	flags, payload = encode(X)
	
	print("Flags:", flags.shape)
	print("Payload:", payload.shape)

	np.concatenate((flags, payload), axis=None).tofile(args.output_file)
	exit()
	
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
