#!/usr/bin/env python3

## Build in
from collections import deque

## Installed
import numpy as np


class Node():
	def __init__(self, tree, token_pos, X):
		tree.N += 1
		self.ID = tree.N
		self.token_pos = token_pos
		self.N = len(X)
		self.nodes = []
		self.flags = 0
		token = np.left_shift(tree.token, token_pos, dtype=np.int32)
		
		if token_pos < tree.payload_type.bits and token_pos >= tree.token_size and self.N == 1:
			self.payload = X.astype(tree.payload_type).flatten()
		else:
			for i, t in enumerate(token):
				mask = np.all(np.bitwise_and(X, 1<<token_pos) == t, axis=-1)
				if np.any(mask):
					if token_pos:
						self.nodes.append(TokenTree.Node(tree, token_pos-1, X[mask]))
					else:
						tree.leaf_nodes += 1
					self.flags |= 1<<i
		print(self)
		pass
	
	def __str__(self):
		return "Node-{:->12}: {:>12} Points, {:>3} Nodes, Bit {:>2}, Flag {:0>8}".format(
			self.ID,
			self.N,
			len(self.nodes),
			self.token_pos+1,
			bin(self.flags)[2:]
			)
	
	def __iter__(self):
		return iter(self.nodes)
	
	def depth_first(self, payload=False):
		if not payload:
			yield self
		elif not self.flags:
			yield self.payload
		
		for node in self:
			for n in node.depth_first(payload):
				yield n


class TokenTree():
	DEPTH_FIRST = 'depth_first'
	BREADTH_FIRST = 'breadth_first'
	payload_type = np.uint8
	iter_mode = BREADTH_FIRST
	
	def __init__(self,
		payload_type = payload_type,
		iter_mode = iter_mode
		):
		"""
		"""
		self.payload_type = np.iinfo(payload_type)
		self.iter_mode = iter_mode
		self.N = 0
		pass
	
	def __len__(self):
		return self.N
	
	def __iter__(self):
		if self.iter_mode is TokenTree.DEPTH_FIRST:
			return self.depth_first()
		elif self.iter_mode is TokenTree.BREADTH_FIRST:
			return self.breadth_first()
		else:
			raise ValueError("Unknown iteration mode: {}!".format(self.iter_mode))
	
	def depth_first(self, payload=False):
		return self.root.depth_first(payload)
		
	def breadth_first(self, payload=False):
		nodes = deque(self.root)
		while nodes:
			node = nodes.popleft()
			nodes += node.nodes
			if not payload:
				yield node
			elif node.flags == 0:
				yield node.payload
	
	def encode(self, X):
		self.N = 0
		self.leaf_nodes = 0
		self.token_size = X.shape[-1]
		self.token_depth = np.iinfo(X.dtype).bits
		self.token = np.arange(1<<self.token_size, dtype=np.uint8).reshape(-1,1)
		self.token = np.unpackbits(self.token, axis=-1)[:,-self.token_size:]
		self.root = TokenTree.Node(self, self.token_depth-1, X)
		self.flags = np.array([node.flags for node in self.breadth_first()], dtype=self.payload_type)
		self.payload = np.array([(*payload,) for payload in self.breadth_first(True)], dtype=self.payload_type)
		return self.flags, self.payload
	
	def decode(self, X):
		pass


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
			default='output.bin'
			)
		
		parser.add_argument(
			'--generator', '-g',
			metavar='STRING',
			choices=('rand','randn'),
			default='randn'
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
	
	tree = TokenTree()
	flags, payload = tree.encode(X)
	
	print("\nData:\n", X)
	print("Leaf Nodes:", tree.leaf_nodes)
	print("Flags:", flags.shape)
	print("Payload:", payload.shape)
	print("Num of nodes:", len(tree))
	
	@ticker.FuncFormatter
	def major_formatter(i, pos):
		return "{:0>8}".format(bin(int(i))[2:])
	
	ax = plt.subplot(111)
	ax.scatter(range(len(flags)), flags, 0.5, marker='.')
	ax.set_ylim(-7, 263)
	ax.yaxis.set_major_formatter(major_formatter)
	plt.show()

	payload = np.concatenate((flags, payload), axis=None)
	payload.tofile(args.filename)
