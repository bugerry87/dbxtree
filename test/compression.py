#!/usr/bin/env python3

## Installed
import numpy as np


class TokenTree():
	DEPTH_FIRST = 'depth_first'
	BREADTH_FIRST = 'breadth_first'
	payload_type = np.uint8
	
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
		
		def breadth_first(self, payload=False):
			for node in self:
				if not payload:
					yield node
				elif not node.flags:
					yield node.payload
			
			for node in self:
				for n in node.breadth_first(payload):
					yield n

	def __init__(self,
		payload_type = payload_type,
		iter_mode = DEPTH_FIRST
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
		if not payload:
			yield self.root
		elif not self.root.flags:
			yield self.root.payload
		
		for node in self.root.breadth_first(payload):
			yield node
	
	def encode(self, X):
		self.N = 0
		self.leaf_nodes = 0
		self.token_size = X.shape[-1]
		self.token_depth = np.iinfo(X.dtype).bits
		self.token = np.arange(1<<self.token_size, dtype=np.uint8).reshape(-1,1)
		self.token = np.unpackbits(self.token, axis=-1)[:,-self.token_size:]
		self.root = TokenTree.Node(self, self.token_depth-1, X)
		self.flags = np.array([node.flags for node in self], dtype=self.payload_type)
		self.payload = np.array([(*payload,) for payload in self.depth_first(True)], dtype=self.payload_type)
		return self.flags, self.payload
	
	def decode(self, X):
		pass


if __name__ == '__main__':
	X = np.random.randn(1000000, 3)
	X -= X.min(axis=0)
	X /= X.max(axis=0)
	X *= np.iinfo(np.uint16).max
	X = np.round(X).astype(np.uint16)
	
	tree = TokenTree()
	flags, payload = tree.encode(X)
	print("Data:\n", X)
	print("Leaf Nodes:", tree.leaf_nodes)
	print("Flags:", flags.shape)
	print("Payload:", payload.shape)
	print("Num of nodes:", len(tree))
	
	payload = np.concatenate((flags, payload), axis=None)
	payload.tofile('test2.bin')