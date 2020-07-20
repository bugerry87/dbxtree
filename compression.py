#!/usr/bin/env python3

## Installed
import numpy as np


class TokenTree():
	DEPTH_FIRST = 'depth_first'
	BREADTH_FIRST = 'breadth_first'
	payload_type = np.uint32
	
	class Node():
		def __init__(self, tree, token_pos, X):
			self.token_pos = token_pos
			self.N = len(X)
			self.nodes = []
			self.flags = 0
			
			token = np.left_shift(tree.token, token_pos, dtype=X.dtype)
			if token_pos and self.N > 1:
				for t in token:
					mask = np.all(np.bitwise_and(X, 2**token_pos) == t, axis=-1)
					if np.any(mask):
						self.nodes.append(TokenTree.Node(tree, token_pos-1, X[mask]))
						self.flags |= 2**token_pos
			else:
				self.payload = X
						
			print("Node: Points {}, Nodes {}, Bit {}".format(self.N, len(self.nodes), token_pos+1))
			pass
		
		def __iter__(self):
			return iter(self.nodes)
		
		def depth_first(self):
			yield self
			for node in self:
				for n in node.depth_first():
					yield n
		
		def breadth_first(self):
			for node in self:
				yield node
			
			for node in self:
				for n in node.breadth_first():
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
	
	def depth_first(self):
		return self.root.depth_first()
		
	def breadth_first(self):
		yield self.root
		for node in self.root.breadth_first():
			yield node
	
	def encode(self, X):
		self.token_size = X.shape[-1]
		self.token_depth = np.iinfo(X.dtype).bits
		self.token = np.arange(2**self.token_size, dtype=np.uint8).reshape(-1,1)
		self.token = np.unpackbits(self.token, axis=-1)[:,-self.token_size:]
		self.root = TokenTree.Node(self, self.token_depth-1, X)
		self.payload = np.array([node.flags for node in self], dtype=self.payload_type)
		self.N = len(self.payload)
		return self.payload
	
	def decode(self, X):
		pass


if __name__ == '__main__':
	X = np.random.randn(1000000, 3)
	X = np.round(X * 30000).astype(np.int16)
	print("Data:\n", X)
	
	tree = TokenTree()
	payload = tree.encode(X)
	print("Payload:\n", np.sum(payload==0))
	print("Num of nodes:", len(tree))
	
	payload.tofile('test.bin')
	np.save('test.npy', payload)