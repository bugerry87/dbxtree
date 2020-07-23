#!/usr/bin/env python3

## Installed
import numpy as np


class SeqTokenTree():
	DEPTH_FIRST = 'depth_first'
	BREADTH_FIRST = 'breadth_first'
	payload_type = np.uint32
	iter_mode = BREADTH_FIRST
	
	class Node():
		def __init__(self, tree, token_pos, X):
			self.N = len(X)
			self.nodes = []
			token = np.left_shift(tree.token, token_pos, dtype=np.int32)
			
			if not self.N:
				pass
			elif token_pos:
				for t in token:
					mask = np.all(np.bitwise_and(X, 1<<token_pos) == t, axis=-1)
					self.nodes.append(TokenTree.Node(tree, token_pos-1, X[mask]))
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
	
	def depth_first(self):
		return self.root.depth_first()
		
	def breadth_first(self):
		yield self.root
		for node in self.root.breadth_first():
			yield node
	
	def extract_payload(self):
		self.payload = np.array([node.N for node in self], dtype=self.payload_type)
		self.N = len(self.payload)
		return self.payload
	
	def encode(self, X):
		self.token_size = X.shape[-1]
		self.token_depth = np.iinfo(X.dtype).bits
		self.token = np.arange(1<<self.token_size, dtype=np.uint8).reshape(-1,1)
		self.token = np.unpackbits(self.token, axis=-1)[:,-self.token_size:]
		self.root = SeqTokenTree.Node(self, self.token_depth-1, X)
		pass
	
	def decode(self, X):
		pass


if __name__ == '__main__':
	print("Prepare Data...")
	X = np.random.randn(1000000, 3)
	X -= X.min(axis=0)
	X /= X.max(axis=0)
	X *= np.iinfo(np.uint16).max
	X = np.round(X).astype(np.uint16)
	
	print("Build Tree...")
	tree = SeqTokenTree()
	tree.encode(X)
	
	print("Extract Payload...")
	payload = tree.extract_payload()
	
	## Show Results
	print("Data:\n", X)
	print("Payload:", payload.shape)
	print("Num of nodes:", len(tree))
	
	## Store the file
	print("\nStore Data...")
	payload.tofile('test.bin')