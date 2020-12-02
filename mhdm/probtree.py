#!/usr/bin/env python3

## Build in
from collections import deque
import os.path as path
import pickle

## Installed
import numpy as np



class ProbTree():
	"""
	"""
	def __init__(self, filename=None, breadth_first=True):
		"""
		"""
		self.breadth_first = breadth_first
		self.state = deque()
		self.model = {}
		self.msg = None
		self.reset()
		
		if path.isfile(filename):
			self.load(filename)
		else:
			self.name = filename
		pass
	
	def __bool__(self):
		return True
	
	def reset(self):
		"""
		"""
		self.msg = None
		self.model.clear()
		self.state.clear()
		
		def next_state(condition):
			if self.msg is None:
				yield next_state(condition)
			else:
				symbol, bits = self.msg
				dim = bits.bit_length() - 1
				self.msg = None
				
				if condition not in self.model:
					self.model[condition] = {symbol:1}
				elif symbol not in self.model[condition]:
					self.model[condition][symbol] = 1
				else:
					self.model[condition][symbol] += 1
				
				for i in range(bits):
					if symbol>>i & 1:
						yield next_state(condition<<dim | i)
		
		self.state.extend(next_state(1))
		pass
	
	def update(self, symbol, bits):
		"""
		"""
		self.msg = (symbol, bits)
		state = self.state.popleft() if self.breadth_first else self.state.pop()
		self.state.extend(state)
	
	def load(self, filename):
		"""
		"""
		self.name = filename
		with open(filename, 'rb') as fid:
			self.model.update(pickle.load(fid))
	
	def save(self, filename):
		"""
		"""
		self.name = filename
		with open(filename, 'wb') as fid:
			pickle.dump(self.model, fid)
