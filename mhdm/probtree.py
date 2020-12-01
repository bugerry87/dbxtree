#!/usr/bin/env python3

## Build in
from collections import deque
import pickle


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
		
		if filename:
			self.name = filename
			self.load(filename)
		else:
			self.name = None
		self.reset()
		pass
	
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
				self.msg = None
				mask = (1<<bits) - 1
				condition <<= bits
				condition |= symbol & mask
				
				if condition not in self.model:
					self.model[condition] = {symbol:1}
				elif symbol not in self.model[condition]:
					self.model[condition][symbol] = 1
				else:
					self.model[condition][symbol] += 1
				
				print('\n', hex(condition), self.model[condition])
				input()
				
				for i in range(bits):
					if symbol>>i & 1:
						yield next_state(condition)
		
		self.state.extend(next_state(0xFF))
		pass
	
	def write(self, symbol, bits, **kwargs):
		"""
		"""
		self.msg = (symbol, bits)
		state = self.state.popleft() if self.breadth_first else self.state.pop()
		self.state.extend(state)
	
	def load(self, filename):
		"""
		"""
		with open(filename, 'rb') as fid:
			self.model |= pickle.load(fid).items()
	
	def save(self, filename):
		"""
		"""
		with open(filename, 'wb') as fid:
			pickle.dump(self.model, fid)
