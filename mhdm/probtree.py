#!/usr/bin/env python3

## Build in
from collections import deque
import os.path as path
import sqlite3 as sql


class ProbTree():
	"""
	"""
	def __init__(self, filename, breadth_first=True):
		"""
		"""
		self.name = filename
		self.breadth_first = breadth_first
		self.state = deque()
		self.con = sql.connect(filename)
		self.cur = self.con.cursor()
		self.msg = None
		self.reset()
		self.cur.execute("CREATE TABLE IF NOT EXISTS model(cond text, symb int)")
		pass
	
	def __del__(self):
		self.con.close()
	
	def __bool__(self):
		return True
	
	def __iter__(self):
		self.reset()
		return self
	
	def __next__(self):
		return self
	
	def __call__(self, symbol, bits):
		self.update(symbol, bits)
	
	def reset(self):
		"""
		"""
		self.msg = None
		self.state.clear()
		
		def next_state(condition):
			if self.msg is None:
				yield next_state(condition)
			else:
				symbol, bits = self.msg
				self.msg = None
				dim = bits.bit_length() - 1
				args = (hex(condition)[2:], symbol)
				self.cur.execute('INSERT INTO model VALUES (?, ?)', args)
				
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

	def save(self):
		self.con.commit()
