'''
Helper functions for this project.

Author: Gerald Baulig
'''

#Global libs
import numpy as np
from time import time
from glob import glob, iglob


def myinput(prompt, default=None, cast=None):
	''' myinput(prompt, default=None, cast=None) -> arg
	Handle an interactive user input.
	Returns a default value if no input is given.
	Casts or parses the input immediately.
	Loops the input prompt until a valid input is given.
	
	Args:
		prompt: The prompt or help text.
		default: The default value if no input is given.
		cast: A cast or parser function.
	'''
	while True:
		arg = input(prompt)
		if arg == '':
			return default
		elif cast != None:
			try:
				return cast(arg)
			except:
				print("Invalid input type. Try again...")
		else:
			return arg
	pass


def ifile(wildcards, sort=False, recursive=True):
	def sglob(wc):
		if sort:
			return sorted(glob(wc, recursive=recursive))
		else:
			return iglob(wc, recursive=recursive)

	if isinstance(wildcards, str):
		for wc in sglob(wildcards):
			yield wc
	elif isinstance(wildcards, list):
		if sort:
			wildcards = sorted(wildcards)
		for wc in wildcards:
			if any(('*?[' in c) for c in wc):
				for c in sglob(wc):
					yield c
			else:
				yield wc
	else:
		raise TypeError("wildecards must be string or list.")


def time_delta(start=None):
	''' time_delta() -> delta
	Captures time delta from last call.
	
	Yields:
		delta: Past time in seconds.
	'''
	if not start:
		start = time()
	
	while True:
		curr = time()
		delta = curr - start
		start = curr
		yield delta


class BitBuffer:
	"""
	Buffers bitwise to a file or memory.
	"""

	def __init__(self,
		filename=None
		):
		"""
		Init a BitBuffer.
		Opens and writes to a file from beginning if filename is given.
		Otherwise, all bits are kept in buffer.
		
		Args:
			filename: Opens a file from beginning.
		"""
		self.fid = open(filename, 'wb') if filename else None
		self.buffer = 0xFF
		self.len = 0
		pass
	
	def __len__(self):
		return self.len
	
	def __del__(self):
		self.close()
	
	def flush(self, hard=False):
		if not self.fid or self.fid.closed:
			return
	
		n_bits = self.buffer.bit_length()
		n_bytes = n_bits // 8
		n_tail = n_bits % 8
		
		if hard:
			buf = self.buffer.to_bytes(n_bytes+bool(n_tail), 'big')
			self.fid.write(buf[1:])
			self.buffer = 0xFF
			self.fid.flush()
		elif n_bytes > 1:
			buf = self.buffer.to_bytes(n_bytes+bool(n_tail), 'big')
			self.fid.write(buf[1:n_bytes])
			self.buffer = (0xFF << n_tail) | buf[-1] if n_tail else 0xFF
		pass
	
	def close(self):
		if self.fid:
			self.flush(True)
			self.fid.close()
	
	def open(self, filename):
		self.close()
		self.fid = open(filename, 'ab')
	
	def write(self, bits, shift, soft_flush=False):
		self.buffer <<= shift
		self.buffer |= bits
		if soft_flush:
			self.flush()
			
