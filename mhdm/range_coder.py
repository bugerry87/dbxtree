

## Installed
import numpy as np

## Local
from . import bitobs


class RangeCoder():
	"""
	"""
	def __init__(self, filename=None, precision=64):
		self.precision = int(precision)
		self.range = self.inner_range
		self.low = 0
		self.output = bitops.BitBuffer(filename, 'wb')
		pass

	def __bytes__(self):
		return bytes(self.output)
	
	def __bool__(self):
		return True
	
	@property
	def total_range(self):
		return 1<<self.precision
	
	@property
	def half_range(self):
		return 1<<self.precision-1
	
	@property
	def inner_range(self):
		return (1<<self.precision)-1
	
	def reset(self):
		self.range = self.inner_range
		self.low = 0
		self.buffer.reset()
	
	def open(self, filename, reset=True):
		self.output.open(filename, 'wb', reset)
		if reset:
			self.reset()
		pass
	
	def close(self, reset=True):
		self.output.close(reset)
		if reset:
			self.reset()
		pass

	def emit(self):
		raise NotImplementedError()

	def underflow(self):
		raise NotImplementedError()
	
	def update(self, start, size, total):
		self.range //= int(total)
		self.low += int(start) * self.range
		self.range *= size
		self.high = self.low + self.range

		while self.low ^ self.high & self.half_range == 0:
			self.emit()
			self.low = (self.low<<1) & self.inner_range
			self.range <<= 1
			self.high = self.low + self.range
		
		while self.low & ~self.high & self.min_range != 0:
			self.underflow()
			self.low = (self.low<<1) ^ self.half_range


	def update_cdf(self, symbol, cdf=None):
		if cdf is None:
			start, size, total = symbol
		elif isinstance(cdf, dict):
			start, size, total = cdf[symbol]
		else:
			start = cdf[symbol]
			size = cdf[symbol+1]
			total = cdf[-1]
		self.update(start, size, total)