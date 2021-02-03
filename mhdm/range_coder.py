

## Installed
import numpy as np

## Local
from . import bitobs


class RangeEncoder():
	"""
	"""
	def __init__(self, filename=None, precision=64):
		self.precision = int(precision)
		self.range = self.inner_range
		self.low = 0
		self.underflow = 0
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

	def _shift(self):
		bit = self.low & self.half_range > 0
		self.output.write(bit, 1)
		while self.underflow:
			self.output.write(bit^1, 1)

	def _underflow(self):
		self.underflow += 1
	
	def update(self, start, size, total):
		self.range //= int(total)
		self.low += int(start) * self.range
		self.range *= size
		self.high = self.low + self.range

		while self.low ^ self.high & self.half_range == 0:
			self._shift()
			self.low = self.low<<1 & self.inner_range
			self.range <<= 1
			self.high = self.low + self.range
		
		while self.low & ~self.high & self.min_range != 0:
			self._underflow()
			self.low = self.low<<1 ^ self.half_range
			self.high = (self.high ^ self.half_range) << 1 | self.half_range | 1
			self.range = self.high - self.low
		pass

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


class RangeDecoder(RangeEncoder):
	"""
	"""
	def __init__(self, input_stream, output_file=None, precision=64):
		super(RangeDecoder, self).__init__(output_file, precision)
		if isinstance(input_stream, str):
			self.input = bitobs.BitBuffer(input_stream, 'rb')
		elif isinstance(input_stream, bytes):
			self.input = bitobs.BitBuffer()
			self.input.buffer = b'\xff' + input_stream
		else:
			raise ValueError("Arg 'input_stream' must be either a filename (str) or bytes.")
		
		self.window = self.input.read(precision)
		pass

	def _shift(self):
		self.window = self.window<<1 & self.inner_range | self.input.read(1)

	def _underflow(self):
		pass

	def update_cdf(self, cdf):
		pass