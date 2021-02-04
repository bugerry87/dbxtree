

## Installed
import numpy as np

## Local
from . import bitops


class RangeCoder():
	"""
	"""
	def __init__(self, filename=None, precision=64):
		self.output = bitops.BitBuffer(filename, 'wb')
		self.precision = int(precision)
		self.reset()
		pass

	def __bytes__(self):
		return bytes(self.output)
	
	def __bool__(self):
		return True
	
	def _shift(self):
		raise NotImplementedError()

	def _underflow(self):
		raise NotImplementedError()
	
	@property
	def total_range(self):
		return 1 << self.precision
	
	@property
	def half_range(self):
		return 1 << self.precision-1
	
	@property
	def inner_range(self):
		return (1 << self.precision) - 1
	
	@property
	def min_range(self):
		return 1 << self.precision-2
	
	def reset(self):
		self.high = self.inner_range
		self.low = 0
		self.range = self.high - self.low + 1
		self.output.reset()
	
	def open_output_file(self, filename, reset=True):
		self.output.open(filename, 'wb', reset)
		if reset:
			self.reset()
		pass
	
	def close_output_file(self, reset=True):
		self.output.close(reset)
		if reset:
			self.reset()
		pass
	
	def update(self, start, end, total):
		assert(start < end <= total)
		assert(self.low < self.high)
		assert(self.low & self.inner_range == self.low)
		assert(self.high & self.inner_range == self.high)
		assert(self.min_range <= self.range <= self.total_range)

		self.range //= int(total)
		self.high = self.low + int(end) * self.range - 1
		self.low = self.low + int(start) * self.range

		while self.low & self.half_range or not self.high & self.half_range:
			self._shift()
			self.low = (self.low<<1) & self.inner_range
			self.high = (self.high<<1) & self.inner_range
			self.high |= 1
		
		while self.low & ~self.high & self.min_range != 0:
			self._underflow()
			self.low = self.low<<1 ^ self.half_range
			self.high = (self.high ^ self.half_range) << 1 | self.half_range | 1
		self.range = self.high - self.low + 1
		pass


class RangeEncoder(RangeCoder):
	"""
	"""
	def __init__(self, filename=None, precision=64):
		super(RangeEncoder).__init__(filename, precision)
	
	def _shift(self):
		bit = self.low & self.half_range > 0
		self.output.write(bit, 1)
		while self.underflow:
			self.output.write(bit^1, 1)
			self.underflow -= 1

	def _underflow(self):
		self.underflow += 1
	
	def reset(self):
		super(RangeEncoder).reset()
		self.underflow = 0

	def update_cdf(self, symbol, cdf=None):
		if cdf is None:
			start, end, total = symbol
		elif isinstance(cdf, dict):
			start, end, total = cdf[symbol]
		else:
			start = cdf[symbol]
			end = cdf[symbol+1]
			total = cdf[-1]
		self.update(start, end, total)
	
	def updates(self, symbols, cdfs=None):
		if cdfs is None:
			for symbol in symbols:
				self.update_cdf(symbol)
		else:
			for symbol, cdf in zip(symbols, cdfs):
				self.update_cdf(symbol, cdf)


class RangeDecoder(RangeCoder):
	"""
	"""
	def __init__(self, input_stream=None, precision=64):
		self.input = bitops.BitBuffer()
		super(RangeDecoder, self).__init__(precision)
		if input_stream:
			self.set_input(input_stream)
		pass

	def _shift(self):
		self.window = self.window<<1 & self.inner_range
		self.window |= self.input.read(1)

	def _underflow(self):
		self.window = (self.window & self.half_range) | (self.window<<1 & self.inner_range>>1)
		self.window |= self.input.read(1)
		pass

	def reset(self):
		super(RangeDecoder, self).reset()
		self.input.reset()
	
	def set_input(self, input_stream, reset=True):
		if reset:
			self.reset()
		if isinstance(input_stream, str):
			self.input.open(input_stream, reset)
		elif isinstance(input_stream, bytes):
			self.input.buffer = b'\xff' + input_stream
		else:
			raise ValueError("Arg 'input_stream' must be either a filename (str) or bytes.")
		self.window = self.input.read(self.precision)

	def update_cdf(self, cdf):
		symbol = 0
		end = len(cdf)
		total = cdf[-1]
		self.range = self.high - self.low + 1
		self.offset = self.window - self.low
		value = ((offset+1) * total - 1) // self.range

		while end - symbol > 1:
			mid = (symbol + end) // 2
			if cdf[mid] > value:
				end = mid
			else:
				symbol = mid
		
		self.update(cdf[symbol], cdf[symbol+1], total)
		return symbol