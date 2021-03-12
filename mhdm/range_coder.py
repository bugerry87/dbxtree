## Installed
import numpy as np

## Local
from . import bitops


def prob2cdf(probs, precision=16, floor=0, dtype=np.uint64):
	probs = np.array(probs, dtype=float) + floor
	shape = [*probs.shape]
	shape[-1] += 1
	cdf = np.zeros(shape)
	cdf[..., 1:] = probs 
	cdf /= np.linalg.norm(cdf, ord=1, axis=-1, keepdims=True)
	cdf = np.cumsum(cdf, axis=-1)
	cdf *= (1<<precision) - 1
	return cdf.astype(dtype)


class RangeCoder():
	"""
	"""
	def __init__(self, filename=None, precision=64):
		self.precision = int(precision)
		self.reset()
		pass
	
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
	def quat_range(self):
		return 1 << self.precision-2
	
	def reset(self):
		self.high = self.inner_range
		self.low = 0
		self.range = self.high - self.low + 1
	
	def update(self, start, end, total):
		assert(start < end <= total)
		assert(self.low < self.high)
		assert(self.low & self.inner_range == self.low)
		assert(self.high & self.inner_range == self.high)
		assert(self.quat_range <= self.range <= self.total_range)

		self.range //= int(total)
		self.high = self.low + int(end) * self.range - 1
		self.low = self.low + int(start) * self.range

		while self.low & self.half_range or not self.high & self.half_range:
			self._shift()
			self.low = (self.low<<1) & self.inner_range
			self.high = (self.high<<1) & self.inner_range
			self.high |= 1
		
		while self.low & ~self.high & self.quat_range != 0:
			self._underflow()
			self.low = self.low<<1 ^ self.half_range
			self.high = (self.high ^ self.half_range) << 1 | self.half_range | 1
		self.range = self.high - self.low + 1
		pass


class RangeEncoder(RangeCoder):
	"""
	"""
	def __init__(self, filename=None, precision=64):
		self.output = bitops.BitBuffer(filename, 'wb')
		super(RangeEncoder, self).__init__(filename, precision)
	
	def __len__(self):
		return len(self.output)
	
	def __bytes__(self):
		buffer = self.output.buffer << 1 | 1
		n_bits = buffer.bit_length()
		n_bytes = n_bits // 8
		n_tail = 8-n_bits % 8
		return (buffer << n_tail).to_bytes(n_bytes+bool(n_tail), 'big')[1:]
	
	def _shift(self):
		bit = self.low & self.half_range > 0
		self.output.write(bit, 1)
		while self.underflow:
			self.output.write(bit^1, 1)
			self.underflow -= 1

	def _underflow(self):
		self.underflow += 1
	
	def reset(self):
		super(RangeEncoder, self).reset()
		self.output.reset()
		self.underflow = 0
	
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

	def update_cdf(self, symbol, cdf=None):
		symbol = int(symbol)
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
		return bytes(self)
	
	def finalize(self):
		self.output.write(1, 1)
		self.output.flush(hard=True)


class RangeDecoder(RangeCoder):
	"""
	"""
	def __init__(self, input=None, precision=64):
		self.input = bitops.BitBuffer()
		self.window = 0
		super(RangeDecoder, self).__init__(precision)
		if input:
			self.set_input(input)
		pass

	def __add__(self, bytes):
		self.input + bytes
		return self
	
	def __radd__(self, bytes):
		self.input + bytes
		return self

	def _shift(self):
		self.window = self.window<<1 & self.inner_range
		self.window |= self.input.read(1, tail_zeros=True)

	def _underflow(self):
		self.window = (self.window & self.half_range) | (self.window<<1 & self.inner_range>>1)
		self.window |= self.input.read(1, tail_zeros=True)
		pass

	def reset(self):
		super(RangeDecoder, self).reset()
		self.input.reset()
	
	def set_input(self, input, reset=True):
		if reset:
			self.reset()
		if isinstance(input, str):
			self.input.open(input, reset)
		elif isinstance(input, bytes):
			self.input + input
		else:
			raise ValueError("Arg 'input' must be either a filename (str) or bytes.")
		if reset:
			self.window = self.input.read(self.precision)

	def update_cdf(self, cdf):
		symbol = 0
		end = len(cdf)
		total = int(cdf[-1])
		self.range = self.high - self.low + 1
		offset = self.window - self.low
		value = ((offset+1) * total - 1) // self.range
		assert(0 <= value < total)

		while end - symbol > 1:
			mid = (symbol + end) // 2
			if cdf[mid] > value:
				end = mid
			else:
				symbol = mid
		
		self.update(cdf[symbol], cdf[symbol+1], total)
		return symbol
	
	def updates(self, cdfs):
		return [self.update_cdf(cdf) for cdf in cdfs]