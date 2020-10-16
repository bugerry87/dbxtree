
## Installed
import numpy as np


def quantization(X, bits_per_dim=None, qtype=object, offset=None, scale=None):
	if bits_per_dim is None:
		if qtype is object:
			raise ValueError("bits_per_dim cannot be estimated from type object!")
		else:
			bits_per_dim = np.iinfo(qtype).bits
	X = X.astype(float)
	
	if offset is None:
		offset = X.min(axis=0)
	X -= offset
	if scale is None:
		scale = (1<<np.array(bits_per_dim) - 1) / X.max(axis=0)
	X *= scale
	X = np.round(X).astype(qtype)
	
	return X, offset, scale 


def realization(X, offset, scale):
	X = X / scale
	X += offset
	return X


def serialize(X, bits_per_dim, dtype=object, offset=None, scale=None):
	X, offset, scale = quantization(X, bits_per_dim, dtype, offset, scale)
	shifts = np.cumsum(bits_per_dim, dtype=dtype) - bits_per_dim[0]
	X = np.sum(X<<shifts, axis=-1)
	return X, offset, scale


def deserialize(X, bits_per_dim):
	X = X.reshape(-1,1)
	masks = 1<<np.array(bits_per_dim) - 1
	shifts = np.cumsum(bits_per_dim, dtype=dtype) - bits_per_dim[0]
	X = X>>shifts & masks
	return X


def sort_bits(X, reverse=False, absp=False):
	shifts = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	p = np.array([np.sum(X>>i & 1) for i in range(shifts)])
	if absp:
		p = np.max((p, len(Y)-p), axis=0)
	p = np.argsort(p)
	if reverse:
		p = p[::-1]
	
	for i in range(shifts):
		Y |= (X>>p[i] & 1) << i
	return Y.reshape(shape), p.astype(np.uint8)


def permute_bits(X, p):
	shifts = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	
	for i in range(shifts):
		Y |= (X>>i & 1) << p[i]
	return Y.reshape(shape)


def reverse_bits(X):
	shifts = np.iinfo(X.dtype).bits
	Y = np.zeros_like(X)
	
	for low, high in zip(range(shifts//2), range(shifts-1, shifts//2 - 1, -1)):
		Y |= (X & 1<<low) << high | (X & 1<<high) >> high-low
	return Y


class BitBuffer:
	"""
	Buffers bitwise to a file or memory.
	"""

	def __init__(self,
		filename=None,
		mode='rb',
		interval=8
		):
		"""
		Init a BitBuffer.
		Opens a file from beginning if filename is given.
		Otherwise, all written bits are kept in buffer.
		
		Args:
			filename: Opens a file from beginning.
			mode: The operation mode, either 'rb', 'ab' or 'wb'.
			interval: Used in case of iterative reading
		"""
		self.fid = None
		self.buffer = 0xFF
		self.interval = interval
		
		if filename:
			self.open(filename, mode)
		pass
	
	def __len__(self):
		return self.buffer.bit_length() - 8
	
	def __del__(self):
		self.close()
	
	def __iter__(self):
		try:
			while True:
				yield self.read(self.interval)
		except (EOFError, BufferError):
			raise StopIteration
	
	@property
	def name(self):
		return self.fid.name if self.fid else None
	
	@property
	def closed(self):
		return self.fid.closed if self.fid else False
	
	def reset(self):
		"""
		Resets the internal buffer!
		"""
		self.buffer = 0xFF
	
	def flush(self, hard=False):
		"""
		Flushes the bit-stream to the internal byte-stream.
		May release some memory.
		
		Args:
			hard: Forces the flush to the byte-stream.
		
		Note:
			A hard-flush will append zeros to complete the last byte!
			Only recommended either on file close or when you are sure all bytes are complete!
		"""
		if self.closed:
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
	
	def close(self, reset=True):
		"""
		Performes a hard flush and closes the file if given.
		
		Args:
			reset: Whether the buffer is to reset on closing.
			       (default=True)
		"""
		if self.fid:
			self.flush(True)
			self.fid.close()
		if reset:
			self.reset()
	
	def open(self, filename, mode='rb', reset=True):
		"""
		(Re)opens a byte-stream to a file.
		The file-mode must be in binary-mode!
		
		Args:
			filename: The path/name of a file to be opened.
			mode: The operation mode, either 'rb', 'ab' or 'wb'.
			reset: Whether the buffer is to reset on re-opening.
			       (default=True)
		"""
		if 'b' not in mode:
			mode += 'b'
		self.close(reset)
		self.fid = open(filename, mode)
		return self
	
	def write(self, bits, shift, soft_flush=False):
		"""
		Write bits to BitBuffer.
		
		Args:
			bits: The bits added by 'or'-operation to the end of the bit-stream.
			shift: The number of shifts applied before bits got added.
			soft_flush: Flushes the bits to the internal byte-stream, if possible.
		
		Note:
			soft_flush requires a file in write mode!
		"""
		shift = int(shift)
		mask = (1<<shift) - 1
		self.buffer <<= shift
		self.buffer |= int(bits) & mask
		if soft_flush:
			self.flush()
	
	def read(self, bits, buf=0):
		"""
		"""
		bits = int(bits)
		
		if self.__len__() < bits and not self.closed:
			n_bytes = max(bits//8, 1)
			buffer = self.fid.read(buf + n_bytes)
			if len(buffer) < n_bytes:
				raise EOFError()
			elif buffer:
				self.buffer <<= len(buffer)
				self.buffer |= int.from_bytes(buffer, 'big')
		
		n_bits = self.__len__()
		if n_bits >= bits:
			mask = (1<<n_bits-bits) - 1
			result = self.buffer >> n_bits-bits
			self.buffer &= mask
		else:
			raise BufferError()
		
		return result