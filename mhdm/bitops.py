

## Build In
import os.path as path

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
		offset = -X.min(axis=0)
	X += offset
	if scale is None:
		scale = X.max(axis=0)
		m = scale != 0
		scale[m] = ((1<<np.array(bits_per_dim)) - 1)[m] / scale[m]
	X *= scale
	X = np.round(X).astype(qtype)
	return X, offset, scale 


def realization(X, offset, scale, xtype=float):
	scale = np.array(scale)
	m = scale != 0
	X = X.astype(xtype)
	X[:,m] /= scale[m]
	X -= offset
	return X


def serialize(X, bits_per_dim, qtype=object, offset=None, scale=None):
	X, offset, scale = quantization(X, bits_per_dim, qtype, offset, scale)
	shifts = np.cumsum([0] + bits_per_dim[:-1], dtype=qtype)
	X = np.sum(X<<shifts, axis=-1, dtype=qtype)
	return X, offset, scale


def deserialize(X, bits_per_dim, qtype=object):
	X = X.reshape(-1,1)
	masks = (1<<np.array(bits_per_dim, dtype=qtype)) - 1
	shifts = np.cumsum([0] + bits_per_dim[:-1], dtype=qtype)
	X = (X>>shifts) & masks
	return X


def pattern(X, bits=None):
	if bits is None:
		bits = np.iinfo(X.dtype).bits
	X = X.flatten()
	p = np.array([np.sum(X>>i & 1) for i in range(bits)])
	p = np.argmax((p, len(X)-p), axis=0)
	p = np.packbits(p, bitorder='little')
	p = int.from_bytes(p.tobytes(), 'little')
	return p


def sort(X, bits=None, reverse=False, absp=False):
	if bits is None:
		bits = np.iinfo(X.dtype).bits
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	p = np.array([np.sum(X>>i & 1) for i in range(bits)])
	if absp:
		p = np.max((p, len(Y)-p), axis=0)
	p = np.argsort(p)
	if reverse:
		p = p[::-1]
	
	for i in range(bits):
		Y |= (X>>p[i] & 1) << i
	return Y.reshape(shape), p.astype(np.uint8)


def permute(X, p):
	shape = X.shape
	X = X.flatten()
	Y = np.zeros_like(X)
	
	for i, p in enumerate(p):
		Y |= (X>>i & 1) << p
	return Y.reshape(shape)


def reverse(X, bits=None):
	if bits is None:
		bits = np.iinfo(X.dtype).bits
	Y = np.zeros_like(X)
	
	for low, high in zip(range(bits//2), range(bits-1, bits//2 - 1, -1)):
		Y |= (X & 1<<low) << (high-low) | (X & 1<<high) >> (high-low)
	return Y


def transpose(X, bits=None, dtype=object):
	if X.ndim > 1 and X.shape[-1] == 8:
		if bits is None:
			bits = np.iinfo(X.dtype).bits
		Xt = np.hstack([np.packbits(X>>i & 1) for i in range(bits)])
	elif X.dtype == np.uint8:
		if bits is None:
			bits = np.iinfo(dtype).bits
		X = X.reshape(bits, -1)
		Xt = np.sum([np.unpackbits(X[i], axis=-1).astype(dtype) << i for i in range(bits)], axis=0, dtype=dtype)
		Xt = Xt.reshape(-1, 8)
	else:
		ValueError("The last dimension must have either a shape of 8 or contain data of type uint8")
	return Xt


class BitBuffer():
	"""
	Buffers bitwise to a file or memory.
	"""

	def __init__(self,
		filename=None,
		mode='rb',
		interval=8,
		buf=1024
		):
		"""
		Init a BitBuffer.
		Opens a file from beginning if filename is given.
		Otherwise, all written bits are kept in buffer.
		
		Args:
			filename: Opens a file from beginning.
			mode: The operation mode, either 'rb', 'ab' or 'wb'.
			interval: Used in case of iterative reading.
			buf: Bytes to buffer on read or write to file.
		"""
		self.fid = None
		self.buffer = 0xFF
		self.interval = interval
		self.buf = buf
		self.size = 0
		
		if filename:
			self.open(filename, mode)
		pass
	
	def __len__(self):
		return self.size
	
	def __del__(self):
		self.close()
	
	def __iter__(self):
		try:
			while True:
				yield self.read(self.interval)
		except (EOFError, BufferError):
			raise StopIteration
	
	def __bytes__(self):
		n_bits = self.buffer.bit_length()
		n_bytes = n_bits // 8
		n_tail = 8-n_bits % 8
		return (self.buffer << n_tail).to_bytes(n_bytes+bool(n_tail), 'big')
	
	def __bool__(self):
		return True
	
	@property
	def name(self):
		return self.fid.name if self.fid else None
	
	@property
	def closed(self):
		return self.fid.closed if self.fid else False
	
	def tell(self):
		return self.fid.tell() if self.fid else 0
	
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
		n_tail = -n_bits % 8
		
		if hard:
			self.buffer <<= n_tail
			buf = self.buffer.to_bytes(n_bytes+bool(n_tail), 'big')
			self.fid.write(buf[1:])
			self.buffer = 0xFF
			self.fid.flush()
		elif n_bytes > self.buf:
			self.buffer <<= n_tail
			buf = self.buffer.to_bytes(n_bytes+bool(n_tail), 'big')
			self.fid.write(buf[1:n_bytes])
			self.buffer = ((0xFF00 | buf[-1]) >> n_tail) if n_tail else 0xFF
		pass
	
	def close(self, reset=True):
		"""
		Performes a hard flush and closes the file if given.
		
		Args:
			reset: Whether the buffer is to reset on closing.
			       (default=True)
		"""
		if self.fid:
			if 'r' not in self.fid.mode:
				self.flush(True)
			self.fid.close()
		if reset:
			self.reset()
		pass
	
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
		if 'r' in mode:
			self.size = path.getsize(self.name)
		else:
			self.size = 0
		pass
	
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
		pass
	
	def read(self, bits):
		"""
		"""
		bits = int(bits)
		n_bits = self.buffer.bit_length() - 8
		
		if n_bits < bits and not self.closed:
			n_bytes = max(bits//8, 1)
			buffer = self.fid.read(self.buf + n_bytes)
			if len(buffer) < n_bytes:
				raise EOFError()
			elif buffer:
				self.buffer <<= len(buffer)*8
				self.buffer |= int.from_bytes(buffer, 'big')
			n_bits = self.buffer.bit_length() - 8
		
		if n_bits >= bits:
			mask = (1<<bits) - 1
			result = (self.buffer >> n_bits-bits) & mask
			self.buffer &= (1<<n_bits-bits) - 1
			self.buffer |= 0xFF << n_bits-bits
		else:
			raise BufferError()
		
		return result