
## Installed
import numpy as np

## Local
import mhdm.bitops as bitops


def gen_codec(i):
	pass


def encode(X, codebook, output=None):
	if not isinstance(output, bitops.BitBuffer):
		output = bitops.BitBuffer(output, 'wb')
	
	for x in X:
		c = codebook[x]
		shift = c.bit_length() - 1
		output.write(c, shift, soft_flush=True)

	return output


class StaticModel():

	def __init__(self, X):
		symbols, probs = np.unique(X, return_counts=True)
		i = np.argsort(probs)
		self.symbols = symbols[i]
		self.probs = probs[i]
		self.codec = 
		pass
