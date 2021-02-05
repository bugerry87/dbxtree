import numpy as np
import mhdm.range_coder as range_coder


def test_encoding_four_updates():
	encoder = range_coder.RangeEncoder()
	cdf = [0, 1<<8, 1<<10, 1<<16]
	encoder.update_cdf(1, cdf)
	encoder.update_cdf(1, cdf)
	encoder.update_cdf(2, cdf)
	encoder.update_cdf(0, cdf)
	pass


def test_encoding_stress_test():
	num_words = 15
	precision = 16
	floor = 0.01
	encoder = range_coder.RangeEncoder()
	def cdfs():
		p = np.random.rand(num_words)
		yield range_coder.cdf(p, precision, floor)
	
	symbols = np.random.random_integers(num_words, size=1000000) - 1
	encoder.updates(symbols, cdfs())
	pass


def test_encode_decode():
	num_words = 255
	precision = 32
	encoder = range_coder.RangeEncoder()
	decoder = range_coder.RangeDecoder()
	org_symbols = np.random.random_integers(num_words, size=1000) - 1
	u, c = np.unique(org_symbols, return_counts=True)
	probs = np.zeros(num_words)
	probs[u] = c
	cdfs = [range_coder.cdf(probs, precision)] * 1000
	
	code = encoder.updates(org_symbols, cdfs)
	decoder.set_input(code)
	dec_symbols = np.array(decoder.updates(cdfs))
	assert(np.all(org_symbols == dec_symbols))