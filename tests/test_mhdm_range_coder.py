import mhdm.range_coder as range_coder


def test_single_encoding_update():
	encoder = range_coder.RangeEncoder()
	cdf = [0, 1<<8, 1<<10, 1<<16]
	encoder.update_cdf(1, cdf)
	encoder.update_cdf(1, cdf)
	encoder.update_cdf(2, cdf)
	encoder.update_cdf(0, cdf)
	pass