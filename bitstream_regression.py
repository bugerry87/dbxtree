
## Build In
from argparse import ArgumentParser

## Installed
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

## Local
from mhdm.utils import ifile
from mhdm.range_coder import RangeEncoder, prob2cdf


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="Bit Stream Regression",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--filenames', '-X',
		metavar='PATH',
		nargs='+',
		help='Filenames to bit-files as input'
		)
	
	main_args.add_argument(
		'--visualize', '-V',
		action='store_true'
		)
	
	return main_args


def fnc(x, a, b):
	return a * np.exp(b * x)


def dx_fnc(x, a, b):
	return a * b * np.exp(b * x)


def slop2prob(slop):
	return slop / 2 + 0.5


def grad2cdf(grad, precision=16, dtype=np.uint64):
	probs = np.vstack((slop2prob(-grad), slop2prob(grad))).T
	cdf = prob2cdf(probs, precision, dtype=dtype)
	return cdf.astype(dtype)


def binarize(fname):
	bits = np.fromfile(fname, np.uint8)
	bits = bits[...,None] >> np.arange(8)[::-1] & 1
	return bits.reshape(-1)


def cumsum_bits(bits):
	y = bits.astype(float)
	y[y==0] = -1
	y = np.cumsum(y, dtype=float) / len(y)
	y -= y.min()
	x = np.arange(len(y), dtype=float) / len(y)
	return x, y


def main(filenames,
	skip=256,
	visualize=True
	):
	"""
	"""
	rc = RangeEncoder()	

	for fname in ifile(filenames):
		bits = binarize(fname)
		x, y = cumsum_bits(bits)
		args, covars = curve_fit(fnc, x[::skip], y[::skip])

		plt.title(fname)
		plt.plot(x[::skip], y[::skip], label='ground truth')
		plt.plot(x[::skip], fnc(x[::skip], *args), '--', label='regression')
		plt.legend()
		plt.show()
		
		grad = dx_fnc(x, *args)
		cdfs = grad2cdf(grad)
		rc.open('{}.rc.bin'.format(fname.replace('.bin', '')))
		for i, (grad, symb, cdf) in enumerate(zip(grad, bits, cdfs)):
			rc.update_cdf(symb, cdf)
			done = 100.0 * i/len(bits)
			print('Gradient: {:>6.2f}, Done: {:>6.2f}%'.format(grad, done), end='\r', flush=True)
		print()
		print('Num of output bits:', len(rc))
		rc.close()


if __name__ == '__main__':
	main_args = init_main_args()
	args = main_args.parse_args()
	main(**args.__dict__)