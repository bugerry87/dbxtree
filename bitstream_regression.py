
## Build In
from argparse import ArgumentParser

## Installed
import numpy as np
import matplotlib.pyplot as plt

## Local
from mhdm.utils import ifile
from mhdm.range_coder import RangeEncoder


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
		'--input', '-X',
		metavar='PATH',
		nargs='+',
		help='Filenames to bit-files as input'
		)
	
	main_args.add_argument(
		'--visualize', '-V',
		action='store_true'
		)
	
	return main_args


def main(args, unparsed):
	plt.title('Bit Stream Evaluation')
	for inp in ifile(args.input):
		y = np.fromfile(inp, np.uint8)
		y = y[..., None] >> np.arange(8)[::-1] & 1
		y = y.reshape(-1).astype(np.int8)
		y[y==0] = -1
		y = np.cumsum(y)
		x = range(len(y))
		skip = 256
		plt.plot(x[::skip], y[::skip], label=inp)
	plt.legend()
	plt.show()


if __name__ == '__main__':
	main_args = init_main_args()
	main(*main_args.parse_known_args())