
## Build in
from argparse import ArgumentParser
from collections import deque

## Installed
import numpy as np

## Local
from mhdm.utils import log, ifile
from mhdm.bitops import BitBuffer
import mhdm.spatial as spatial
import mhdm.bitops as bitops
import mhdm.lidar as lidar


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="SpatialTree",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	main_args.add_argument(
		'--outfile', '-o',
		metavar='PATH',
		default='',
		help='A file path for the output data'
		)
	
	main_args.set_defaults(
		run=lambda **kwargs: main_args.print_help()
		)
	
	return main_args


def init_encode_args(parents=[], subparser=None):
	if subparser:
		encode_args = subparser.add_parser('encode',
			help='Encode datapoints to a SpatialTree',
			conflict_handler='resolve',
			parents=parents
			)
	else:
		encode_args = ArgumentParser(
			description='Encode datapoints to a SpatialTree',
			conflict_handler='resolve',
			parents=parents
			)
	
	encode_args.add_argument(
		'--infile', '-X',
		metavar='FILE',
		help='Name of the input file'
		)
	
	encode_args.add_argument(
		'--radius', '-r',
		metavar='FLOAT',
		type=float,
		default=0.03,
		help='Accepted radius of error'
		)
	
	encode_args.add_argument(
		'--dim', '-d',
		metavar='INT',
		type=int,
		default=1,
		help='Dimensionality of the tree'
		)
	
	encode_args.add_argument(
		'--xtype', '-t',
		metavar='TYPE',
		default='float32',
		help='The expected data-type of the datapoints (datault=float32)'
		)
	
	encode_args.add_argument(
		'--xshape',
		nargs='+',
		metavar='SHAPE',
		default=(-1,4),
		help='Dimensionality of the input shape'
		)
	
	encode_args.add_argument(
		'--oshape',
		nargs='+',
		metavar='SHAPE',
		default=(-1,3),
		help='Dimensionality of the output shape'
		)
	
	encode_args.set_defaults(
		run=encode
		)
	
	return encode_args


def encode(infile, outfile,
	dim=1,
	radius=0.03,
	xshape=(-1,4),
	xtype='float32',
	oshape=(-1,3),
	**kwargs
	):
	"""
	"""
	def expand(X, bbox, i):
		if len(i) == 0:
			encode.count += 1
			log("BBox:", bbox, "bits:", i, "Points Detected:", encode.count)
			return
		if np.all(np.all(np.abs(X) <= radius, axis=-1)):
			flags.write(0, 1<<len(i), soft_flush=True)
			encode.count += 1
			log("BBox:", bbox, "bits:", i, "Points Detected:", encode.count)
			return
		m = X[...,i] >= 0
		bbox[...,i] *= 0.5
		X[...,i] += (1 - m.astype(bool)*2) * bbox[...,i]

		flag = 0
		t = np.packbits(m, -1, 'little').reshape(-1)
		for d in range(1<<len(i)):
			m = t==d
			if np.any(m):
				flag |= 1<<d
				args = np.argsort(bbox)[::-1]
				args = args[bbox[args] >= radius][:dim]
				yield expand(X[m].copy(), bbox.copy(), args)
		flags.write(flag, 1<<len(i), soft_flush=True)
	
	encode.count = 0
	flags = BitBuffer(outfile.replace('.bin', '.flg.bin'), 'wb')
	X = lidar.load(infile, xshape, xtype)[..., :oshape[-1]]
	bbox = np.abs(X).max(axis=0)
	i = np.argsort(bbox)[::-1][:dim]
	nodes = deque(expand(X, bbox, i))
	while nodes:
		node = nodes.popleft()
		nodes.extend(node)
	
	flags.close()
	log("Done")
	pass


def main(args, unparsed):
	log.verbose = args.verbose
	args.run(**args.__dict__)

if __name__ == '__main__':
	main_args = init_main_args()
	subparser = main_args.add_subparsers(help='Application Modes:')
	init_encode_args([main_args], subparser)
	main(*main_args.parse_known_args())