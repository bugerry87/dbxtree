#!/usr/bin/env python3

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
	radius=0.03,
	xshape=(-1,4),
	xtype='float32',
	oshape=(-1,3),
	**kwargs
	):
	"""
	"""
	def expand(X):
		flag = 0
		if np.all(X == 0):
			encode.count += 1
			log("Points detected:", encode.count)
			buffer.write(flag, 2, soft_flush=True)
			return
		m = (X & 1).astype(bool)
		flag <<= 1
		if np.any(m):
			flag |= 1
			yield expand(X[m] >> 1)
		flag <<= 1
		if np.any(~m):
			flag |= 1
			yield expand(X[~m] >> 1)
		buffer.write(flag, 2, soft_flush=True)
	
	bits_per_dim = [16,16,16]
	encode.count = 0
	buffer = BitBuffer(outfile, 'wb')
	X = lidar.load(infile, xshape, xtype)[..., :oshape[-1]]
	X, offset, scale = bitops.serialize(X, bits_per_dim, scale=[100,100,100], qtype=np.uint64)
	X, permute, pattern = bitops.sort(X, sum(bits_per_dim), True, False)

	nodes = deque(expand(X))
	while nodes:
		node = nodes.popleft()
		nodes.extend(node)
	
	buffer.close()
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