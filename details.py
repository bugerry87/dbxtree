#!/usr/bin/env python3

## Build In
from argparse import ArgumentParser
from os import path

## Installed
import numpy as np
from scipy.spatial import cKDTree
import py7zr

## Local
from mhdm.utils import log, ifile
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
		description="details",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	main_args.add_argument(
		'--ground_truth', '-X',
		metavar='WILDCARD',
		help='Name of the ground_truth file'
		)
	
	main_args.add_argument(
		'--samples', '-Y',
		metavar='WILDCARD',
		help='Name of the sample file'
		)
	
	main_args.add_argument(
		'--prefix', '-p',
		metavar='PATH',
		help='Prefix of the output files'
		)
	
	main_args.set_defaults(
		run=lambda **kwargs: main_args.print_help()
		)
	return main_args

def extract(X, Y):
	T = cKDTree(X)
	nn = T.query(Y)[-1]
	D = X[nn] - Y
	D = np.round(D * 1000)
	return D

def merge(Y, D):
	return Y + D / 1000

def main(args, unparsed):
	log.verbose = args.verbose
	for i, (X, Y) in enumerate(zip(ifile(args.ground_truth), ifile(args.samples))):
		X = lidar.load(X, shape=(-1,4), dtype=np.float32)[...,:3]
		Y = lidar.load(Y, shape=(-1,3), dtype=np.float32)
		D = extract(X, Y)
		Y = merge(Y, D)

		filename = f"{args.prefix}.{i:05d}.pts.bin"
		arcname = f"{args.prefix}.{i:05d}.dtl.bin"
		arcfile = f"{args.prefix}.{i:05d}.dtl.7z"
		
		Y.tofile(filename)
		D.astype(np.int8).tofile(arcname)

		with py7zr.SevenZipFile(arcfile, 'w') as z:
			z.write(arcname, path.basename(arcname))
		log(arcfile)
		pass


if __name__ == '__main__':
	main_args = init_main_args()
	main(*main_args.parse_known_args())