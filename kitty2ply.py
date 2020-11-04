#!/usr/bin/env python

## BuildIn
from argparse import ArgumentParser
import os.path as path

## Installed
import numpy as np
import pcl

## Local
from mhdm.utils import ifile, log


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="Kitti2PLY",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--input', '-X',
		metavar='WILDCARD',
		nargs='+',
		help='A wildcard to a set of kitti scans'
		)
	
	main_args.add_argument(
		'--output', '-Y',
		metavar='DIR',
		default=None,
		help='A directory for the output data'
		)
	
	main_args.add_argument(
		'--binary', '-b',
		action='store_true',
		help='Flag whether the PLY-files are to be stored in binary or (default) as text'
		)
	
	main_args.add_argument(
		'--insensity', '-i',
		action='store_true',
		help='Flag whether to include the intensites or (default) not '
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	return main_args


def main(args):
	files = ifile(args.input)
	log.verbose = args.verbose
	
	for f in files:
		output = path.join(args.output, path.splitext(path.basename(f))[0] + '.ply')
		X = np.fromfile(f, dtype=np.float32).reshape(-1,4)
		if not args.insensity:
			P = pcl.PointCloud(X[:,:-1])
		else:
			P = pcl.PointCloud_PointXYZI(X)
		pcl.save(P, output, binary=args.binary)
		log("Convert {} to {}".format(f, output))


if __name__ == '__main__':
	main(init_main_args().parse_known_args()[0])
