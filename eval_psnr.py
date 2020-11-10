#!/usr/bin/env python

## BuildIn
from argparse import ArgumentParser
import os.path as path

## Installed
import numpy as np
import pcl
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

## Local
from mhdm.utils import ifile, log
from mhdm import lidar


def init_main_args(parents=[]):
	"""
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		main_args: The ArgumentParsers.
	"""
	main_args = ArgumentParser(
		description="Eval PSNR",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--decompressed', '-X',
		metavar='WILDCARD',
		nargs='+',
		help='A wildcard to a set of decompressed PLY files'
		)
	
	main_args.add_argument(
		'--compressed', '-Y',
		metavar='WILDCARD',
		nargs='*',
		help='A wildcard to the compressed files (must have same alphanumeric order!)'
		)
	
	main_args.add_argument(
		'--ground_truth', '-T',
		metavar='PLY',
		required=True,
		help='One PLY file as ground truth'
		)
	
	main_args.add_argument(
		'--peak', '-p',
		metavar='FLOAT',
		type=float,
		default=100.0,
		help='The expected peak of the PSNR (default=100m)'
		)
	
	main_args.add_argument(
		'--acc', '-a',
		metavar='FLOAT',
		type=float,
		default=0.03,
		help='The accuracy threshold (default=0.03m)'
		)
	
	main_args.add_argument(
		'--jobs', '-j',
		metavar='INT',
		type=int,
		default=8,
		help='Jobs for the KDTree query (default=8)'
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	main_args.add_argument(
		'--title',
		metavar='STRING',
		default="Point Cloud Compression Evaluation",
		help='A title for the plot'
		)
	
	return main_args


def main(args):
	files = ifile(args.decompressed, sort=True)
	if args.compressed:
		files = zip(files, ifile(args.compressed, sort=True))
	log.verbose = args.verbose
	T = pcl.load(args.ground_truth)
	T = np.asarray(T)
	tree = cKDTree(T)
	report = []
	
	for f in files:
		if args.compressed:
			x, y = f
		else:
			x, y = f, None
		X = pcl.load(x)
		X = np.asarray(X)
		delta, nn = tree.query(X, n_jobs=args.jobs)
		
		points = len(X)
		psnr = lidar.psnr(np.mean(delta**2), peak=args.peak)
		acc = np.sum(delta <= args.acc) * 100.0 / points
		max_delta = delta.max() * 1000
		if y:
			size = path.getsize(y)
			bpp = float(size) / points * 8
		else:
			size = 0
			bpp = 0
		entry = (x, points, psnr, acc, max_delta, size, bpp)
		report.append(entry)
		log(("{}:"
			"\n\tpoints={}"
			"\n\tpsnr={:2.2f}dB"
			"\n\tacc={:2.2f}%"
			"\n\tmax={:2.2f}mm"
			"\n\tsize={}"
			"\n\tbpp={:2.2f}").format(*entry))
	
	files, points, psnr, acc, max_delta, size, bpp = np.asarray(report).T
	files = [path.basename(f) for f in files]
	
	fig, axes = plt.subplots(1, 2, figsize=(12,8))
	fig.suptitle(args.title)
	axl = axes[0]
	axl.grid()
	axl.set_xticklabels(files, rotation=45, ha='right')
	axl.plot(files, points.astype(int), label='Points')
	axl.plot(files, size.astype(int), label='Size(Byte)')
	axl.legend()
	
	axl = axes[1]
	axl.grid()
	axl.set_ylabel('PSNR(dB), Acc(%), Max(mm)')
	axl.set_xticklabels(files, rotation=45, ha='right')
	axl.plot(files, psnr.astype(float), label='PSNR({}m)'.format(args.peak))
	axl.plot(files, acc.astype(float), label='Acc(3cm)')
	axl.plot(files, max_delta.astype(float), label='Max(mm)')
	axl.legend(loc='center left')
	
	axr = axl.twinx()
	axr.set_ylabel('bpp')
	axr.plot(files, bpp.astype(float), '--', label='bpp')
	axr.legend(loc='center right')
	
	plt.subplots_adjust(left=0.1, bottom=0.17, wspace=0.2)
	plt.show()
	pass


if __name__ == '__main__':
	main(init_main_args().parse_known_args()[0])
