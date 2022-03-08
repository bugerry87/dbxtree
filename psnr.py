#!/usr/bin/env python

## BuildIn
from argparse import ArgumentParser
import os.path as path

## Installed
import numpy as np
#import matplotlib.pyplot as plt
from scipy.spatial import cKDTree, Delaunay

## Local
from mhdm import spatial, lidar, nbittree
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
		description="Eval PSNR",
		conflict_handler='resolve',
		parents=parents
		)
	
	main_args.add_argument(
		'--samples', '-X',
		metavar='WILDCARD',
		required=True,
		nargs='+',
		help='A wildcard to the decompressed files'
		)
	
	main_args.add_argument(
		'--ground_truth', '-Y',
		metavar='WILDCARD',
		required=True,
		nargs='+',
		help='A wildcard to the original files (must have same alphanumeric order!)'
		)
	
	main_args.add_argument(
		'--peak', '-p',
		metavar='FLOAT',
		type=float,
		default=0,
		help='The expected peak of the PSNR (default=0), (dynamic=0)'
		)
	
	main_args.add_argument(
		'--acc', '-a',
		metavar='FLOAT',
		type=float,
		default=0.03,
		help='The accuracy threshold (default=0.03m)'
		)
	
	main_args.add_argument(
		'--knn',
		metavar='INT',
		type=int,
		default=1,
		help='Number of nearest neighbors for normal estimation (default=1)'
		)
	
	main_args.add_argument(
		'--verbose', '-v',
		action='store_true'
		)
	
	main_args.add_argument(
		'--Visualize', '-V',
		action='store_true'
		)
	
	main_args.add_argument(
		'--title',
		metavar='STR',
		default="Peak Signal to Noise Ratio",
		help='A title for the plot'
		)
	
	main_args.add_argument(
		'--jobs', '-j',
		metavar='INT',
		type=int,
		default=8,
		help='Jobs for the KDTree query (default=8)'
		)
	return main_args


def vec_normals(X):
	U = lidar.xyz2uvd(X, z_off=0.0, d_off=0.0, mode='sphere') #-0.13 0.03
	U[:,(0,1)] *= (100, 200)
	Ti = Delaunay(U[:,:2]).simplices
	fN = spatial.face_normals(X[Ti], True)
	vN = spatial.vec_normals(Ti, fN, True)
	return vN


def eigen_normals(X):
	return np.vstack([np.linalg.eig(np.cov(x, rowvar=False))[-1][None,:,-1] for x in X])


def reverb(iteratable, default=None):
	while True:
		i = next(iteratable, default)
		if i != default:
			j = i
			yield j
		else:
			yield j


def main(args):
	log.verbose = args.verbose
	files = zip(ifile(args.samples, sort=True), reverb(ifile(args.ground_truth, sort=True)))
	report = []
	
	for in_file, gt_file in files:
		if path.splitext(in_file)[-1] == '.pkl':
			X = nbittree.decode(in_file)[0][...,:3]
		elif path.splitext(in_file)[-1] == '.bin':
			X = lidar.load(in_file, (-1,3), np.float32)[...,:3]
		else:
			X = lidar.load(in_file)[...,:3]
		
		if path.splitext(gt_file)[-1] == '.bin':
			Y = lidar.load(gt_file, (-1,4), np.float32)[...,:3]
		else:
			Y = lidar.load(gt_file)[...,:3]
		
		X = np.unique(X, axis=0)
		Y = np.unique(Y, axis=0)
		Xpoints = len(X)
		Ypoints = len(Y)
		Xtree = cKDTree(X)
		Ytree = cKDTree(Y)

		knn = args.knn if args.knn > 3 else 1
		XYdelta, XYnn = Xtree.query(Y, k=knn, n_jobs=args.jobs)
		YXdelta, YXnn = Ytree.query(X, k=knn, n_jobs=args.jobs)
		
		if args.knn <= 1:
			XYmse = np.mean(XYdelta**2)
			YXmse = np.mean(YXdelta**2)
		elif args.knn <= 3:
			XYvn = vec_normals(X)
			YXvn = vec_normals(Y)
			XYmse = np.sum(((X[XYnn] - Y) * YXvn)**2, axis=-1).mean()
			YXmse = np.sum(((Y[YXnn] - X) * XYvn)**2, axis=-1).mean()
		else:
			print("Estimate eigen normals XY")
			XYvn = eigen_normals(X[XYnn])
			print("Estimate eigen normals YX")
			YXvn = eigen_normals(Y[YXnn])
			xy = np.argmin(XYdelta, axis=-1)
			yx = np.argmin(YXdelta, axis=-1)
			XYdelta = XYdelta[range(len(XYdelta)),xy]
			YXdelta = YXdelta[range(len(YXdelta)),yx]
			XYnn = XYnn[range(len(XYnn)),xy]
			YXnn = YXnn[range(len(YXnn)),yx]
			XYmse = np.sum(((X[XYnn] - Y) * XYvn)**2, axis=-1).mean()
			YXmse = np.sum(((Y[YXnn] - X) * YXvn)**2, axis=-1).mean()
		
		Xpeak = X.max(axis=0).prod()
		Ypeak = Y.max(axis=0).prod()
		XYpsnr = lidar.psnr(XYmse, peak=args.peak or Xpeak)
		YXpsnr = lidar.psnr(YXmse, peak=args.peak or Ypeak)
		XYacc = np.sum(XYdelta <= args.acc) * 100.0 / Ypoints
		YXacc = np.sum(YXdelta <= args.acc) * 100.0 / Xpoints
		sym_psnr = lidar.psnr(XYmse + YXmse, peak=args.peak or Xpeak + Ypeak)
		XYcd = np.mean(XYdelta)
		YXcd = np.mean(YXdelta)
		sym_cd = XYcd + YXcd

		entry = (Xpoints, Ypoints, XYpsnr, YXpsnr, XYacc, YXacc, Xpeak**0.5, Ypeak**0.5, XYmse, YXmse, XYcd, YXcd, sym_cd, sym_psnr)
		report.append(entry)
		log(("{}:"
			"\n            {:^10}   {:^10}"
			"\n  points:   {:>10}   {:>10}"
			"\n  psnr:     {:10.2f}dB {:10.2f}dB"
			"\n  acc:      {:10.2f}%  {:10.2f}%"
			"\n  peak:     {:10.4f}m  {:10.4f}m"
			"\n  mse:      {:10.4f}m  {:10.4f}m"
			"\n  cd:       {:10.4f}m  {:10.4f}m"
			"\n  sym cd:   {:10.4f}m"
			"\n  sym psnr: {:10.2f}dB"
			).format(in_file, 'XY', 'YX', *entry))
		del Xtree
		del Ytree
	
	means = np.asarray(report, float).mean(axis=0)
	log(("MEANS:"
		"\n            {:^10}   {:^10}"
		"\n  points:   {:10.0f}   {:10.0f}"
		"\n  psnr:     {:10.2f}dB {:10.2f}dB"
		"\n  acc:      {:10.2f}%  {:10.2f}%"
		"\n  peak:     {:10.4f}m  {:10.4f}m"
		"\n  me:       {:10.4f}m  {:10.4f}m"
		"\n  cd:       {:10.4f}m  {:10.4f}m"
		"\n  sym cd:   {:10.4f}m"
		"\n  sym psnr: {:10.2f}dB"
		).format('XY', 'YX', *means))
	
	mins = np.asarray(report, float).min(axis=0)
	log(("MIN:"
		"\n            {:^10}   {:^10}"
		"\n  points:   {:10.0f}   {:10.0f}"
		"\n  psnr:     {:10.2f}dB {:10.2f}dB"
		"\n  acc:      {:10.2f}%  {:10.2f}%"
		"\n  peak:     {:10.4f}m  {:10.4f}m"
		"\n  me:       {:10.4f}m  {:10.4f}m"
		"\n  cd:       {:10.4f}m  {:10.4f}m"
		"\n  sym cd:   {:10.4f}m"
		"\n  sym psnr: {:10.2f}dB"
		).format('XY', 'YX', *mins))
	
	maxs = np.asarray(report, float).max(axis=0)
	log(("Max:"
		"\n            {:^10}   {:^10}"
		"\n  points:   {:10.0f}   {:10.0f}"
		"\n  psnr:     {:10.2f}dB {:10.2f}dB"
		"\n  acc:      {:10.2f}%  {:10.2f}%"
		"\n  peak:     {:10.4f}m  {:10.4f}m"
		"\n  me:       {:10.4f}m  {:10.4f}m"
		"\n  cd:       {:10.4f}m  {:10.4f}m"
		"\n  sym cd:   {:10.4f}m"
		"\n  sym psnr: {:10.2f}dB"
		).format('XY', 'YX', *maxs))
	pass


if __name__ == '__main__':
	main(init_main_args().parse_known_args()[0])
