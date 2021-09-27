#!/usr/bin/env python
""" 
Author: Gerald Baulig
"""

#Standard libs
from argparse import ArgumentParser

#3rd-Party libs
import numpy as np
from scipy.spatial import Delaunay

#Local libs
import mhdm.viz as viz
import mhdm.spatial as spatial
import mhdm.lidar as lidar
from mhdm.utils import *


def init_argparse(parents=[]):
	''' init_argparse(parents=[]) -> parser
	Initialize an ArgumentParser for this module.
	
	Args:
		parents: A list of ArgumentParsers of other scripts, if there are any.
		
	Returns:
		parser: The ArgumentParsers.
	'''
	parser = ArgumentParser(
		#description="Demo for embedding data via LDA",
		parents=parents
		)
	
	parser.add_argument(
		'--data', '-X',
		metavar='WILDCARD',
		help="Wildcard to the LiDAR files.",
		default='**.bin'
		)
	
	parser.add_argument(
		'--sort', '-s',
		action='store_true'
		)
	
	return parser


def validate(args):
	args.data = myinput(
		"Wildcard to the LiDAR files.\n" + 
		"	data ('**.bin'): ",
		default='**.bin'
		)
	
	args.sort = myinput(
		"Sort the files?\n" + 
		"	sort (False): ",
		default=False
		)
	return args


def yield_frames(files):
	for file in files:
		yield np.fromfile(file, np.float32).reshape(-1,4)


def main(args):
	# Load the data
	files = ifile(args.data, args.sort)	
	frames = yield_frames(files)
	fig = viz.create_figure()
	#plot = None

	for X in frames:
		if not len(X):
			break
		
		print("Input size:", X.shape)
		print("Plot X...")
		viz.vertices(X, X[:,3], fig, None)
		if input():
			break

		viz.clear_figure(fig)
		P = lidar.xyz2uvd(X[:,(1,0,2)])
		P[:,(0,1)] *= (100, 200)
		
		print("Plot polar...")
		viz.vertices(P, P[:,2], fig, None)
		if input():
			break
		viz.clear_figure(fig)
		
		print("Generate Surface...")
		mesh = Delaunay(P[:,(0,1)])
		Ti = mesh.simplices
		
		viz.mesh(P, Ti, None, fig, None)
		if input():
			break
		viz.clear_figure(fig)
		
		print("Remove planars...")
		fN = spatial.face_normals(X[Ti,:3])
		vN = spatial.vec_normals(Ti, fN)
		Mask = spatial.mask_planar(vN, fN, Ti.flatten(), 0.95)
		P = P[Mask]
		X = X[Mask]
		mesh = Delaunay(P[:,(0,1)])
		Ti = mesh.simplices
		
		print("New size:", X.shape)
		viz.mesh(X, Ti, None, fig, None)
		if input():
			break
		viz.clear_figure(fig)
		
		print("Remove Narrow Z-faces...")
		fN = spatial.face_normals(X[Ti,:3])
		Mask = np.abs(fN[:,1]) > 0.05
		Ti = Ti[Mask]
		
		viz.mesh(P[:,(2,0,1)], Ti, None, fig, None)
		if input():
			break
		viz.clear_figure(fig)
		
		print("Unfold View...")
		viz.mesh(X, Ti, None, fig, None)
		if input():
			break
		viz.clear_figure(fig)
		
		print("Final shape:", (np.unique(Ti.flatten())).shape)
		X.tofile('data/planarXYZ.bin')
		break
	return 0

if __name__ == '__main__':
	import sys
	parser = init_argparse()
	args, _ = parser.parse_known_args()
	if len(sys.argv) == 1:
		args = validate(args)
	exit(main(args))
