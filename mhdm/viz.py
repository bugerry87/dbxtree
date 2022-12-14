#!/usr/bin/env python3
"""
Author: Gerald Baulig
"""

from mayavi import mlab
from mayavi.core.ui.api import MlabSceneModel, SceneEditor
from traits.api import HasTraits
from traitsui.api import View, Item, Group

class GUI(HasTraits):
	def __init__(self, items=None):
		self.fig = Instance(MlabSceneModel, ())
		self.view = View(
			Group(
				Item('scene1',
					editor=SceneEditor(), height=250,
					width=300),
				'button1',
				show_labels=False
				),
			resizable=True
			)


def create_figure(**kwargs):
	return mlab.figure(**kwargs)


def clear_figure(fig):
	mlab.clf(fig)


def show_figure(**kwargs):
	mlab.show(**kwargs)


def mesh(X, T, Y, fig, plot=None, **kwargs):
	if not len(X):
		raise ValueError("Error: Empty frame!")

	if plot == None:
		plot = mlab.triangular_mesh(
			X[:,0],
			X[:,1],
			X[:,2],
			T,
			scalars=Y,
			figure=fig,
			**kwargs
			)
	else:
		plot.mlab_source.reset(
			x = X[:,0],
			y = X[:,1],
			z = X[:,2],
			triangles = T,
			scalars = Y
		)
	return plot


def normals(X, N, **kwargs):
	mlab.quiver3d(
		X[:,0],
		X[:,1],
		X[:,2],
		N[:,0],
		N[:,1],
		N[:,2],
		**kwargs)


def vertices(X, Y, fig, plot=None, **kwargs):
	if not len(X):
		raise ValueError("Error: Empty frame!")
	if 'mode' not in kwargs:
		kwargs['mode'] = 'point'

	if plot == None:
		plot = mlab.points3d(
			X[:,0],
			X[:,1],
			X[:,2],
			Y,
			figure=fig,
			**kwargs
			)
	else:
		plot.mlab_source.reset(
			x = X[:, 0],
			y = X[:, 1],
			z = X[:, 2],
			scalars = Y
		)
	return plot


def lines(X, Y, fig, plot=None, colormap='spectral', tube_radius=None, **kwargs):
	if not len(X):
		raise ValueError("Error: Empty frame!")

	if plot == None:
		plot = mlab.plot3d(
			X[:,0],
			X[:,1],
			X[:,2],
			Y,
			colormap=colormap,  # 'bone', 'copper',
			tube_radius=tube_radius,
			figure=fig,
			**kwargs
			)
	else:
		plot.mlab_source.reset(
			x = X[:, 0],
			y = X[:, 1],
			z = X[:, 2],
			scalars = Y
		)
	return plot


if __name__ == '__main__' or 'PLOT_MAIN' in globals():
	#Standard libs
	import sys
	from argparse import ArgumentParser

	#3rd-Party libs
	import numpy as np
	import pykitti

	#Local libs
	from utils import *


	def init_argparse(parents=[]):
		''' init_argparse(parents=[]) -> parser
		Initialize an ArgumentParser for this module.
		
		Args:
			parents: A list of ArgumentParsers of other scripts, if there are any.
			
		Returns:
			parser: The ArgumentParsers.
		'''
		parser = ArgumentParser(
			description="Demo for embedding data via LDA",
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
			metavar='BOOL',
			nargs='?',
			type=bool,
			help="Sort the files?",
			default=False,
			const=True
			)
		
		parser.add_argument(
			'--delay', '-t',
			metavar='INT',
			type=int,
			help="Animation delay",
			default=1000
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


	def main(args):
		# Load the data
		files = ifile(args.data, args.sort)	
		frames = pykitti.utils.yield_velo_scans(files)
		fig = create_figure()

		@mlab.animate(delay=args.delay)
		def animation():
			plot = None
			X = next(frames, [])
			while len(X):
				plot = vertices(X, X[:,3], fig, plot)
				fig.render() 
				yield
				X = next(frames, [])

		animator = animation()
		mlab.show()
		return 0


	parser = init_argparse()
	args, _ = parser.parse_known_args()
	if len(sys.argv) == 1:
		args = validate(args)
	exit(main(args))